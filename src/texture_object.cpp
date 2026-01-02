/*******************************************************************************
    Copyright (c) 2021, Andrea Maggiordomo, Paolo Cignoni and Marco Tarini

    This file is part of TextureDefrag, a reference implementation for
    the paper ``Texture Defragmentation for Photo-Reconstructed 3D Models''
    by Andrea Maggiordomo, Paolo Cignoni and Marco Tarini.

    TextureDefrag is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TextureDefrag is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TextureDefrag. If not, see <https://www.gnu.org/licenses/>.
*******************************************************************************/

#include "texture_object.h"
#include "logging.h"
#include "utils.h"
#include "gl_utils.h"
#include "texture_conversion.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <QImageReader>
#include <QImage>
#include <QOpenGLContext>


TextureObject::TextureObject()
{
    SetCacheBudgetGB(8.0);
    LOG_INFO << "Texture GPU cache budget set to " << GetCacheBudgetGB() << " GB (default)";
    ResetCacheStats();
}

TextureObject::~TextureObject()
{
    if (QOpenGLContext::currentContext() != nullptr) {
        for (std::size_t i = 0; i < texNameVec.size(); ++i)
            Release(i);
    }
}

bool TextureObject::AddImage(std::string path)
{
    QFileInfo fi(path.c_str());
    std::string absPath = fi.absoluteFilePath().toStdString();
    QImageReader qir(QString(absPath.c_str()));
    if (qir.canRead()) {
        TextureImageInfo tii = {};
        tii.path = absPath;
        tii.size = { qir.size().width(), qir.size().height() };
        texInfoVec.push_back(tii);
        texNameVec.push_back(0);
        texBytesVec_.push_back(0);
        currentROIs_.push_back(vcg::Box2d()); // Default: empty box means full or none
        return true;
    } else return false;
}

void TextureObject::Bind(int i)
{
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    ensure(i >= 0 && i < (int) texInfoVec.size());

    // Check if already loaded at full resolution
    bool isFull = !currentROIs_[i].IsEmpty() && 
                  currentROIs_[i].min.X() <= 0.0001 && currentROIs_[i].min.Y() <= 0.0001 && 
                  currentROIs_[i].max.X() >= 0.9999 && currentROIs_[i].max.Y() >= 0.9999;

    // load texture from qimage on first use or if only a region was loaded
    if (texNameVec[i] == 0 || !isFull) {
        // If it was a region or not loaded, load full now
        Release(i); 
        cacheMisses_++;
        QImage img(texInfoVec[i].path.c_str());
        ensure(!img.isNull());
        if ((img.format() != QImage::Format_RGB32) && (img.format() != QImage::Format_ARGB32)) {
            QImage glimg = img.convertToFormat(QImage::Format_ARGB32);
            img = glimg;
        }

        // Before allocating, ensure we have space within the GPU cache budget
        const uint64_t bytesNeeded = static_cast<uint64_t>(img.width()) * static_cast<uint64_t>(img.height()) * 4ull;
        EvictIfNeeded(bytesNeeded);

        glFuncs->glGenTextures(1, &texNameVec[i]);

        Mirror(img);
        glFuncs->glBindTexture(GL_TEXTURE_2D, texNameVec[i]);
        glFuncs->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

        glFuncs->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.width(), img.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, img.constBits());
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        CHECK_GL_ERROR();

        // Track memory usage and LRU
        texBytesVec_[i] = bytesNeeded;
        currentCacheBytes_ += bytesNeeded;
        currentROIs_[i].Set(vcg::Point2d(0, 0), vcg::Point2d(1, 1));
        TouchLRU(i);
    }
    else {
        cacheHits_++;
        glFuncs->glBindTexture(GL_TEXTURE_2D, texNameVec[i]);
        TouchLRU(i);
        CHECK_GL_ERROR();
    }
}

void TextureObject::BindRegion(int i, vcg::Box2d roi)
{
    ensure(i >= 0 && i < (int) texInfoVec.size());
    std::string rawPath = TextureConversion::GetRawTilePath(texInfoVec[i].path);
    std::ifstream in(rawPath, std::ios::binary);
    if (!in) {
        fallbacks_++;
        Bind(i);
        return;
    }

    int width, height, tileSize;
    in.read(reinterpret_cast<char*>(&width), sizeof(int));
    in.read(reinterpret_cast<char*>(&height), sizeof(int));
    in.read(reinterpret_cast<char*>(&tileSize), sizeof(int));

    // Correctly map OpenGL [0,1] (bottom-up) to Image space (top-down)
    // OpenGL v=0 is bottom of image, v=1 is top of image.
    // Image space y=0 is top of image, y=height is bottom.
    double imgY_top_f = (1.0 - roi.max.Y()) * height;
    double imgY_bottom_f = (1.0 - roi.min.Y()) * height;

    int ty0 = std::max(0, (int)std::floor(imgY_top_f / tileSize));
    int ty1 = std::min((height + tileSize - 1) / tileSize - 1, (int)std::floor(imgY_bottom_f / tileSize));
    int tx0 = std::max(0, (int)std::floor(roi.min.X() * width / tileSize));
    int tx1 = std::min((width + tileSize - 1) / tileSize - 1, (int)std::floor(roi.max.X() * width / tileSize));

    int loadImgX0 = tx0 * tileSize;
    int loadImgY0 = ty0 * tileSize;
    int loadImgX1 = std::min(width, (tx1 + 1) * tileSize);
    int loadImgY1 = std::min(height, (ty1 + 1) * tileSize);
    
    int loadW = loadImgX1 - loadImgX0;
    int loadH = loadImgY1 - loadImgY0;

    // The actual ROI we are loading (aligned to tiles, bottom-up for the shader)
    vcg::Box2d loadROI;
    loadROI.min = vcg::Point2d((double)loadImgX0 / width, 1.0 - (double)loadImgY1 / height);
    loadROI.max = vcg::Point2d((double)loadImgX1 / width, 1.0 - (double)loadImgY0 / height);

    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();

    // Check if current ROI is sufficient
    if (texNameVec[i] != 0 && !currentROIs_[i].IsEmpty()) {
        if (currentROIs_[i].min.X() <= loadROI.min.X() + 1e-7 &&
            currentROIs_[i].min.Y() <= loadROI.min.Y() + 1e-7 &&
            currentROIs_[i].max.X() >= loadROI.max.X() - 1e-7 &&
            currentROIs_[i].max.Y() >= loadROI.max.Y() - 1e-7)
        {
            cacheHits_++;
            glFuncs->glBindTexture(GL_TEXTURE_2D, texNameVec[i]);
            TouchLRU(i);
            return;
        }
    }

    // Need to load new ROI
    cacheMisses_++;
    uint64_t bytesNeeded = (uint64_t)loadW * loadH * 4;
    EvictIfNeeded(bytesNeeded);

    if (texNameVec[i] == 0) {
        glFuncs->glGenTextures(1, &texNameVec[i]);
    }
    
    glFuncs->glBindTexture(GL_TEXTURE_2D, texNameVec[i]);
    glFuncs->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    std::vector<uint32_t> roiBuffer(loadW * loadH, 0);
    int tilesX = (width + tileSize - 1) / tileSize;

    auto t_disk_start = std::chrono::high_resolution_clock::now();
    uint64_t tilesInThisROI = 0;
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            std::streamoff tileBase = 3 * sizeof(int) + (std::streamoff(ty) * tilesX + tx) * tileSize * tileSize * sizeof(uint32_t);
            
            int targetX = tx * tileSize - loadImgX0;
            int targetY = ty * tileSize - loadImgY0;
            
            // Calculate how much of this tile's width actually fits in our ROI
            int readW = std::min(tileSize, loadW - targetX);
            if (readW <= 0) continue;

            tilesInThisROI++;
            for (int r = 0; r < tileSize; ++r) {
                int roiY = targetY + r;
                if (roiY < 0 || roiY >= loadH) continue;
                
                // Seek to the specific row in this specific tile
                in.seekg(tileBase + std::streamoff(r) * tileSize * sizeof(uint32_t));
                in.read(reinterpret_cast<char*>(&roiBuffer[roiY * loadW + targetX]), readW * sizeof(uint32_t));
            }
        }
    }
    auto t_disk_end = std::chrono::high_resolution_clock::now();
    
    tilesLoaded_ += tilesInThisROI;
    bytesRead_ += (uint64_t)tilesInThisROI * tileSize * tileSize * 4;
    diskReadTimeS_ += std::chrono::duration<double>(t_disk_end - t_disk_start).count();

    // Mirror the patch vertically so it's compatible with OpenGL convention (v=0 at bottom)
    for (int r = 0; r < loadH / 2; ++r) {
        uint32_t* line0 = &roiBuffer[r * loadW];
        uint32_t* line1 = &roiBuffer[(loadH - 1 - r) * loadW];
        std::swap_ranges(line0, line0 + loadW, line1);
    }

    glFuncs->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, loadW, loadH, 0, GL_RGBA, GL_UNSIGNED_BYTE, roiBuffer.data());
    glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    CHECK_GL_ERROR();

    texBytesVec_[i] = bytesNeeded;
    currentCacheBytes_ += bytesNeeded;
    currentROIs_[i] = loadROI;
    TouchLRU(i);
}

vcg::Box2d TextureObject::GetCurrentROI(int i) const
{
    ensure(i >= 0 && i < (int) currentROIs_.size());
    return currentROIs_[i];
}

void TextureObject::Release(int i)
{
    ensure(i >= 0 && i < (int) texInfoVec.size());
    if (texNameVec[i]) {
        OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
        glFuncs->glDeleteTextures(1, &texNameVec[i]);
        texNameVec[i] = 0;
        // Update memory tracking and LRU map
        if (texBytesVec_.size() > static_cast<size_t>(i)) {
            currentCacheBytes_ -= texBytesVec_[i];
            texBytesVec_[i] = 0;
        }
        currentROIs_[i] = vcg::Box2d(); // Reset ROI tracking
        RemoveFromLRU(i);
    }
}

void TextureObject::ReleaseAll()
{
    for (std::size_t i = 0; i < texNameVec.size(); ++i) {
        Release(static_cast<int>(i));
    }
}

int TextureObject::TextureWidth(std::size_t i)
{
    ensure(i < texInfoVec.size());
    return texInfoVec[i].size.w;
}

int TextureObject::TextureHeight(std::size_t i)
{
    ensure(i < texInfoVec.size());
    return texInfoVec[i].size.h;
}

int TextureObject::MaxSize()
{
    int maxsz = 0;
    for (unsigned i = 0; i < ArraySize(); ++i) {
        maxsz = std::max(maxsz, TextureWidth(i));
        maxsz = std::max(maxsz, TextureHeight(i));
    }
    return maxsz;
}

std::vector<TextureSize> TextureObject::GetTextureSizes()
{
    std::vector<TextureSize> texszVec;
    for (unsigned i = 0; i < ArraySize(); ++i)
        texszVec.push_back({TextureWidth(i), TextureHeight(i)});
    return texszVec;

}

std::size_t TextureObject::ArraySize()
{
    return texInfoVec.size();
}

int64_t TextureObject::TextureArea(std::size_t i)
{
    ensure(i < ArraySize());
    return ((int64_t) TextureWidth(i)) * TextureHeight(i);
}

double TextureObject::GetResolutionInMegaPixels()
{
    int64_t totArea = 0;
    for (unsigned i = 0; i < ArraySize(); ++i) {
        totArea += TextureArea(i);
    }
    return totArea / 1000000.0;
}

std::vector<std::pair<double, double>> TextureObject::ComputeRelativeSizes()
{
    std::vector<TextureSize> texSizeVec = GetTextureSizes();
    int maxsz = 0;
    for (auto tsz : texSizeVec) {
        maxsz = std::max(maxsz, tsz.h);
        maxsz = std::max(maxsz, tsz.w);
    }
    std::vector<std::pair<double, double>> trs;
    for (auto tsz : texSizeVec) {
        double rw = tsz.w / (double) maxsz;
        double rh = tsz.h / (double) maxsz;
        trs.push_back(std::make_pair(rw, rh));
    }
    return trs;
}

void Mirror(QImage& img)
{
    const int height = img.height();
    const int width = img.width();

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < height / 2; ++i) {
        QRgb *line0 = (QRgb *) img.scanLine(i);
        QRgb *line1 = (QRgb *) img.scanLine(height - 1 - i);
        std::swap_ranges(line0, line0 + width, line1);
    }
}

void TextureObject::SetCacheBudgetGB(double gigabytes)
{
    if (gigabytes <= 0) {
        cacheBudgetBytes_ = 0;
    } else {
        cacheBudgetBytes_ = static_cast<uint64_t>(gigabytes * 1024.0 * 1024.0 * 1024.0);
    }
}

double TextureObject::GetCacheBudgetGB() const
{
    return static_cast<double>(cacheBudgetBytes_) / (1024.0 * 1024.0 * 1024.0);
}

void TextureObject::SetCacheBudgetBytes(uint64_t bytes)
{
    cacheBudgetBytes_ = bytes;
}

uint64_t TextureObject::GetCacheBudgetBytes() const
{
    return cacheBudgetBytes_;
}

uint64_t TextureObject::GetCurrentCacheBytes() const
{
    return currentCacheBytes_;
}

void TextureObject::EvictIfNeeded(uint64_t bytesToAdd)
{
    if (cacheBudgetBytes_ == 0) return; // unlimited
    // Evict while exceeding budget
    while (currentCacheBytes_ + bytesToAdd > cacheBudgetBytes_) {
        if (lruList_.empty()) break;
        std::size_t victim = lruList_.back();
        lruList_.pop_back();
        lruMap_.erase(victim);
        if (victim < texNameVec.size() && texNameVec[victim] != 0) {
            OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
            glFuncs->glDeleteTextures(1, &texNameVec[victim]);
            texNameVec[victim] = 0;
            if (victim < currentROIs_.size()) {
                currentROIs_[victim] = vcg::Box2d();
            }
            if (victim < texBytesVec_.size()) {
                cacheEvictions_++;
                bytesEvicted_ += texBytesVec_[victim];
                currentCacheBytes_ -= texBytesVec_[victim];
                texBytesVec_[victim] = 0;
            }
        }
    }
}

void TextureObject::TouchLRU(std::size_t idx)
{
    auto it = lruMap_.find(idx);
    if (it != lruMap_.end()) {
        lruList_.erase(it->second);
    }
    lruList_.push_front(idx);
    lruMap_[idx] = lruList_.begin();
}

void TextureObject::RemoveFromLRU(std::size_t idx)
{
    auto it = lruMap_.find(idx);
    if (it != lruMap_.end()) {
        lruList_.erase(it->second);
        lruMap_.erase(it);
    }
}

void TextureObject::ResetCacheStats() {
    cacheHits_ = 0;
    cacheMisses_ = 0;
    cacheEvictions_ = 0;
    bytesEvicted_ = 0;
    tilesLoaded_ = 0;
    bytesRead_ = 0;
    diskReadTimeS_ = 0.0;
    fallbacks_ = 0;
}

TextureObject::CacheStats TextureObject::GetCacheStats() const {
    CacheStats s;
    s.hits = cacheHits_;
    s.misses = cacheMisses_;
    s.evictions = cacheEvictions_;
    s.bytesEvicted = bytesEvicted_;
    s.tilesLoaded = tilesLoaded_;
    s.bytesRead = bytesRead_;
    s.diskReadTimeS = diskReadTimeS_;
    s.fallbacks = fallbacks_;
    return s;
}