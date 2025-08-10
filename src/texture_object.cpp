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

#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <QImageReader>
#include <QImage>


TextureObject::TextureObject()
{
    SetCacheBudgetGB(8.0);
    LOG_INFO << "Texture GPU cache budget set to " << GetCacheBudgetGB() << " GB (default)";
    ResetCacheStats();
}

TextureObject::~TextureObject()
{
    for (std::size_t i = 0; i < texNameVec.size(); ++i)
        Release(i);
}

bool TextureObject::AddImage(std::string path)
{
    QImageReader qir(QString(path.c_str()));
    if (qir.canRead()) {
        TextureImageInfo tii = {};
        tii.path = path;
        tii.size = { qir.size().width(), qir.size().height() };
        texInfoVec.push_back(tii);
        texNameVec.push_back(0);
        texBytesVec_.push_back(0);
        return true;
    } else return false;
}

void TextureObject::Bind(int i)
{
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    ensure(i >= 0 && i < (int) texInfoVec.size());
    // load texture from qimage on first use
    if (texNameVec[i] == 0) {
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
        CHECK_GL_ERROR();

        // Track memory usage and LRU
        texBytesVec_[i] = bytesNeeded;
        currentCacheBytes_ += bytesNeeded;
        TouchLRU(i);
    }
    else {
        cacheHits_++;
        glFuncs->glBindTexture(GL_TEXTURE_2D, texNameVec[i]);
        TouchLRU(i);
        CHECK_GL_ERROR();
    }
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
        RemoveFromLRU(i);
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
}

TextureObject::CacheStats TextureObject::GetCacheStats() const {
    CacheStats s;
    s.hits = cacheHits_;
    s.misses = cacheMisses_;
    s.evictions = cacheEvictions_;
    s.bytesEvicted = bytesEvicted_;
    return s;
}