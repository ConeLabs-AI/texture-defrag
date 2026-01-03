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
#include <chrono>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <QImageReader>
#include <QImage>
#include <QFileInfo>
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
        ReleaseAll();
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
        currentROIs_.push_back(vcg::Box2d());
        rawTileInfo_.push_back(RawTileInfo{});
        pageTables_.push_back(PageTable{});
        return true;
    } else return false;
}

void TextureObject::InitVirtualTexturing()
{
    EnsureVTInitialized();
}

int TextureObject::PageTableWidth(int i) const
{
    if (i < 0 || i >= (int)pageTables_.size()) return 0;
    return pageTables_[i].tilesX;
}

int TextureObject::PageTableHeight(int i) const
{
    if (i < 0 || i >= (int)pageTables_.size()) return 0;
    return pageTables_[i].tilesY;
}

int TextureObject::TileCacheLayers() const
{
    return tileArrayLayers_;
}

void TextureObject::EnsureVTInitialized()
{
    if (vtInitialized_) return;
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();

    const int ts = TileSize();
    const int border = TileBorder();
    const int phys = PhysicalTileSize();
    const uint64_t bytesPerTile = (uint64_t)phys * (uint64_t)phys * 4ull;

    GLint maxArrayLayersGL = 0;
    glFuncs->glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxArrayLayersGL);
    if (maxArrayLayersGL <= 0) maxArrayLayersGL = 2048;

    int layersByBudget = 1024;
    if (cacheBudgetBytes_ > 0) {
        layersByBudget = (int)std::max<uint64_t>(1, cacheBudgetBytes_ / bytesPerTile);
    }
    tileArrayLayers_ = std::max(1, std::min(layersByBudget, (int)maxArrayLayersGL));

    // Clamp budget to what we can actually allocate as array layers.
    cacheBudgetBytes_ = (uint64_t)tileArrayLayers_ * bytesPerTile;

    glFuncs->glGenTextures(1, &tileArrayTex_);
    glFuncs->glBindTexture(GL_TEXTURE_2D_ARRAY, tileArrayTex_);
    glFuncs->glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, phys, phys, tileArrayLayers_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glFuncs->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFuncs->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFuncs->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glFuncs->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFuncs->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
    glFuncs->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
    CHECK_GL_ERROR();

    freeLayers_.clear();
    freeLayers_.reserve(tileArrayLayers_);
    for (int l = tileArrayLayers_ - 1; l >= 0; --l) freeLayers_.push_back(l);

    rawTileInfo_.resize(texInfoVec.size());
    pageTables_.resize(texInfoVec.size());

    const uint32_t INVALID = 0xFFFFFFFFu;
    for (int i = 0; i < (int)texInfoVec.size(); ++i) {
        RawTileInfo info;
        info.rawPath = TextureConversion::GetRawTilePath(texInfoVec[i].path);
        std::ifstream in(info.rawPath, std::ios::binary);
        if (!in) {
            info.valid = false;
            rawTileInfo_[i] = info;
            fallbacks_++;
            continue;
        }
        in.read(reinterpret_cast<char*>(&info.width), sizeof(int));
        in.read(reinterpret_cast<char*>(&info.height), sizeof(int));
        in.read(reinterpret_cast<char*>(&info.tileSize), sizeof(int));
        if (!in || info.width <= 0 || info.height <= 0 || info.tileSize != ts) {
            info.valid = false;
            rawTileInfo_[i] = info;
            fallbacks_++;
            continue;
        }
        info.tilesX = (info.width + ts - 1) / ts;
        info.tilesY = (info.height + ts - 1) / ts;
        info.valid = true;
        rawTileInfo_[i] = info;

        PageTable pt;
        pt.tilesX = info.tilesX;
        pt.tilesY = info.tilesY;
        glFuncs->glGenTextures(1, &pt.texName);
        glFuncs->glBindTexture(GL_TEXTURE_2D, pt.texName);
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        std::vector<uint32_t> initData((size_t)pt.tilesX * (size_t)pt.tilesY, INVALID);
        glFuncs->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glFuncs->glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, pt.tilesX, pt.tilesY, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, initData.data());
        CHECK_GL_ERROR();

        pageTables_[i] = pt;
    }

    vtInitialized_ = true;
    LOG_INFO << "[VT] Initialized: tileSize=" << ts << " border=" << border << " phys=" << phys
             << " layers=" << tileArrayLayers_ << " budgetMB=" << (double(cacheBudgetBytes_) / (1024.0 * 1024.0));
}

void TextureObject::SetPageTableEntry(int imageIdx, int tx, int ty, uint32_t value)
{
    if (!vtInitialized_) return;
    if (imageIdx < 0 || imageIdx >= (int)pageTables_.size()) return;
    auto &pt = pageTables_[imageIdx];
    if (pt.texName == 0) return;
    if (tx < 0 || ty < 0 || tx >= pt.tilesX || ty >= pt.tilesY) return;
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    glFuncs->glBindTexture(GL_TEXTURE_2D, pt.texName);
    glFuncs->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glFuncs->glTexSubImage2D(GL_TEXTURE_2D, 0, tx, ty, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &value);
    CHECK_GL_ERROR();
}

void TextureObject::EvictIfNeeded(uint64_t bytesToAdd, const std::unordered_set<TileKey, TileKeyHasher>& pinned)
{
    if (cacheBudgetBytes_ == 0) return;
    const uint64_t bytesPerTile = (uint64_t)PhysicalTileSize() * (uint64_t)PhysicalTileSize() * 4ull;
    const uint32_t INVALID = 0xFFFFFFFFu;

    while (currentCacheBytes_ + bytesToAdd > cacheBudgetBytes_) {
        if (tileLRUList_.empty()) break;

        // Find an evictable victim (skip pinned)
        auto itLRU = tileLRUList_.end();
        do {
            if (itLRU == tileLRUList_.begin()) { itLRU = tileLRUList_.end(); break; }
            --itLRU;
        } while (itLRU != tileLRUList_.begin() && pinned.count(*itLRU) > 0);

        if (itLRU == tileLRUList_.end() || pinned.count(*itLRU) > 0) {
            // Nothing evictable
            break;
        }

        TileKey victimKey = *itLRU;
        auto it = tileCache_.find(victimKey);
        if (it == tileCache_.end()) {
            tileLRUList_.erase(itLRU);
            continue;
        }

        int victimLayer = it->second.layer;
        tileCache_.erase(it);
        tileLRUList_.erase(itLRU);

        // Invalidate page table entry
        SetPageTableEntry(victimKey.imageIdx, victimKey.tx, victimKey.ty, INVALID);

        freeLayers_.push_back(victimLayer);
        cacheEvictions_++;
        bytesEvicted_ += bytesPerTile;
        if (currentCacheBytes_ >= bytesPerTile) currentCacheBytes_ -= bytesPerTile;
    }
}

void TextureObject::TouchTileLRU(const TileKey& key)
{
    auto it = tileCache_.find(key);
    if (it != tileCache_.end()) {
        tileLRUList_.erase(it->second.lruIt);
        tileLRUList_.push_front(key);
        it->second.lruIt = tileLRUList_.begin();
    }
}

bool TextureObject::LoadTileWithBorderRGBA(int imageIdx, int tx, int tyTopDown, std::vector<uint32_t>& out, uint64_t& bytesReadOut)
{
    bytesReadOut = 0;
    if (imageIdx < 0 || imageIdx >= (int)rawTileInfo_.size()) return false;
    const auto &info = rawTileInfo_[imageIdx];
    if (!info.valid) return false;

    const int ts = info.tileSize;
    const int border = TileBorder();
    const int phys = ts + 2 * border;
    out.assign((size_t)phys * (size_t)phys, 0u);

    std::ifstream in(info.rawPath, std::ios::binary);
    if (!in) return false;

    const std::streamoff headerBytes = 3 * sizeof(int);
    const int x0 = tx * ts;
    const int y0 = tyTopDown * ts;

    auto t_disk_start = std::chrono::high_resolution_clock::now();
    for (int dy = 0; dy < phys; ++dy) {
        int srcY = y0 - border + dy;
        if (srcY < 0) srcY = 0;
        if (srcY >= info.height) srcY = info.height - 1;

        int tileY = srcY / ts;
        int inY = srcY - tileY * ts;

        int srcXmin = x0 - border;
        int srcXmax = x0 + ts + border - 1;

        int clampedStart = std::max(0, srcXmin);
        int clampedEnd = std::min(info.width - 1, srcXmax);
        if (clampedStart > clampedEnd) continue;

        uint32_t* destRow = &out[(size_t)dy * (size_t)phys];

        int leftPad = clampedStart - srcXmin;
        int rightPad = srcXmax - clampedEnd;

        int tileXStart = clampedStart / ts;
        int tileXEnd = clampedEnd / ts;
        for (int tileX = tileXStart; tileX <= tileXEnd; ++tileX) {
            int segStart = std::max(clampedStart, tileX * ts);
            int segEnd = std::min(clampedEnd, tileX * ts + ts - 1);
            int len = segEnd - segStart + 1;
            int inX = segStart - tileX * ts;
            int destOff = segStart - srcXmin;

            std::streamoff tileIndex = (std::streamoff(tileY) * info.tilesX + tileX);
            std::streamoff tileBase = headerBytes + tileIndex * (std::streamoff)ts * ts * (std::streamoff)sizeof(uint32_t);
            std::streamoff rowBase = tileBase + (std::streamoff)inY * ts * (std::streamoff)sizeof(uint32_t) + (std::streamoff)inX * (std::streamoff)sizeof(uint32_t);

            in.seekg(rowBase);
            in.read(reinterpret_cast<char*>(&destRow[destOff]), (std::streamsize)len * (std::streamsize)sizeof(uint32_t));
            bytesReadOut += (uint64_t)len * sizeof(uint32_t);
        }

        // Replicate edge pixels into padded region (clamp-to-edge behavior)
        if (leftPad > 0) {
            uint32_t v = destRow[leftPad];
            for (int k = 0; k < leftPad; ++k) destRow[k] = v;
        }
        if (rightPad > 0) {
            int last = phys - 1 - rightPad;
            uint32_t v = destRow[last];
            for (int k = last + 1; k < phys; ++k) destRow[k] = v;
        }
    }
    auto t_disk_end = std::chrono::high_resolution_clock::now();
    diskReadTimeS_ += std::chrono::duration<double>(t_disk_end - t_disk_start).count();

    // Mirror vertically for OpenGL convention (v=0 bottom)
    for (int r = 0; r < phys / 2; ++r) {
        uint32_t* line0 = &out[(size_t)r * (size_t)phys];
        uint32_t* line1 = &out[(size_t)(phys - 1 - r) * (size_t)phys];
        std::swap_ranges(line0, line0 + phys, line1);
    }

    return true;
}

void TextureObject::EnsureTileResident(int i, int tx, int ty)
{
    EnsureVTInitialized();
    ensure(i >= 0 && i < (int)texInfoVec.size());
    if (i < 0 || i >= (int)rawTileInfo_.size() || !rawTileInfo_[i].valid) {
        fallbacks_++;
        return;
    }

    TileKey key{i, tx, ty};
    auto it = tileCache_.find(key);
    if (it != tileCache_.end()) {
        cacheHits_++;
        TouchTileLRU(key);
        return;
    }

    cacheMisses_++;

    const uint64_t bytesNeeded = (uint64_t)PhysicalTileSize() * (uint64_t)PhysicalTileSize() * 4ull;
    EvictIfNeeded(bytesNeeded, pinnedTiles_);

    if (freeLayers_.empty()) {
        // Force evict something (non-pinned) to get a layer.
        EvictIfNeeded(bytesNeeded, pinnedTiles_);
    }
    if (freeLayers_.empty()) {
        LOG_WARN << "[VT] No free tile layers available; cannot load tile.";
        return;
    }

    const uint32_t INVALID = 0xFFFFFFFFu;
    if (tx < 0 || ty < 0 || tx >= pageTables_[i].tilesX || ty >= pageTables_[i].tilesY) {
        SetPageTableEntry(i, tx, ty, INVALID);
        return;
    }

    int layer = freeLayers_.back();
    freeLayers_.pop_back();

    std::vector<uint32_t> buf;
    uint64_t bytesReadLocal = 0;
    bool ok = LoadTileWithBorderRGBA(i, tx, ty, buf, bytesReadLocal);
    if (!ok) {
        // Upload black tile to avoid undefined sampling
        buf.assign((size_t)PhysicalTileSize() * (size_t)PhysicalTileSize(), 0u);
        fallbacks_++;
    }

    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    glFuncs->glBindTexture(GL_TEXTURE_2D_ARRAY, tileArrayTex_);
    glFuncs->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glFuncs->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer,
                            PhysicalTileSize(), PhysicalTileSize(), 1,
                            GL_RGBA, GL_UNSIGNED_BYTE, buf.data());
    CHECK_GL_ERROR();

    // Insert into cache and LRU
    tileLRUList_.push_front(key);
    CacheEntry ent;
    ent.layer = layer;
    ent.lruIt = tileLRUList_.begin();
    tileCache_[key] = ent;

    currentCacheBytes_ += bytesNeeded;
    tilesLoaded_++;
    bytesRead_ += bytesReadLocal;

    SetPageTableEntry(i, tx, ty, (uint32_t)layer);
}

bool TextureObject::IsTileResident(int i, int tx, int ty) const
{
    TileKey key{i, tx, ty};
    return tileCache_.find(key) != tileCache_.end();
}

void TextureObject::BindVirtualTexturing(int i, int tileCacheUnit, int pageTableUnit)
{
    EnsureVTInitialized();
    if (i < 0 || i >= (int)pageTables_.size()) return;
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    glFuncs->glActiveTexture(GL_TEXTURE0 + tileCacheUnit);
    glFuncs->glBindTexture(GL_TEXTURE_2D_ARRAY, tileArrayTex_);
    glFuncs->glActiveTexture(GL_TEXTURE0 + pageTableUnit);
    glFuncs->glBindTexture(GL_TEXTURE_2D, pageTables_[i].texName);
}

void TextureObject::BeginTilePinning(int imageIdx, const std::vector<std::pair<int,int>>& tiles)
{
    pinnedTiles_.clear();
    pinnedTiles_.reserve(tiles.size());
    for (auto &t : tiles) {
        pinnedTiles_.insert(TileKey{imageIdx, t.first, t.second});
    }
}

void TextureObject::EndTilePinning()
{
    pinnedTiles_.clear();
}

void TextureObject::Bind(int i)
{
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    ensure(i >= 0 && i < (int) texInfoVec.size());

    // For legacy/fallback: if we load a full texture, we treat it as a special case.
    // To simplify, we'll just use the old texNameVec[i] but it also consumes budget.
    if (texNameVec[i] == 0) {
        cacheMisses_++;
        QImage img(texInfoVec[i].path.c_str());
        ensure(!img.isNull());
        if ((img.format() != QImage::Format_RGB32) && (img.format() != QImage::Format_ARGB32)) {
            img = img.convertToFormat(QImage::Format_ARGB32);
        }

        const uint64_t bytesNeeded = static_cast<uint64_t>(img.width()) * static_cast<uint64_t>(img.height()) * 4ull;
        EvictIfNeeded(bytesNeeded, pinnedTiles_);

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

        currentCacheBytes_ += bytesNeeded;
        currentROIs_[i].Set(0, 0, 1, 1);
    } else {
        cacheHits_++;
        glFuncs->glBindTexture(GL_TEXTURE_2D, texNameVec[i]);
    }
}

void TextureObject::Release(int i)
{
    ensure(i >= 0 && i < (int) texInfoVec.size());
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    // For now, Release(i) only clears fallback full-texture (if any).
    // Virtual-texture tiles are released globally via ReleaseAll().
    if (i >= 0 && i < (int)texNameVec.size() && texNameVec[i]) {
        glFuncs->glDeleteTextures(1, &texNameVec[i]);
        texNameVec[i] = 0;
    }
    if (i >= 0 && i < (int)currentROIs_.size()) currentROIs_[i] = vcg::Box2d();
}

void TextureObject::ReleaseAll()
{
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();
    // Delete page tables
    for (auto &pt : pageTables_) {
        if (pt.texName) {
            glFuncs->glDeleteTextures(1, &pt.texName);
            pt.texName = 0;
        }
    }
    pageTables_.clear();

    // Delete tile array
    if (tileArrayTex_) {
        glFuncs->glDeleteTextures(1, &tileArrayTex_);
        tileArrayTex_ = 0;
    }

    tileCache_.clear();
    tileLRUList_.clear();
    freeLayers_.clear();
    pinnedTiles_.clear();
    vtInitialized_ = false;
    tileArrayLayers_ = 0;

    for (std::size_t i = 0; i < texNameVec.size(); ++i) {
        if (texNameVec[i]) {
            glFuncs->glDeleteTextures(1, &texNameVec[i]);
            texNameVec[i] = 0;
        }
        currentROIs_[i] = vcg::Box2d();
    }
    currentCacheBytes_ = 0;
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