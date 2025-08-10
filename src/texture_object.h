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

#ifndef TEXTURE_OBJECT_H
#define TEXTURE_OBJECT_H

#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include <list>
#include <unordered_map>

class QImage;
class TextureObject;

typedef std::shared_ptr<TextureObject> TextureObjectHandle;

struct TextureSize {
    int w;
    int h;
};

struct TextureImageInfo {
    std::string path;
    TextureSize size;
};

/* wrapper to an array of textures */
struct TextureObject {

    std::vector<TextureImageInfo> texInfoVec;
    std::vector<uint32_t> texNameVec;

    TextureObject();
    ~TextureObject();

    TextureObject(const TextureObject &) = delete;
    TextureObject &operator=(const TextureObject &) = delete;

    /* Add QImage ref to the texture object */
    bool AddImage(std::string path);

    /* Binds the texture at index i */
    void Bind(int i);

    /* Releases the texture i, without unbinding it if it is bound */
    void Release(int i);

    int TextureWidth(std::size_t i);
    int TextureHeight(std::size_t i);

    int MaxSize();
    std::vector<TextureSize> GetTextureSizes();

    std::size_t ArraySize();

    int64_t TextureArea(std::size_t i);
    double GetResolutionInMegaPixels();

    std::vector<std::pair<double, double>> ComputeRelativeSizes();

    // GPU texture cache stats API
    struct CacheStats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t bytesEvicted = 0;
    };
    void ResetCacheStats();
    CacheStats GetCacheStats() const;

    // Cache budget configuration (in GB/bytes)
    void SetCacheBudgetGB(double gigabytes);
    double GetCacheBudgetGB() const;
    void SetCacheBudgetBytes(uint64_t bytes);
    uint64_t GetCacheBudgetBytes() const;
    uint64_t GetCurrentCacheBytes() const;

private:
    // LRU cache of GPU textures by index
    void EvictIfNeeded(uint64_t bytesToAdd);
    void TouchLRU(std::size_t idx);
    void RemoveFromLRU(std::size_t idx);

    uint64_t cacheBudgetBytes_ = 0;      // GPU memory budget for cached textures
    uint64_t currentCacheBytes_ = 0;     // Currently used GPU memory by cached textures
    std::vector<uint64_t> texBytesVec_;  // Tracked bytes per texture index
    std::list<std::size_t> lruList_;     // Most-recently-used at front, LRU at back
    std::unordered_map<std::size_t, std::list<std::size_t>::iterator> lruMap_;

    // Cache stats counters
    uint64_t cacheHits_ = 0;
    uint64_t cacheMisses_ = 0;
    uint64_t cacheEvictions_ = 0;
    uint64_t bytesEvicted_ = 0;
};

/* Vertically mirrors a QImage in-place, useful to match the OpenGL convention
 * for texture data storage */
void Mirror(QImage& img);


#endif // TEXTURE_OBJECT_H
