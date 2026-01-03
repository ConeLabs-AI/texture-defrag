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
#include <unordered_set>
#include <vcg/space/box2.h>

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

    /**
     * Initializes the virtual texturing resources (tile cache + per-texture page tables).
     * Safe to call multiple times; will no-op after first init.
     */
    void InitVirtualTexturing();

    /**
     * Ensures a specific 256x256 tile (tx,ty) is resident in the GPU physical cache.
     * Coordinates are in .rawtile tile space (top-down ty as stored on disk).
     * The tile is uploaded with a 2-pixel border to support filtering across tile edges.
     */
    void EnsureTileResident(int i, int tx, int ty);

    /**
     * Returns true iff the given tile is currently resident in the GPU cache.
     */
    bool IsTileResident(int i, int tx, int ty) const;

    /**
     * Bind virtual texturing resources for source texture i.
     * - Binds the global tile cache array to tileCacheUnit
     * - Binds the per-texture page table to pageTableUnit
     */
    void BindVirtualTexturing(int i, int tileCacheUnit = 0, int pageTableUnit = 1);

    /**
     * Pin a set of tiles so eviction will not remove them during a loading phase.
     * Intended for the renderer's pass scheduler.
     */
    void BeginTilePinning(int imageIdx, const std::vector<std::pair<int,int>>& tiles);
    void EndTilePinning();

    /* Releases the texture i, without unbinding it if it is bound */
    void Release(int i);
    /* Releases all textures */
    void ReleaseAll();

    int TextureWidth(std::size_t i);
    int TextureHeight(std::size_t i);
    int TileSize() const { return 256; }
    int TileBorder() const { return 2; }
    int PhysicalTileSize() const { return TileSize() + 2 * TileBorder(); }
    int PageTableWidth(int i) const;
    int PageTableHeight(int i) const;
    int TileCacheLayers() const;

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

        // Tiled I/O stats
        uint64_t tilesLoaded = 0;
        uint64_t bytesRead = 0;
        double diskReadTimeS = 0.0;
        uint64_t fallbacks = 0;
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
    struct TileKey {
        int imageIdx;
        int tx, ty;
        bool operator==(const TileKey& other) const {
            return imageIdx == other.imageIdx && tx == other.tx && ty == other.ty;
        }
    };

    struct TileKeyHasher {
        std::size_t operator()(const TileKey& k) const {
            return std::hash<int>()(k.imageIdx) ^ (std::hash<int>()(k.tx) << 1) ^ (std::hash<int>()(k.ty) << 2);
        }
    };

    struct PageTable {
        uint32_t texName = 0; // GL texture name (R32UI)
        int tilesX = 0;
        int tilesY = 0;
    };

    struct RawTileInfo {
        std::string rawPath;
        int width = 0;
        int height = 0;
        int tileSize = 0;
        int tilesX = 0;
        int tilesY = 0;
        bool valid = false;
    };

    struct CacheEntry {
        int layer = -1;
        std::list<TileKey>::iterator lruIt;
    };

    // Virtual texturing internals
    void EnsureVTInitialized();
    void EvictIfNeeded(uint64_t bytesToAdd, const std::unordered_set<TileKey, TileKeyHasher>& pinned);
    void TouchTileLRU(const TileKey& key);
    void SetPageTableEntry(int imageIdx, int tx, int ty, uint32_t value);
    bool LoadTileWithBorderRGBA(int imageIdx, int tx, int tyTopDown, std::vector<uint32_t>& out, uint64_t& bytesReadOut);

    uint64_t cacheBudgetBytes_ = 0;      // GPU memory budget for cached textures
    uint64_t currentCacheBytes_ = 0;     // Currently used GPU memory by cached textures

    // Physical cache: single GL_TEXTURE_2D_ARRAY with fixed-size bordered tiles
    uint32_t tileArrayTex_ = 0;
    int tileArrayLayers_ = 0;
    std::vector<int> freeLayers_;

    // LRU for resident tiles -> array layer
    std::list<TileKey> tileLRUList_; // MRU front
    std::unordered_map<TileKey, CacheEntry, TileKeyHasher> tileCache_;

    // Per-source-texture page table textures
    std::vector<PageTable> pageTables_;
    std::vector<RawTileInfo> rawTileInfo_;
    bool vtInitialized_ = false;
    std::unordered_set<TileKey, TileKeyHasher> pinnedTiles_;

    // Cache stats counters
    uint64_t cacheHits_ = 0;
    uint64_t cacheMisses_ = 0;
    uint64_t cacheEvictions_ = 0;
    uint64_t bytesEvicted_ = 0;

    // Tiled I/O stats
    uint64_t tilesLoaded_ = 0;
    uint64_t bytesRead_ = 0;
    double diskReadTimeS_ = 0.0;
    uint64_t fallbacks_ = 0;

    std::vector<vcg::Box2d> currentROIs_; // Track current ROI loaded for each texture index (fallback for Bind/BindRegion)
};

/* Vertically mirrors a QImage in-place, useful to match the OpenGL convention
 * for texture data storage */
void Mirror(QImage& img);


#endif // TEXTURE_OBJECT_H