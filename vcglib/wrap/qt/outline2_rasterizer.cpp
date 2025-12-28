#include <wrap/qt/outline2_rasterizer.h>
#include <wrap/qt/col_qt_convert.h>
#include <vcg/space/color4.h>

#include <fstream>
#include <unordered_map>
#include <list>
#include <mutex>
#include <cmath>
#include <memory>
#include <chrono>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace vcg;
using namespace std;

namespace {
// Composite key for cache entries
struct CacheKey {
    uint64_t pointsHash;
    uint32_t scaleQ;       // quantized scale (e.g., 1e-5)
    uint16_t rotationNum;
    uint16_t baseRastI;
    uint16_t gutterWidth;  // effective gutter used in rendering (after doubling)
    bool operator==(const CacheKey& o) const {
        return pointsHash == o.pointsHash && scaleQ == o.scaleQ && rotationNum == o.rotationNum
            && baseRastI == o.baseRastI && gutterWidth == o.gutterWidth;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const noexcept {
        // simple 64-bit mixing
        uint64_t h = k.pointsHash;
        auto mix = [&](uint64_t v) {
            v ^= v >> 33;
            v *= 0xff51afd7ed558ccdULL;
            v ^= v >> 33;
            v *= 0xc4ceb9fe1a85ec53ULL;
            v ^= v >> 33;
            return v;
        };
        h ^= mix((uint64_t(k.scaleQ) << 32) ^ (uint64_t(k.rotationNum) << 16) ^ k.baseRastI);
        h ^= mix(uint64_t(k.gutterWidth));
        return size_t(h);
    }
};

struct CacheValue {
    // Base rasterization grid for (points, base orientation, scale, gutter).
    // We generate 90° rotations on demand per call.
    shared_ptr<BitGrid> baseGrid;
    size_t bytes; 
};

static size_t g_cacheMaxBytes = (size_t)15ULL << 30; // default 15GB
static thread_local size_t g_cacheCurrBytes = 0;
static std::atomic<size_t> g_cacheGlobalTotalBytes{0}; // Global atomic to track total usage across all threads
static thread_local std::list<CacheKey> g_lru;
static thread_local std::unordered_map<CacheKey, pair<std::list<CacheKey>::iterator, CacheValue>, CacheKeyHash> g_cache;

static std::mutex g_statsMtx;
static thread_local QtOutline2Rasterizer::CacheStats g_stats; 

// Thread-local buffers to avoid frequent allocations in parallel packing
static thread_local std::unique_ptr<QImage> t_sharedImageBuffer;

inline uint32_t quantizeScale(float s) {
    double q = std::round(double(s) * 1e5);
    if (q < 0) q = 0;
    if (q > double(std::numeric_limits<uint32_t>::max())) q = double(std::numeric_limits<uint32_t>::max());
    return uint32_t(q);
}

inline uint64_t hashPoints(const vector<Point2f>& pts) {
    // Hash raw float bits to be exact across attempts; order matters.
    uint64_t h = 1469598103934665603ULL; // FNV offset
    const uint64_t prime = 1099511628211ULL;
    for (const auto& p : pts) {
        uint32_t bx, by;
        static_assert(sizeof(float) == 4, "float must be 32-bit");
        memcpy(&bx, &p.X(), 4);
        memcpy(&by, &p.Y(), 4);
        h ^= bx; h *= prime;
        h ^= by; h *= prime;
    }
    return h;
}

inline void lruTouch(const CacheKey& key) {
    auto it = g_cache.find(key);
    if (it == g_cache.end()) return;
    g_lru.erase(it->second.first);
    g_lru.push_front(key);
    it->second.first = g_lru.begin();
}

inline void lruInsert(const CacheKey& key, CacheValue val) {
    g_lru.push_front(key);
    size_t valBytes = val.bytes;
    g_cacheCurrBytes += valBytes;
    g_cacheGlobalTotalBytes += valBytes; // Update global atomic
    g_cache[key] = { g_lru.begin(), std::move(val) };
    while (g_cacheCurrBytes > g_cacheMaxBytes && !g_lru.empty()) {
        const CacheKey& oldKey = g_lru.back();
        auto oit = g_cache.find(oldKey);
        if (oit != g_cache.end()) {
            size_t evictedBytes = oit->second.second.bytes;
            g_cacheCurrBytes -= evictedBytes;
            g_cacheGlobalTotalBytes -= evictedBytes; // Update global atomic
            g_cache.erase(oit);
            // Track evictions (no lock needed for thread_local g_stats)
            g_stats.evictions++;
        }
        g_lru.pop_back();
    }
}
} // namespace

void QtOutline2Rasterizer::setCacheMaxBytes(std::size_t bytes) {
    size_t numThreads = 1;
#ifdef _OPENMP
    numThreads = std::max((size_t)1, (size_t)omp_get_max_threads());
#endif
    // Divide total budget by thread count to prevent N * budget OOM
    g_cacheMaxBytes = bytes / numThreads;

    while (g_cacheCurrBytes > g_cacheMaxBytes && !g_lru.empty()) {
        const CacheKey& oldKey = g_lru.back();
        auto oit = g_cache.find(oldKey);
        if (oit != g_cache.end()) {
            size_t evictedBytes = oit->second.second.bytes;
            g_cacheCurrBytes -= evictedBytes;
            g_cacheGlobalTotalBytes -= evictedBytes;
            g_cache.erase(oit);
            g_stats.evictions++;
        }
        g_lru.pop_back();
    }
}

void QtOutline2Rasterizer::clearCache() {
    g_cache.clear();
    g_lru.clear();
    g_cacheGlobalTotalBytes -= g_cacheCurrBytes;
    g_cacheCurrBytes = 0;
}

QtOutline2Rasterizer::CacheStats QtOutline2Rasterizer::statsSnapshot(bool resetCounters) {
    // Note: Since g_stats is now thread_local, this snapshot only represents the calling thread's work.
    // For the global byte count, we use our atomic.
    CacheStats out = g_stats;
    out.bytesCurrent = g_cacheGlobalTotalBytes.load();
    
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = std::max(1, omp_get_max_threads());
#endif
    out.bytesMax = g_cacheMaxBytes * numThreads;

    if (resetCounters) {
        g_stats = CacheStats{};
    }
    return out;
}

void QtOutline2Rasterizer::rasterize(RasterizedOutline2 &poly,
                                 float scale,
                                 int rast_i,
                                 int rotationNum,
                                 int gutterWidth,
                                 bool bypassCache)
{
    auto t_total_start = std::chrono::high_resolution_clock::now();
    g_stats.calls++;

    // since the brush is centered on the outline, a gutter of N pixels requires a pen of 2*N width
    int effectiveGutter = gutterWidth * 2;
    float rotRad = M_PI*2.0f*float(rast_i) / float(rotationNum);

    BitGrid tetrisGrid;
    CacheKey key;
    bool hit = false;

    if (!bypassCache) {
        auto t_lookup_start = std::chrono::high_resolution_clock::now();
        key.pointsHash = hashPoints(poly.getPointsConst());
        key.scaleQ = quantizeScale(scale);
        key.rotationNum = (uint16_t)rotationNum;
        key.baseRastI = (uint16_t)rast_i;
        key.gutterWidth = (uint16_t)effectiveGutter;

        {
            auto it = g_cache.find(key);
            if (it != g_cache.end()) {
                hit = true;
                lruTouch(key);
                auto t_hit_copy_start = std::chrono::high_resolution_clock::now();
                tetrisGrid = *it->second.second.baseGrid; // Deep copy
                auto t_hit_copy_end = std::chrono::high_resolution_clock::now();
                
                g_stats.hits++;
                g_stats.t_hit_copy_s += std::chrono::duration<double>(t_hit_copy_end - t_hit_copy_start).count();
            }
        }
        auto t_lookup_end = std::chrono::high_resolution_clock::now();
        g_stats.t_lookup_s += std::chrono::duration<double>(t_lookup_end - t_lookup_start).count();
    }

    if (!hit) {
        auto t_miss_start = std::chrono::high_resolution_clock::now();
        // Cache miss (or bypass), do the rasterization
        Box2f bb;
        const vector<Point2f>& pointvec = poly.getPointsConst();
        for(size_t i=0;i<pointvec.size();++i) {
            Point2f pp=pointvec[i];
            pp.Rotate(rotRad);
            bb.Add(pp);
        }

        QVector<QPointF> points;
        points.reserve(pointvec.size());
        for (const auto& p : pointvec) {
            points.push_back(QPointF(p.X(), p.Y()));
        }

        int safetyBuffer = 2;
        int sizeX = (int)ceil(bb.DimX()*scale) + effectiveGutter + safetyBuffer;
        int sizeY = (int)ceil(bb.DimY()*scale) + effectiveGutter + safetyBuffer;

        // Optimization: Use thread-local buffer to avoid re-allocations
        if (!t_sharedImageBuffer || t_sharedImageBuffer->width() < sizeX || t_sharedImageBuffer->height() < sizeY) {
            // Allocate a generous buffer (e.g., 4k or slightly more than needed) to minimize future re-allocs
            int allocW = std::max(2048, sizeX);
            int allocH = std::max(2048, sizeY);
            t_sharedImageBuffer = std::make_unique<QImage>(allocW, allocH, QImage::Format_Alpha8);
        }

        // We only clear the region we're going to use
        // QImage doesn't have a clear(QRect) so we use QPainter to clear the ROI
        QPainter painter;
        painter.begin(t_sharedImageBuffer.get());
        painter.setCompositionMode(QPainter::CompositionMode_Source);
        painter.fillRect(0, 0, sizeX, sizeY, Qt::transparent);
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.setBrush(QBrush(Qt::white, Qt::SolidPattern));

        QPen qp(Qt::white);
        qp.setWidth(effectiveGutter);
        qp.setCosmetic(false);
        qp.setJoinStyle(Qt::MiterJoin);
        painter.setPen(qp);

        painter.resetTransform();
        painter.translate(QPointF(-(bb.min.X()*scale) + (effectiveGutter + safetyBuffer)/2.0f, -(bb.min.Y()*scale) + (effectiveGutter + safetyBuffer)/2.0f));
        painter.rotate(math::ToDeg(rotRad));
        painter.scale(scale,scale);

        painter.drawPolygon(QPolygonF(points));
        painter.end();

        // Extract result from the ROI of the shared buffer
        int minX = sizeX, minY = sizeY, maxX = -1, maxY = -1;
        for (int i = 0; i < sizeY; ++i) {
            const uchar *line = t_sharedImageBuffer->constScanLine(i);
            bool hasPixel = false;
            for (int j = 0; j < sizeX; ++j) {
                if (line[j] != 0) {
                    if (j < minX) minX = j;
                    if (j > maxX) maxX = j;
                    hasPixel = true;
                }
            }
            if (hasPixel) {
                if (i < minY) minY = i;
                if (i > maxY) maxY = i;
            }
        }

        if (maxX >= minX) {
            int cropW = (maxX - minX) + 1;
            int cropH = (maxY - minY) + 1;
            tetrisGrid.init(cropW, cropH);
            for (int y = 0; y < cropH; y++) {
                const uchar* line = t_sharedImageBuffer->constScanLine(minY + y);
                for(int x = 0; x < cropW; ++x) {
                    if (line[minX + x] != 0) {
                        tetrisGrid.set(x, y);
                    }
                }
            }
        } else {
            tetrisGrid.clear();
        }

        if (!bypassCache) {
            // Insert into cache
            if (g_cache.find(key) == g_cache.end()) {
                CacheValue val;
                val.baseGrid = make_shared<BitGrid>(tetrisGrid);
                val.bytes = tetrisGrid.data.size() * sizeof(uint64_t);
                lruInsert(key, std::move(val));
                g_stats.inserts++;
            } else {
                 lruTouch(key);
            }
        }
        auto t_miss_end = std::chrono::high_resolution_clock::now();
        g_stats.misses++;
        g_stats.t_miss_raster_s += std::chrono::duration<double>(t_miss_end - t_miss_start).count();

        // Memory Retention Fix: Reset if buffer grew too large (>64MB)
        if (t_sharedImageBuffer && t_sharedImageBuffer->sizeInBytes() > 64 * 1024 * 1024) {
            t_sharedImageBuffer.reset();
        }
    }

    if (tetrisGrid.empty()) { // Handle empty rasterization
        int num_rotations_to_generate = (rotationNum >= 4) ? 4 : 1;
        int rotationOffset = (rotationNum >= 4) ? rotationNum / 4 : 0;
        auto t_rot_start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < num_rotations_to_generate; j++) {
            poly.getGrids(rast_i + rotationOffset*j).clear();
            poly.initFromGrid(rast_i + rotationOffset*j);
        }
        auto t_rot_end = std::chrono::high_resolution_clock::now();
        g_stats.t_rotate_s += std::chrono::duration<double>(t_rot_end - t_rot_start).count();
        
        auto t_total_end = std::chrono::high_resolution_clock::now();
        g_stats.t_total_s += std::chrono::duration<double>(t_total_end - t_total_start).count();
        // Update snapshot counters for the calling thread
        g_stats.bytesCurrent = g_cacheCurrBytes;
        g_stats.bytesMax = g_cacheMaxBytes;
        return;
    }

    // Process the grid (from cache or new) to create 90 degree rotations
    int num_rotations_to_generate = (rotationNum >= 4) ? 4 : 1;
    int rotationOffset = (rotationNum >= 4) ? rotationNum / 4 : 0;
    auto t_rot_start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < num_rotations_to_generate; j++) {
        if (j != 0)  {
            tetrisGrid = rotateGridCWise(tetrisGrid);
        }
        //add the grid to the poly's vector of grids
        poly.getGrids(rast_i + rotationOffset*j) = tetrisGrid;

        //initializes bottom/left/deltaX/deltaY vectors of the poly, for the current rasterization
        poly.initFromGrid(rast_i + rotationOffset*j);
    }
    auto t_rot_end = std::chrono::high_resolution_clock::now();
    g_stats.t_rotate_s += std::chrono::duration<double>(t_rot_end - t_rot_start).count();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    g_stats.t_total_s += std::chrono::duration<double>(t_total_end - t_total_start).count();
    // Update snapshot counters for the calling thread
    g_stats.bytesCurrent = g_cacheCurrBytes;
    g_stats.bytesMax = g_cacheMaxBytes;
}

BitGrid QtOutline2Rasterizer::rotateGridCWise(const BitGrid& inGrid) {
    BitGrid outGrid;
    outGrid.init(inGrid.h, inGrid.w);
    for (int y = 0; y < inGrid.h; y++) {
        for (int x = 0; x < inGrid.w; x++) {
            if (inGrid.get(x, y)) {
                outGrid.set(inGrid.h - 1 - y, x);
            }
        }
    }
    return outGrid;
}