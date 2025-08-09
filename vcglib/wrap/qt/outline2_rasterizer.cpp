#include <wrap/qt/outline2_rasterizer.h>
#include <wrap/qt/col_qt_convert.h>
#include <vcg/space/color4.h>
#include <wrap/qt/col_qt_convert.h>

#include <fstream>
#include <unordered_map>
#include <list>
#include <mutex>
#include <cmath>
#include <memory>

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
    shared_ptr<vector<vector<uint8_t>>> baseGrid;
    size_t bytes; // width * height
};

static std::mutex g_cacheMtx;
static size_t g_cacheMaxBytes = (size_t)15ULL << 30; // default 15GB
static size_t g_cacheCurrBytes = 0;
static std::list<CacheKey> g_lru;
static std::unordered_map<CacheKey, pair<std::list<CacheKey>::iterator, CacheValue>, CacheKeyHash> g_cache;

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
    g_cacheCurrBytes += val.bytes;
    g_cache[key] = { g_lru.begin(), std::move(val) };
    while (g_cacheCurrBytes > g_cacheMaxBytes && !g_lru.empty()) {
        const CacheKey& oldKey = g_lru.back();
        auto oit = g_cache.find(oldKey);
        if (oit != g_cache.end()) {
            g_cacheCurrBytes -= oit->second.second.bytes;
            g_cache.erase(oit);
        }
        g_lru.pop_back();
    }
}
} // namespace

void QtOutline2Rasterizer::setCacheMaxBytes(std::size_t bytes) {
    std::lock_guard<std::mutex> lk(g_cacheMtx);
    g_cacheMaxBytes = bytes;
    while (g_cacheCurrBytes > g_cacheMaxBytes && !g_lru.empty()) {
        const CacheKey& oldKey = g_lru.back();
        auto oit = g_cache.find(oldKey);
        if (oit != g_cache.end()) {
            g_cacheCurrBytes -= oit->second.second.bytes;
            g_cache.erase(oit);
        }
        g_lru.pop_back();
    }
}

void QtOutline2Rasterizer::clearCache() {
    std::lock_guard<std::mutex> lk(g_cacheMtx);
    g_cache.clear();
    g_lru.clear();
    g_cacheCurrBytes = 0;
}

void QtOutline2Rasterizer::rasterize(RasterizedOutline2 &poly,
                                 float scale,
                                 int rast_i,
                                 int rotationNum,
                                 int gutterWidth)
{
    // since the brush is centered on the outline, a gutter of N pixels requires a pen of 2*N width
    int effectiveGutter = gutterWidth * 2;
    float rotRad = M_PI*2.0f*float(rast_i) / float(rotationNum);

    vector<vector<uint8_t>> tetrisGrid;
    {
        CacheKey key;
        key.pointsHash = hashPoints(poly.getPointsConst());
        key.scaleQ = quantizeScale(scale);
        key.rotationNum = (uint16_t)rotationNum;
        key.baseRastI = (uint16_t)rast_i;
        key.gutterWidth = (uint16_t)effectiveGutter;

        bool hit = false;
        {
            std::lock_guard<std::mutex> lk(g_cacheMtx);
            auto it = g_cache.find(key);
            if (it != g_cache.end()) {
                hit = true;
                lruTouch(key);
                tetrisGrid = *it->second.second.baseGrid; // Deep copy
            }
        }

        if (!hit) {
            // Cache miss, do the rasterization
            Box2f bb;
            vector<Point2f> pointvec = poly.getPoints();
            for(size_t i=0;i<pointvec.size();++i) {
                Point2f pp=pointvec[i];
                pp.Rotate(rotRad);
                bb.Add(pp);
            }

            QVector<QPointF> points;
            points.reserve(pointvec.size());
            for (const auto& p : poly.getPoints()) {
                points.push_back(QPointF(p.X(), p.Y()));
            }

            int safetyBuffer = 2;
            int sizeX = (int)ceil(bb.DimX()*scale) + effectiveGutter + safetyBuffer;
            int sizeY = (int)ceil(bb.DimY()*scale) + effectiveGutter + safetyBuffer;

            QImage img(sizeX, sizeY, QImage::Format_Alpha8);
            img.fill(0);

            QPainter painter;
            painter.begin(&img);
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

            int minX = img.width(), minY = img.height(), maxX = -1, maxY = -1;
            for (int i = 0; i < img.height(); ++i) {
                const uchar *line = img.constScanLine(i);
                bool hasPixel = false;
                for (int j = 0; j < img.width(); ++j) {
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
                img = img.copy(minX, minY, (maxX - minX) + 1, (maxY - minY) + 1);
                tetrisGrid.assign(img.height(), vector<uint8_t>(img.width()));
                for (int y = 0; y < img.height(); y++) {
                    const uchar* line = img.constScanLine(y);
                    for(int x = 0; x < img.width(); ++x) {
                        tetrisGrid[y][x] = (line[x] != 0) ? 1 : 0;
                    }
                }
            } else {
                tetrisGrid.clear();
            }

            // Insert into cache
            {
                std::lock_guard<std::mutex> lk(g_cacheMtx);
                if (g_cache.find(key) == g_cache.end()) {
                    CacheValue val;
                    val.baseGrid = make_shared<vector<vector<uint8_t>>>(tetrisGrid);
                    size_t bytes = 0;
                    if (!tetrisGrid.empty()) {
                        bytes = (size_t)tetrisGrid.size() * (size_t)tetrisGrid[0].size();
                    }
                    val.bytes = bytes;
                    lruInsert(key, std::move(val));
                } else {
                     lruTouch(key);
                }
            }
        }
    }

    if (tetrisGrid.empty()) { // Handle empty rasterization
        int num_rotations_to_generate = (rotationNum >= 4) ? 4 : 1;
        int rotationOffset = (rotationNum >= 4) ? rotationNum / 4 : 0;
        for (int j = 0; j < num_rotations_to_generate; j++) {
            poly.getGrids(rast_i + rotationOffset*j).clear();
            poly.initFromGrid(rast_i + rotationOffset*j);
        }
        return;
    }

    // Process the grid (from cache or new) to create 90 degree rotations
    int num_rotations_to_generate = (rotationNum >= 4) ? 4 : 1;
    int rotationOffset = (rotationNum >= 4) ? rotationNum / 4 : 0;
    for (int j = 0; j < num_rotations_to_generate; j++) {
        if (j != 0)  {
            tetrisGrid = rotateGridCWise(tetrisGrid);
        }
        //add the grid to the poly's vector of grids
        poly.getGrids(rast_i + rotationOffset*j) = tetrisGrid;

        //initializes bottom/left/deltaX/deltaY vectors of the poly, for the current rasterization
        poly.initFromGrid(rast_i + rotationOffset*j);
    }
}

// rotates the grid 90 degree clockwise (by simple swap)
// used to lower the cost of rasterization.
vector<vector<uint8_t> > QtOutline2Rasterizer::rotateGridCWise(vector< vector<uint8_t> >& inGrid) {
    vector<vector<uint8_t> > outGrid(inGrid[0].size());
    for (size_t i = 0; i < inGrid[0].size(); i++) {
        outGrid[i].reserve(inGrid.size());
        for (size_t j = 0; j < inGrid.size(); j++) {
            outGrid[i].push_back(inGrid[inGrid.size() - j - 1][i]);
        }
    }
    return outGrid;
}