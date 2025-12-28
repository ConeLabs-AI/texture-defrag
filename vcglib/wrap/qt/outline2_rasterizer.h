#ifndef QTPOLYRASTERIZER_H
#define QTPOLYRASTERIZER_H

#include <cstdint>
#include <vector>

#include <QImage>
//#include <QSvgGenerator>
#include <QPainter>
#include <vcg/space/point2.h>
#include <vcg/space/color4.h>
#include <vcg/space/box2.h>
#include <vcg/math/similarity2.h>
#include <vcg/space/rasterized_outline2_packer.h>

///this class is used to draw polygons on an image could be vectorial or not
class QtOutline2Rasterizer
{
public:
    static void rasterize(vcg::RasterizedOutline2 &poly,
                          float scaleFactor,
                          int rast_i, int rotationNum, int gutterWidth,
                          bool bypassCache = false);

    static vcg::BitGrid rotateGridCWise(const vcg::BitGrid& inGrid);

    // Cache control (thread-safe)
    static void setCacheMaxBytes(std::size_t bytes);
    static void clearCache();

    // Cache/rasterizer statistics (thread-safe)
    struct CacheStats {
        uint64_t calls = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t inserts = 0;
        uint64_t evictions = 0;
        std::size_t bytesCurrent = 0;
        std::size_t bytesMax = 0;
        double t_lookup_s = 0.0;       // time spent in cache lookups
        double t_miss_raster_s = 0.0;  // time spent doing actual QImage/QPainter rasterization + crop + grid build + insert
        double t_hit_copy_s = 0.0;     // time spent copying cached grid into working buffer
        double t_rotate_s = 0.0;       // time spent generating 90-degree rotations + initFromGrid
        double t_total_s = 0.0;        // total time spent inside rasterize()
    };

    // Take a snapshot of statistics. If resetCounters==true, counters and timers are reset after the snapshot.
    static CacheStats statsSnapshot(bool resetCounters);
};
#endif // QTPOLYRASTERIZER_H