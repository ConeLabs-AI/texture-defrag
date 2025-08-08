#include <wrap/qt/outline2_rasterizer.h>
#include <wrap/qt/col_qt_convert.h>
#include <vcg/space/color4.h>
#include <wrap/qt/col_qt_convert.h>

#include <fstream>

using namespace vcg;
using namespace std;

void QtOutline2Rasterizer::rasterize(RasterizedOutline2 &poly,
                                 float scale,
                                 int rast_i,
                                 int rotationNum,
                                 int gutterWidth)
{
    // since the brush is centered on the outline, a gutter of N pixels requires a pen of 2*N width
    gutterWidth *= 2;

    float rotRad = M_PI*2.0f*float(rast_i) / float(rotationNum);

    //get polygon's BB, rotated according to the input parameter
    Box2f bb;
    vector<Point2f> pointvec = poly.getPoints();
    for(size_t i=0;i<pointvec.size();++i) {
        Point2f pp=pointvec[i];
        pp.Rotate(rotRad);
        bb.Add(pp);
    }

    //create the polygon to print it
    QVector<QPointF> points;
    points.reserve(pointvec.size());
    for (const auto& p : poly.getPoints()) {
        points.push_back(QPointF(p.X(), p.Y()));
    }

    // Compute the raster space size by rounding up the scaled bounding box size
    // and adding the gutter width. A small safety buffer prevents clipping.
    int safetyBuffer = 2;
    int sizeX = (int)ceil(bb.DimX()*scale) + gutterWidth + safetyBuffer;
    int sizeY = (int)ceil(bb.DimY()*scale) + gutterWidth + safetyBuffer;

    // Use a 1-byte-per-pixel format, which is 4x smaller and faster to process.
    QImage img(sizeX, sizeY, QImage::Format_Alpha8);
    img.fill(0); // Transparent background

    QPainter painter;
    painter.begin(&img);
    painter.setRenderHint(QPainter::Antialiasing, false);
 
    // Fill the interior of the polygon
    painter.setBrush(QBrush(Qt::white, Qt::SolidPattern));
 
    // Draw the boundary with a thick pen to create the gutter
    QPen qp(Qt::white);
    qp.setWidth(gutterWidth);
    qp.setCosmetic(false); // Use physical width, not a 1px cosmetic pen
    qp.setJoinStyle(Qt::MiterJoin);
    painter.setPen(qp);
 
    // Setup transformation
    painter.resetTransform();
    painter.translate(QPointF(-(bb.min.X()*scale) + (gutterWidth + safetyBuffer)/2.0f, -(bb.min.Y()*scale) + (gutterWidth + safetyBuffer)/2.0f));
    painter.rotate(math::ToDeg(rotRad));
    painter.scale(scale,scale);
 
    // A single draw call for efficiency
    painter.drawPolygon(QPolygonF(points));
 
    painter.end();
 
    // --- Cropping ---
    // Optimized to find bounds and perform a single copy
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
 
    if (maxX < minX) { // Empty rasterization
        int num_rotations_to_generate = (rotationNum >= 4) ? 4 : 1;
        int rotationOffset = (rotationNum >= 4) ? rotationNum / 4 : 0;
        for (int j = 0; j < num_rotations_to_generate; j++) {
            poly.getGrids(rast_i + rotationOffset*j).clear();
            poly.initFromGrid(rast_i + rotationOffset*j);
        }
        return;
    }
 
    img = img.copy(minX, minY, (maxX - minX) + 1, (maxY - minY) + 1);

    // --- Grid Creation ---
    vector<vector<uint8_t>> tetrisGrid(img.height(), vector<uint8_t>(img.width()));
    for (int y = 0; y < img.height(); y++) {
        const uchar* line = img.constScanLine(y);
        for(int x = 0; x < img.width(); ++x) {
            tetrisGrid[y][x] = (line[x] != 0) ? 1 : 0;
        }
    }

    //create the 4 rasterizations (one every 90°) using the discrete representation grid we've just created
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