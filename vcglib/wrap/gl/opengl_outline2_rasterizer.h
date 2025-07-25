#ifndef OPENGL_OUTLINE2_RASTERIZER_H
#define OPENGL_OUTLINE2_RASTERIZER_H

#include <vcg/space/rasterized_outline2_packer.h>
#include <vector>

class OpenGLOutline2Rasterizer
{
public:
    static void rasterize(vcg::RasterizedOutline2 &poly,
                          float scaleFactor,
                          int rast_i, int rotationNum, int gutterWidth);

private:
    static std::vector<std::vector<int>> rotateGridCWise(std::vector<std::vector<int>>& inGrid);
};

#endif // OPENGL_OUTLINE2_RASTERIZER_H