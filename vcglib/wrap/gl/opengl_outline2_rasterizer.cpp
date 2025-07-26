#include <wrap/gl/opengl_outline2_rasterizer.h>
#include <gl_utils.h>
#include <logging.h>

#include <vcg/space/box2.h>
#include <vcg/math/matrix44.h>

#include <Eigen/Core>
#include <igl/triangle/triangulate.h>

#include <QOpenGLContext>
#include <QSurfaceFormat>
#include <QOffscreenSurface>
#include <QCoreApplication>

#include <vector>
#include <cmath>

using namespace vcg;

// Minimal shaders for rendering a solid color shape
static const char *vs_text[] = {
    "#version 410 core\n"
    "uniform mat4 transform;\n"
    "in vec2 position;\n"
    "void main() {\n"
    "    gl_Position = transform * vec4(position, 0.0, 1.0);\n"
    "}\n"
};

static const char *fs_text[] = {
    "#version 410 core\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
    "}\n"
};

static bool TriangulatePolygon(const std::vector<Point2f>& points, Eigen::MatrixXf& V_out, Eigen::MatrixXi& F_out)
{
    if (points.size() < 3) return false;

    Eigen::MatrixXd V_in(points.size(), 2);
    Eigen::MatrixXi E_in(points.size(), 2);
    for(size_t i = 0; i < points.size(); ++i) {
        V_in(i, 0) = points[i].X();
        V_in(i, 1) = points[i].Y();
        E_in(i, 0) = i;
        E_in(i, 1) = (i + 1) % points.size();
    }

    Eigen::MatrixXd V_igl;
    Eigen::MatrixXi F_igl;
    Eigen::MatrixXd H; // No holes

    // 'p' enforces polygon boundaries. 'Y' prohibits Steiner points on the boundary.
    // Using "-q30" for quality might be too slow, "-Q" for quiet is enough.
    std::string flags = "pYQ";
    igl::triangle::triangulate(V_in, E_in, H, flags, V_igl, F_igl);

    if (F_igl.rows() == 0) {
        LOG_WARN << "Triangulation resulted in 0 faces. The polygon may be degenerate.";
        return false;
    }

    V_out = V_igl.cast<float>();
    F_out = F_igl;

    return true;
}

void OpenGLOutline2Rasterizer::rasterize(RasterizedOutline2 &poly, float scaleFactor, int rast_i, int rotationNum, int gutterWidth)
{
    if (!QCoreApplication::instance()) {
        static int argc = 1;
        static char* argv[] = { (char*)"texture-defrag", nullptr };
        new QCoreApplication(argc, argv);
        LOG_DEBUG << "Created QCoreApplication for OpenGL context.";
    }

    bool contextAvailable = (QOpenGLContext::currentContext() != nullptr);
    QOpenGLContext context;
    QOffscreenSurface surface;
    GLint defaultFBO = 0;

    if (!contextAvailable) {
        QSurfaceFormat format;
        format.setVersion(4, 1);
        format.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
        context.setFormat(format);
        if (!context.create()) { LOG_ERR << "Failed to create opengl context for rasterizer"; return; }
        surface.setFormat(context.format());
        surface.create();
        if (!context.makeCurrent(&surface)) { LOG_ERR << "Failed to make OpenGL context current for rasterizer"; return; }
    } else {
        GetOpenGLFunctionsHandle()->glGetIntegerv(GL_FRAMEBUFFER_BINDING, &defaultFBO);
    }
    OpenGLFunctionsHandle glFuncs = GetOpenGLFunctionsHandle();

    float rotRad = M_PI * 2.0f * float(rast_i) / float(rotationNum);
    std::vector<Point2f> original_points = poly.getPoints();
    if (original_points.size() < 3) return;

    Box2f bb;
    for(size_t i = 0; i < original_points.size(); ++i) {
        Point2f p = original_points[i];
        p.Rotate(rotRad);
        bb.Add(p);
    }

    // The old rasterizer used a pen width of `gutterWidth * 2`. A centered pen of this width
    // dilates the shape by `gutterWidth` pixels on each side. We replicate this.
    int gutter_pixels = gutterWidth;
    int sizeX = (int)ceil(bb.DimX() * scaleFactor) + (gutter_pixels * 2);
    int sizeY = (int)ceil(bb.DimY() * scaleFactor) + (gutter_pixels * 2);

    if (sizeX <= 0 || sizeY <= 0) { LOG_WARN << "Skipping rasterization of degenerate chart."; return; }

    Eigen::MatrixXf V_tri_uv;
    Eigen::MatrixXi F_tri;
    if (!TriangulatePolygon(original_points, V_tri_uv, F_tri)) {
        return; // Triangulation failed, skip this chart.
    }

    Matrix44f scale, rot, trans, offset, proj;
    scale.SetScale(Point3f(scaleFactor, scaleFactor, 1.f));
    rot.SetRotateDeg(rotRad * 180.0f / M_PI, Point3f(0,0,1));
    trans.SetTranslate(Point3f(-bb.min.X(), -bb.min.Y(), 0));
    offset.SetTranslate(Point3f(gutter_pixels, gutter_pixels, 0));
    proj.Orthographic(0, sizeX, 0, sizeY, -1, 1);
    Matrix44f mvp = proj * offset * scale * trans * rot;
    
    // Setup OpenGL objects
    GLuint vao, vbo_tri, ebo_tri, vbo_boundary;
    glFuncs->glGenVertexArrays(1, &vao);
    glFuncs->glBindVertexArray(vao);

    glFuncs->glGenBuffers(1, &vbo_tri);
    glFuncs->glBindBuffer(GL_ARRAY_BUFFER, vbo_tri);
    glFuncs->glBufferData(GL_ARRAY_BUFFER, V_tri_uv.size() * sizeof(float), V_tri_uv.data(), GL_STATIC_DRAW);

    GLint program = CompileShaders(vs_text, fs_text);
    glFuncs->glUseProgram(program);
    GLint pos_location = glFuncs->glGetAttribLocation(program, "position");
    glFuncs->glVertexAttribPointer(pos_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glFuncs->glEnableVertexAttribArray(pos_location);
    glFuncs->glUniformMatrix4fv(glFuncs->glGetUniformLocation(program, "transform"), 1, GL_FALSE, mvp.V());

    glFuncs->glGenBuffers(1, &ebo_tri);
    glFuncs->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_tri);
    glFuncs->glBufferData(GL_ELEMENT_ARRAY_BUFFER, F_tri.size() * sizeof(int), F_tri.data(), GL_STATIC_DRAW);

    glFuncs->glGenBuffers(1, &vbo_boundary);
    glFuncs->glBindBuffer(GL_ARRAY_BUFFER, vbo_boundary);
    glFuncs->glBufferData(GL_ARRAY_BUFFER, original_points.size() * sizeof(Point2f), original_points.data(), GL_STATIC_DRAW);
    
    GLuint fbo, renderTarget;
    glFuncs->glGenFramebuffers(1, &fbo);
    glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFuncs->glGenTextures(1, &renderTarget);
    glFuncs->glBindTexture(GL_TEXTURE_2D, renderTarget);
    glFuncs->glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, sizeX, sizeY, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glFuncs->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTarget, 0);

    // Render
    glFuncs->glViewport(0, 0, sizeX, sizeY);
    glFuncs->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glFuncs->glClear(GL_COLOR_BUFFER_BIT);
    glFuncs->glDisable(GL_DEPTH_TEST);
    glFuncs->glDisable(GL_CULL_FACE);

    // Pass 1: Render filled triangles
    glFuncs->glBindBuffer(GL_ARRAY_BUFFER, vbo_tri);
    glFuncs->glVertexAttribPointer(pos_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glFuncs->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_tri);
    glFuncs->glDrawElements(GL_TRIANGLES, F_tri.size(), GL_UNSIGNED_INT, 0);

    // Pass 2: Render dilated boundary
    glFuncs->glLineWidth(gutter_pixels * 2.0f);
    glFuncs->glBindBuffer(GL_ARRAY_BUFFER, vbo_boundary);
    glFuncs->glVertexAttribPointer(pos_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glFuncs->glDrawArrays(GL_LINE_LOOP, 0, original_points.size());

    // Read back and build grid
    std::vector<unsigned char> pixels(sizeX * sizeY);
    glFuncs->glReadPixels(0, 0, sizeX, sizeY, GL_RED, GL_UNSIGNED_BYTE, pixels.data());
    
    std::vector<std::vector<int>> tetrisGrid(sizeY, std::vector<int>(sizeX));
    for (int y = 0; y < sizeY; ++y) {
        for (int x = 0; x < sizeX; ++x) {
            // OpenGL renders with (0,0) at bottom-left, but we want top-left for the grid
            if (pixels[(y * sizeX) + x] > 0) {
                tetrisGrid[sizeY - 1 - y][x] = 1;
            }
        }
    }
    
    // Cleanup
    glFuncs->glUseProgram(0);
    glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, defaultFBO);
    glFuncs->glDeleteProgram(program);
    glFuncs->glDeleteBuffers(1, &vbo_tri);
    glFuncs->glDeleteBuffers(1, &ebo_tri);
    glFuncs->glDeleteBuffers(1, &vbo_boundary);
    glFuncs->glDeleteVertexArrays(1, &vao);
    glFuncs->glDeleteTextures(1, &renderTarget);
    glFuncs->glDeleteFramebuffers(1, &fbo);
    if (!contextAvailable) context.doneCurrent();
    
    // Generate other 3 rotations by rotating the grid on CPU
    int rotationOffset = rotationNum / 4;
    for (int j = 0; j < 4; j++) {
        if (j != 0) {
            tetrisGrid = rotateGridCWise(tetrisGrid);
        }
        if (!poly.hasGrid(rast_i + rotationOffset * j)) {
            poly.getGrids(rast_i + rotationOffset * j) = tetrisGrid;
            poly.initFromGrid(rast_i + rotationOffset * j);
        }
    }
}

std::vector<std::vector<int>> OpenGLOutline2Rasterizer::rotateGridCWise(std::vector<std::vector<int>>& inGrid) {
    if (inGrid.empty()) return {};
    size_t rows = inGrid.size();
    size_t cols = inGrid[0].size();
    std::vector<std::vector<int>> outGrid(cols, std::vector<int>(rows));
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            outGrid[c][rows - 1 - r] = inGrid[r][c];
        }
    }
    return outGrid;
}