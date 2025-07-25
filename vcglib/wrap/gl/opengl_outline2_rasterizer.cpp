#include <wrap/gl/opengl_outline2_rasterizer.h>
#include <gl_utils.h>
#include <logging.h>

#include <vcg/space/box2.h>
#include <vcg/math/matrix44.h>

#include <Eigen/Core>

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

// Triangulates a simple polygon (without holes) using the ear-clipping algorithm.
static void TriangulatePolygon(const std::vector<Point2f>& points, Eigen::MatrixXf& V, Eigen::MatrixXi& F)
{
    size_t num_points = points.size();
    if (num_points < 3) return;

    V.resize(num_points, 2);
    for(size_t i = 0; i < num_points; ++i) {
        V(i, 0) = points[i].X();
        V(i, 1) = points[i].Y();
    }

    std::vector<int> indices;
    indices.reserve(num_points);
    for(size_t i = 0; i < num_points; ++i) {
        indices.push_back(i);
    }

    std::vector<int> result_indices;
    result_indices.reserve((num_points - 2) * 3);

    auto cross_product_z = [](const Point2f& p1, const Point2f& p2, const Point2f& p3) {
        return (p2.X() - p1.X()) * (p3.Y() - p1.Y()) - (p2.Y() - p1.Y()) * (p3.X() - p1.X());
    };

    auto is_inside_triangle = [&](const Point2f& p, const Point2f& a, const Point2f& b, const Point2f& c) {
        return cross_product_z(a, b, p) >= 0 &&
               cross_product_z(b, c, p) >= 0 &&
               cross_product_z(c, a, p) >= 0;
    };

    int n = num_points;
    int current_vertex_idx = 0;
    int pass_counter = 0;
    while (n > 2) {
        if (pass_counter++ > n) {
             LOG_WARN << "Triangulation failed, could not find an ear in a full pass. Polygon may be self-intersecting or degenerate.";
             F.resize(0, 3);
             return;
        }

        int prev_idx_in_list = (current_vertex_idx + n - 1) % n;
        int next_idx_in_list = (current_vertex_idx + 1) % n;

        int p_prev_i = indices[prev_idx_in_list];
        int p_curr_i = indices[current_vertex_idx];
        int p_next_i = indices[next_idx_in_list];

        const Point2f& p_prev = points[p_prev_i];
        const Point2f& p_curr = points[p_curr_i];
        const Point2f& p_next = points[p_next_i];

        bool is_ear = true;
        if (cross_product_z(p_prev, p_curr, p_next) < 1e-9) {
            is_ear = false;
        } else {
            for (int i = 0; i < n; ++i) {
                int vertex_to_check_i = indices[i];
                if (vertex_to_check_i == p_prev_i || vertex_to_check_i == p_curr_i || vertex_to_check_i == p_next_i) continue;
                if (is_inside_triangle(points[vertex_to_check_i], p_prev, p_curr, p_next)) {
                    is_ear = false;
                    break;
                }
            }
        }

        if (is_ear) {
            result_indices.push_back(p_prev_i);
            result_indices.push_back(p_curr_i);
            result_indices.push_back(p_next_i);

            indices.erase(indices.begin() + current_vertex_idx);
            n--;
            pass_counter = 0;
            if (current_vertex_idx >= n && n > 0) current_vertex_idx = 0;
        } else {
            current_vertex_idx = (current_vertex_idx + 1);
            if (current_vertex_idx >= n) current_vertex_idx = 0;
        }
    }

    if (result_indices.empty() && num_points >= 3) {
        F.resize(0,3);
        return;
    }

    F.resize(result_indices.size() / 3, 3);
    for (size_t i = 0; i < result_indices.size() / 3; ++i) {
        F(i, 0) = result_indices[i * 3 + 0];
        F(i, 1) = result_indices[i * 3 + 1];
        F(i, 2) = result_indices[i * 3 + 2];
    }
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

    Eigen::MatrixXf V_tri_uv, V_boundary_uv;
    Eigen::MatrixXi F_tri;
    TriangulatePolygon(original_points, V_tri_uv, F_tri);
    if (F_tri.rows() == 0) return;

    Matrix44f scale = Matrix44f::createScaling(Point3f(scaleFactor, scaleFactor, 1.f));
    Matrix44f rot = Matrix44f::createRotation(rotRad, Point3f(0,0,1));
    Matrix44f trans = Matrix44f::createTranslation(Point3f(-bb.min.X(), -bb.min.Y(), 0));
    Matrix44f offset = Matrix44f::createTranslation(Point3f(gutter_pixels, gutter_pixels, 0));
    Matrix44f proj = Matrix44f::createOrthographic(0, sizeX, 0, sizeY, -1, 1);
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