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

/*
 * References for the bicubic interpolated texture lookup:
 *  - GPU gems 2 ch 20 (Sigg and Hadwiger, 2005)
 *  - Efficient GPU-Based Texture Interpolation using Uniform B-Splines  (Ruijters et al., 2009)
 * */

#include "mesh.h"
#include "texture_rendering.h"
#include "gl_utils.h"
#include "pushpull.h"
#include "mesh_attribute.h"
#include "logging.h"

#include <iostream>
#include <algorithm>
#include <memory>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <QImage>
#include <QFileInfo>
#include <QDir>
#include <QString>

#include <QOpenGLContext>
#include <QSurfaceFormat>
#include <QOffscreenSurface>
#include <QFile>

#include <chrono>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "texture_conversion.h"


static const char *vs_text[] = {
    "#version 410 core                                           \n"
    "                                                            \n"
    "in vec2 position;                                           \n"
    "in vec2 texcoord;                                           \n"
    "in vec4 color;                                              \n"
    "out vec2 uv;                                                \n"
    "out vec4 fcolor;                                            \n"
    "uniform vec2 tile_min;                                      \n"
    "uniform vec2 tile_scale;                                    \n"
    "                                                            \n"
    "void main(void)                                             \n"
    "{                                                           \n"
    "    uv = texcoord;                                         \n"
    "    fcolor = color;                                         \n"
    "    vec2 local = (position - tile_min) / tile_scale;        \n"
    "    vec2 p = vec2(2.0 * local.x - 1.0, 1.0 - 2.0 * local.y);\n"
    "    gl_Position = vec4(p, 0.5, 1.0);                        \n"
    "}                                                           \n"
};

static const char *fs_text[] = {
    "#version 410 core                                                     \n"
    "                                                                      \n"
    "uniform sampler2DArray tile_cache;                                    \n"
    "uniform usampler2D page_table;                                        \n"
    "                                                                      \n"
    "uniform vec2 src_texture_size;                                        \n"
    "uniform vec2 tile_tex_size;                                           \n"
    "uniform float tile_size;                                              \n"
    "uniform float border;                                                 \n"
    "uniform int render_mode;                                              \n"
    "                                                                      \n"
    "in vec2 uv;                                                           \n"
    "in vec4 fcolor;                                                       \n"
    "                                                                      \n"
    "out vec4 texelColor;                                                  \n"
    "                                                                      \n"
    "void main(void)                                                       \n"
    "{                                                                     \n"
    "    // Sentinel check: handle invalid faces (uv.s == -1.0)            \n"
    "    if (uv.s < -0.5) {                                                \n"
    "        texelColor = vec4(0, 0, 0, 1);                                \n"
    "        return;                                                       \n"
    "    }                                                                 \n"
    "                                                                      \n"
    "    // Clamp UVs to handle floating point noise on chart edges        \n"
    "    vec2 src_uv = clamp(uv, 0.0, 1.0);                                \n"
    "                                                                      \n"
    "    ivec2 ptSize = textureSize(page_table, 0);                        \n"
    "    vec2 fileCoord = vec2(src_uv.x * src_texture_size.x,              \n"
    "                         (1.0 - src_uv.y) * src_texture_size.y);      \n"
    "    ivec2 vtile = ivec2(floor(fileCoord / tile_size));                \n"
    "    vtile = clamp(vtile, ivec2(0,0), ptSize - ivec2(1,1));            \n"
    "                                                                      \n"
    "    uint layer = texelFetch(page_table, vtile, 0).r;                  \n"
    "    if (layer == 0xFFFFFFFFu) {                                       \n"
    "        texelColor = vec4(0, 0, 0, 1);                                \n"
    "        return;                                                       \n"
    "    }                                                                 \n"
    "                                                                      \n"
    "    // Compute tile-local UV (bottom-up) using the same ROI remap idea\n"
    "    vec2 roi_min = vec2(float(vtile.x) * tile_size / src_texture_size.x,\n"
    "                       1.0 - float(vtile.y + 1) * tile_size / src_texture_size.y);\n"
    "    vec2 roi_scale = vec2(tile_size / src_texture_size.x,             \n"
    "                         tile_size / src_texture_size.y);             \n"
    "    vec2 local_uv = clamp((src_uv - roi_min) / roi_scale, 0.0, 1.0);  \n"
    "    vec2 phys_uv = (local_uv * tile_size + border) / tile_tex_size;   \n"
    "                                                                      \n"
    "    if (render_mode == 0) {                                           \n"
    "        texelColor = vec4(texture(tile_cache, vec3(phys_uv, float(layer))).rgb, 1);\n"
    "    } else if (render_mode == 1) {                                    \n"
    "        // Bicubic interpolation logic using phys_uv on the bordered tile\n"
    "        vec2 coord = phys_uv * tile_tex_size - vec2(0.5, 0.5);        \n"
    "        vec2 idx = floor(coord);                                      \n"
    "        vec2 fraction = coord - idx;                                  \n"
    "        vec2 one_frac = vec2(1.0, 1.0) - fraction;                    \n"
    "        vec2 one_frac2 = one_frac * one_frac;                         \n"
    "        vec2 fraction2 = fraction * fraction;                         \n"
    "        vec2 w0 = (1.0/6.0) * one_frac2 * one_frac;                   \n"
    "        vec2 w1 = (2.0/3.0) - 0.5 * fraction2 * (2.0 - fraction);     \n"
    "        vec2 w2 = (2.0/3.0) - 0.5 * one_frac2 * (2.0 - one_frac);     \n"
    "        vec2 w3 = (1.0/6.0) * fraction2 * fraction;                   \n"
    "        vec2 g0 = w0 + w1;                                            \n"
    "        vec2 g1 = w2 + w3;                                            \n"
    "        vec2 h0 = (w1 / g0) - 0.5 + idx;                              \n"
    "        vec2 h1 = (w3 / g1) + 1.5 + idx;                              \n"
    "        vec4 tex00 = texture(tile_cache, vec3(vec2(h0.x, h0.y) / tile_tex_size, float(layer)));\n"
    "        vec4 tex10 = texture(tile_cache, vec3(vec2(h1.x, h0.y) / tile_tex_size, float(layer)));\n"
    "        vec4 tex01 = texture(tile_cache, vec3(vec2(h0.x, h1.y) / tile_tex_size, float(layer)));\n"
    "        vec4 tex11 = texture(tile_cache, vec3(vec2(h1.x, h1.y) / tile_tex_size, float(layer)));\n"
    "        tex00 = mix(tex00, tex01, g1.y);                              \n"
    "        tex10 = mix(tex10, tex11, g1.y);                              \n"
    "        texelColor = mix(tex00, tex10, g1.x);                         \n"
    "    } else {                                                          \n"
    "        texelColor = fcolor;                                          \n"
    "    }                                                                 \n"
    "}                                                                     \n"
};

// A struct to manage persistent OpenGL state throughout the rendering of all texture sheets.
// This avoids costly creation/destruction of contexts, programs, and buffers for each sheet.
struct RenderingContext {
    OpenGLFunctionsHandle glFuncs;
    std::unique_ptr<QOpenGLContext> context;
    std::unique_ptr<QOffscreenSurface> surface;
    bool ownContext = false;

    GLuint program = 0;
    GLuint vao = 0;
    GLuint vertexbuf = 0;
    GLuint fbo = 0;
    GLuint renderTarget = 0;
    int renderedTexWidth = -1;
    int renderedTexHeight = -1;
    GLint initialDrawBuffer = 0;

    // Aggregated profile stats
    double total_vbo_s = 0.0;
    double total_draw_s = 0.0;
    double total_read_s = 0.0;

    // Cached uniform locations
    GLint loc_tile_cache = -1;
    GLint loc_page_table = -1;
    GLint loc_src_texture_size = -1;
    GLint loc_tile_tex_size = -1;
    GLint loc_tile_size = -1;
    GLint loc_border = -1;
    GLint loc_render_mode = -1;
    GLint loc_tile_min = -1;
    GLint loc_tile_scale = -1;

    RenderingContext() {
        if (QOpenGLContext::currentContext() == nullptr) {
            LOG_ERR << "No current OpenGL context. Ensure a persistent context is created before rendering.";
            std::exit(-1);
        }

        glFuncs = GetOpenGLFunctionsHandle();
        if (ownContext) {
            glFuncs->glGetIntegerv(GL_DRAW_BUFFER, &initialDrawBuffer);
        }

        CHECK_GL_ERROR();

        program = CompileShaders(vs_text, fs_text);
        glFuncs->glUseProgram(program);

        glFuncs->glGenVertexArrays(1, &vao);
        glFuncs->glBindVertexArray(vao);

        glFuncs->glGenBuffers(1, &vertexbuf);
        glFuncs->glBindBuffer(GL_ARRAY_BUFFER, vertexbuf);

        GLint pos_location = glFuncs->glGetAttribLocation(program, "position");
        glFuncs->glVertexAttribPointer(pos_location, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
        glFuncs->glEnableVertexAttribArray(pos_location);

        GLint tc_location = glFuncs->glGetAttribLocation(program, "texcoord");
        glFuncs->glVertexAttribPointer(tc_location, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(2 * sizeof(float)));
        glFuncs->glEnableVertexAttribArray(tc_location);

        GLint color_location = glFuncs->glGetAttribLocation(program, "color");
        glFuncs->glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 5 * sizeof(float), (void *)(4 * sizeof(float)));
        glFuncs->glEnableVertexAttribArray(color_location);

        glFuncs->glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Cache uniform locations
        loc_tile_cache = glFuncs->glGetUniformLocation(program, "tile_cache");
        loc_page_table = glFuncs->glGetUniformLocation(program, "page_table");
        loc_src_texture_size = glFuncs->glGetUniformLocation(program, "src_texture_size");
        loc_tile_tex_size = glFuncs->glGetUniformLocation(program, "tile_tex_size");
        loc_tile_size = glFuncs->glGetUniformLocation(program, "tile_size");
        loc_border = glFuncs->glGetUniformLocation(program, "border");
        loc_render_mode = glFuncs->glGetUniformLocation(program, "render_mode");
        loc_tile_min = glFuncs->glGetUniformLocation(program, "tile_min");
        loc_tile_scale = glFuncs->glGetUniformLocation(program, "tile_scale");

        glFuncs->glGenFramebuffers(1, &fbo);
        glFuncs->glGenTextures(1, &renderTarget);

        glFuncs->glDisable(GL_DEPTH_TEST);
        glFuncs->glDisable(GL_STENCIL_TEST);
    }

    ~RenderingContext() {
        glFuncs->glUseProgram(0);
        glFuncs->glBindVertexArray(0);
        glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glFuncs->glDeleteProgram(program);
        glFuncs->glDeleteVertexArrays(1, &vao);
        glFuncs->glDeleteBuffers(1, &vertexbuf);
        glFuncs->glDeleteFramebuffers(1, &fbo);
        glFuncs->glDeleteTextures(1, &renderTarget);

        if (ownContext) {
            glFuncs->glDrawBuffer(initialDrawBuffer);
            context->doneCurrent();
        }
    }

    void prepareRenderTarget(int width, int height) {
        if (width != renderedTexWidth || height != renderedTexHeight) {
            LOG_DEBUG << "Configuring render target for size " << width << "x" << height;

            glFuncs->glBindTexture(GL_TEXTURE_2D, renderTarget);
            glFuncs->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
            glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glFuncs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFuncs->glBindTexture(GL_TEXTURE_2D, 0);

            glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFuncs->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTarget, 0);

            if (glFuncs->glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                LOG_ERR << "[OPENGL] FATAL: Framebuffer is not complete.";
                CHECK_GL_ERROR();
                std::exit(-1);
            }

            renderedTexWidth = width;
            renderedTexHeight = height;
        }
        // Always set viewport for robustness
        glFuncs->glViewport(0, 0, width, height);
    }
};

// Simple background saving queue to overlap PNG compression with rendering
class ImageSaveQueue {
public:
    explicit ImageSaveQueue(int maxInFlight = 2) : maxInFlight(maxInFlight) {
        worker = std::thread([this]() { this->run(); });
    }
    ~ImageSaveQueue() {
        finish();
    }
    void enqueue(QImage image, const QString& absolutePath, int quality) {
        std::unique_lock<std::mutex> lock(mutex);
        auto t_wait_start = std::chrono::high_resolution_clock::now();
        notFull.wait(lock, [this]() { return stop || queue.size() < static_cast<size_t>(maxInFlight); });
        auto t_wait_end = std::chrono::high_resolution_clock::now();
        totalEnqueueWaitS += std::chrono::duration<double>(t_wait_end - t_wait_start).count();
        if (stop) return;
        queue.push(Task{std::move(image), absolutePath, quality});
        tasksEnqueued++;
        notEmpty.notify_one();
    }
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        notEmpty.notify_all();
        notFull.notify_all();
        if (worker.joinable()) worker.join();
    }
    struct SaveStats {
        int enqueued = 0;
        int saved = 0;
        double totalSaveS = 0.0;
        double minSaveS = std::numeric_limits<double>::infinity();
        double maxSaveS = 0.0;
        double totalEnqueueWaitS = 0.0;
    };
    SaveStats statsSnapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        SaveStats s;
        s.enqueued = tasksEnqueued;
        s.saved = tasksSaved;
        s.totalSaveS = totalSaveS;
        s.minSaveS = (tasksSaved > 0) ? minSaveS : 0.0;
        s.maxSaveS = maxSaveS;
        s.totalEnqueueWaitS = totalEnqueueWaitS;
        return s;
    }
    void resetStats() {
        std::lock_guard<std::mutex> lock(mutex);
        tasksEnqueued = 0;
        tasksSaved = 0;
        totalSaveS = 0.0;
        minSaveS = std::numeric_limits<double>::infinity();
        maxSaveS = 0.0;
        totalEnqueueWaitS = 0.0;
    }
private:
    struct Task {
        QImage image;
        QString path;
        int quality;
    };
    void run() {
        for (;;) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                notEmpty.wait(lock, [this]() { return stop || !queue.empty(); });
                if (stop && queue.empty()) break;
                task = std::move(queue.front());
                queue.pop();
                notFull.notify_one();
            }
            auto t_save_start = std::chrono::high_resolution_clock::now();
            bool ok = task.image.save(task.path, "png", task.quality);
            auto t_save_end = std::chrono::high_resolution_clock::now();
            double t_save_s = std::chrono::duration<double>(t_save_end - t_save_start).count();
            {
                std::lock_guard<std::mutex> lock(mutex);
                tasksSaved++;
                totalSaveS += t_save_s;
                if (t_save_s > maxSaveS) maxSaveS = t_save_s;
                if (t_save_s < minSaveS) minSaveS = t_save_s;
            }
            if (!ok) {
                LOG_ERR << "Error saving texture file " << task.path.toStdString();
                QFileInfo fi(task.path);
                QDir dir = fi.absoluteDir();
                if (!dir.exists()) {
                    LOG_ERR << "  - Reason: Directory does not exist.";
                } else {
                    LOG_ERR << "  - Reason: Path exists but save failed. Check permissions.";
                }
            }
        }
    }
    std::thread worker;
    std::mutex mutex;
    std::condition_variable notEmpty;
    std::condition_variable notFull;
    std::queue<Task> queue;
    int maxInFlight = 2;
    bool stop = false;
    // Stats
    int tasksEnqueued = 0;
    int tasksSaved = 0;
    double totalSaveS = 0.0;
    double minSaveS = std::numeric_limits<double>::infinity();
    double maxSaveS = 0.0;
    double totalEnqueueWaitS = 0.0;
};

static std::shared_ptr<QImage> RenderTexture(RenderingContext& ctx,
                                             std::vector<Mesh::FacePointer>& fvec,
                                             Mesh &m, TextureObjectHandle textureObject,
                                             bool filter, RenderMode imode,
                                             int textureWidth, int textureHeight);


int FacesByTextureIndex(Mesh& m, std::vector<std::vector<Mesh::FacePointer>>& fv)
{
    fv.clear();

    // Detect the number of required textures
    int nTex = 1;
    for (auto&f : m.face) {
        nTex = std::max(nTex, f.cWT(0).N() + 1);
    }

    fv.resize(nTex);

    for (auto& f : m.face) {
        int ti = f.cWT(0).N();
        ensure(ti < nTex);
        fv[ti].push_back(&f);
    }

    return fv.size();
}

void RenderTextureAndSave(const std::string& outFileName, Mesh& m, TextureObjectHandle textureObject, const std::vector<TextureSize> &texSizes,
                                                   bool filter, RenderMode imode)
{
    // Fail fast if textures are unavailable
    if (!textureObject || textureObject->ArraySize() == 0) {
        LOG_ERR << "No textures available for rendering. Ensure input OBJ references an MTL with map_Kd textures.";
        std::exit(-1);
    }

    // Automatic On-Demand Conversion:
    // Generate .rawtile files only if they don't exist, right before rendering begins.
    LOG_INFO << "Ensuring optimized .rawtile caches exist...";
    for (size_t i = 0; i < textureObject->ArraySize(); ++i) {
        std::string inputPath = textureObject->texInfoVec[i].path;
        std::string rawPath = TextureConversion::GetRawTilePath(inputPath);
        if (!QFile::exists(QString::fromStdString(rawPath))) {
            LOG_INFO << "  [Preparing Cache] " << inputPath;
            if (!TextureConversion::ConvertToRawTile(inputPath, rawPath)) {
                LOG_ERR << "  - Conversion failed. Tiled I/O disabled for this file.";
            }
        }
    }

    // Reset GPU texture cache stats for this rendering pass
    if (textureObject) textureObject->ResetCacheStats();

    std::vector<std::vector<Mesh::FacePointer>> facesByTexture;
    int nTex = FacesByTextureIndex(m, facesByTexture);

    ensure(nTex <= (int) texSizes.size());

    // Validate output texture sizes
    for (int i = 0; i < nTex; ++i) {
        if (texSizes[i].w <= 0 || texSizes[i].h <= 0) {
            LOG_ERR << "Invalid output texture size for sheet " << i << ": " << texSizes[i].w << "x" << texSizes[i].h;
            std::exit(-1);
        }
    }

    m.textures.clear();

    QFileInfo fi(outFileName.c_str());
    QString wd = QDir::currentPath();
    QDir outDir = fi.absoluteDir();
    if (!outDir.exists()) {
        if (!outDir.mkpath(".")) {
            LOG_ERR << "Failed to create output directory: " << outDir.absolutePath().toStdString();
            std::exit(-1);
        }
    }
    QDir::setCurrent(outDir.absolutePath());

    auto t_total_start = std::chrono::high_resolution_clock::now();
    double t_total_render_s = 0.0;
    double t_total_savequeue_enqueue_s = 0.0;
    double t_total_png_save_s = 0.0; // captured by queue
    double t_save_wait_s = 0.0;      // time waiting in finish()
    int64_t total_pixels_rendered = 0;

    RenderingContext renderingContext;
    // Initialize virtual texturing (tile cache + page tables) once per render pass
    if (textureObject) textureObject->InitVirtualTexturing();
    ImageSaveQueue saveQueue(2);
    saveQueue.resetStats();

    int lastProgressLog = -1;
    for (int i = 0; i < nTex; ++i) {
        int progressPercent = (i * 100) / nTex;
        if (progressPercent / 10 != lastProgressLog) {
            LOG_INFO << "Rendering Progress: " << progressPercent << "% (" << (i + 1) << " / " << nTex << " sheets)";
            lastProgressLog = progressPercent / 10;
        }

        auto t_render_start = std::chrono::high_resolution_clock::now();
        std::shared_ptr<QImage> teximg = RenderTexture(renderingContext, facesByTexture[i], m, textureObject, filter, imode, texSizes[i].w, texSizes[i].h);
        auto t_render_end = std::chrono::high_resolution_clock::now();
        double t_render_s = std::chrono::duration<double>(t_render_end - t_render_start).count();
        t_total_render_s += t_render_s;
        total_pixels_rendered += int64_t(texSizes[i].w) * int64_t(texSizes[i].h);

        std::stringstream suffix;
        suffix << "_texture_" << i << ".png";
        std::string s(outFileName);
        std::string texturePath = s.substr(0, s.find_last_of('.')).append(suffix.str());

        QFileInfo texFI(texturePath.c_str());
        m.textures.push_back(texFI.fileName().toStdString());
        const QString absPath = texFI.absoluteFilePath();
        // Enqueue save to overlap compression with next sheet rendering
        auto t_enqueue_start = std::chrono::high_resolution_clock::now();
        saveQueue.enqueue(*teximg, absPath, 75);
        auto t_enqueue_end = std::chrono::high_resolution_clock::now();
        t_total_savequeue_enqueue_s += std::chrono::duration<double>(t_enqueue_end - t_enqueue_start).count();
    }

    // Ensure all pending saves are complete before restoring working directory
    auto t_save_finish_start = std::chrono::high_resolution_clock::now();
    saveQueue.finish();
    auto t_save_finish_end = std::chrono::high_resolution_clock::now();
    t_save_wait_s += std::chrono::duration<double>(t_save_finish_end - t_save_finish_start).count();
    QDir::setCurrent(wd);

    // Snapshot queue save stats
    auto saveStats = saveQueue.statsSnapshot();
    t_total_png_save_s = saveStats.totalSaveS;

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double t_total_s = std::chrono::duration<double>(t_total_end - t_total_start).count();

    // Log performance summary
    LOG_INFO << "[RENDER-STATS] total_s=" << t_total_s
             << " sheets=" << nTex
             << " pixels=" << total_pixels_rendered
             << " render_s=" << t_total_render_s
             << " png_save_s=" << t_total_png_save_s;

    LOG_VERBOSE << "[RENDER-STATS-VERBOSE] enqueue_s=" << t_total_savequeue_enqueue_s
             << " save_wait_s=" << t_save_wait_s
             << " png_min_s=" << saveStats.minSaveS
             << " png_max_s=" << saveStats.maxSaveS
             << " png_saved=" << saveStats.saved;

    LOG_INFO << "[RENDER-PROFILE-SUMMARY] total_vbo_s=" << renderingContext.total_vbo_s
             << " total_draw_s=" << renderingContext.total_draw_s
             << " total_read_s=" << renderingContext.total_read_s;

    // Log GPU texture cache and Tiled I/O stats
    if (textureObject) {
        auto cs = textureObject->GetCacheStats();
        uint64_t lookups = cs.hits + cs.misses;
        double hitRate = lookups ? double(cs.hits) / double(lookups) : 0.0;
        double mbRead = double(cs.bytesRead) / (1024.0 * 1024.0);
        double mbEvicted = double(cs.bytesEvicted) / (1024.0 * 1024.0);
        double diskSpeed = (cs.diskReadTimeS > 0) ? (mbRead / cs.diskReadTimeS) : 0.0;

        LOG_INFO << "--- Texture Rendering Observability Summary ---";
        LOG_INFO << "  [I/O Efficiency] Tiles Loaded: " << cs.tilesLoaded << " (" << mbRead << " MB total)";
        LOG_INFO << "  [Disk Performance] Raw I/O Time: " << cs.diskReadTimeS << "s (" << diskSpeed << " MB/s)";
        LOG_INFO << "  [GPU Cache] Hit Rate: " << (hitRate * 100.0) << "% (" << lookups << " lookups)";
        LOG_INFO << "  [GPU Pressure] Evictions: " << cs.evictions << " (" << mbEvicted << " MB evicted)";
        if (cs.fallbacks > 0) {
            LOG_WARN << "  [Optimization] Tiled I/O Fallbacks: " << cs.fallbacks << " (missing .rawtile files)";
        }
        LOG_INFO << "  [Final State] Bytes in Use: " << (double(textureObject->GetCurrentCacheBytes()) / (1024.0 * 1024.0)) << " MB";
        LOG_INFO << "-----------------------------------------------";

        LOG_VERBOSE << "[TEX-CACHE] lookups=" << lookups
                 << " hits=" << cs.hits
                 << " misses=" << cs.misses
                 << " hitRate=" << hitRate
                 << " evictions=" << cs.evictions
                 << " bytesEvicted=" << cs.bytesEvicted
                 << " bytesInUse=" << textureObject->GetCurrentCacheBytes()
                 << "/budget=" << textureObject->GetCacheBudgetBytes()
                 << " tiles=" << cs.tilesLoaded
                 << " readMB=" << mbRead
                 << " diskS=" << cs.diskReadTimeS;
    }

    if (textureObject) textureObject->ReleaseAll();
}

static std::shared_ptr<QImage> RenderTexture(RenderingContext& ctx,
                                             std::vector<Mesh::FacePointer>& fvec,
                                             Mesh &m, TextureObjectHandle textureObject,
                                             bool filter, RenderMode imode,
                                             int textureWidth, int textureHeight)
{
    ensure(textureObject && textureObject->ArraySize() > 0);
    auto WTCSh = GetWedgeTexCoordStorageAttribute(m);

    OpenGLFunctionsHandle glFuncs = ctx.glFuncs;
    glFuncs->glUseProgram(ctx.program);
    glFuncs->glBindVertexArray(ctx.vao);
    CHECK_GL_ERROR();

    // Query GL limits and determine output render-tile size
    GLint maxTexSize = 0;
    GLint maxRenderbufferSize = 0;
    GLint maxViewportDims[2] = {0, 0};
    glFuncs->glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
    glFuncs->glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRenderbufferSize);
    glFuncs->glGetIntegerv(GL_MAX_VIEWPORT_DIMS, maxViewportDims);
    int maxSide = std::min(std::min(maxTexSize, maxRenderbufferSize), std::min(maxViewportDims[0], maxViewportDims[1]));
    int outTileWMax = std::min(textureWidth, maxSide);
    int outTileHMax = std::min(textureHeight, maxSide);

    struct OutputTile {
        int x, y, w, h;
        std::vector<Mesh::FacePointer> faces;
    };
    std::vector<OutputTile> outputTiles;
    for (int y = 0; y < textureHeight; y += outTileHMax) {
        int th = std::min(outTileHMax, textureHeight - y);
        for (int x = 0; x < textureWidth; x += outTileWMax) {
            int tw = std::min(outTileWMax, textureWidth - x);
            outputTiles.push_back({x, y, tw, th, {}});
        }
    }

    // Spatial face partitioning: assign faces to output tiles they intersect (output UV space)
    for (auto fptr : fvec) {
        vcg::Box2d bbox;
        for (int i = 0; i < 3; ++i) bbox.Add(fptr->cWT(i).P());
        // Be conservative about Y axis convention (some parts of the pipeline use top-down image space).
        // If we get the convention wrong, we might incorrectly drop faces for a tile and leave holes.
        vcg::Box2d bboxFlipY(
            vcg::Point2d(bbox.min.X(), 1.0 - bbox.max.Y()),
            vcg::Point2d(bbox.max.X(), 1.0 - bbox.min.Y()));

        for (auto& otile : outputTiles) {
            vcg::Box2d otileBox(
                vcg::Point2d((double)otile.x / textureWidth, (double)otile.y / textureHeight),
                vcg::Point2d((double)(otile.x + otile.w) / textureWidth, (double)(otile.y + otile.h) / textureHeight));
            if (bbox.Collide(otileBox) || bboxFlipY.Collide(otileBox)) otile.faces.push_back(fptr);
        }
    }

    std::shared_ptr<QImage> textureImage = std::make_shared<QImage>(textureWidth, textureHeight, QImage::Format_ARGB32);
    if (textureImage->isNull()) {
        LOG_ERR << "[DIAG] FATAL: QImage allocation FAILED.";
        std::exit(-1);
    }
    textureImage->fill(qRgba(0, 0, 0, 255));

    const int tileSize = textureObject->TileSize();
    const int border = textureObject->TileBorder();
    const int physTileSize = textureObject->PhysicalTileSize();

    // Bind sampler units once
    glFuncs->glUniform1i(ctx.loc_tile_cache, 0);
    glFuncs->glUniform1i(ctx.loc_page_table, 1);
    glFuncs->glUniform1f(ctx.loc_tile_size, (float)tileSize);
    glFuncs->glUniform1f(ctx.loc_border, (float)border);
    glFuncs->glUniform2f(ctx.loc_tile_tex_size, (float)physTileSize, (float)physTileSize);

    int renderMode = 0;
    if (imode == Cubic) renderMode = 1;
    else if (imode == FaceColor) renderMode = 2;

    // Render output tile-by-tile (GPU limits), but sample inputs via page-table virtual texturing
    for (auto& otile : outputTiles) {
        if (otile.faces.empty()) continue;

        ctx.prepareRenderTarget(otile.w, otile.h);
        glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, ctx.fbo);
        glFuncs->glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glFuncs->glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glFuncs->glClear(GL_COLOR_BUFFER_BIT);

        glFuncs->glUniform2f(ctx.loc_tile_min, (float)otile.x / textureWidth, (float)otile.y / textureHeight);
        glFuncs->glUniform2f(ctx.loc_tile_scale, (float)otile.w / textureWidth, (float)otile.h / textureHeight);

        // Group faces by source texture index (input)
        std::unordered_map<int, std::vector<Mesh::FacePointer>> bySrc;
        bySrc.reserve(64);
        for (auto fptr : otile.faces) {
            int ti = WTCSh[fptr].tc[0].N();
            if (ti < 0 || ti >= (int)textureObject->ArraySize()) continue;
            bySrc[ti].push_back(fptr);
        }

        for (auto &entry : bySrc) {
            int ti = entry.first;
            auto &faces = entry.second;
            if (faces.empty()) continue;

            const int srcW = textureObject->TextureWidth(ti);
            const int srcH = textureObject->TextureHeight(ti);
            if (srcW <= 0 || srcH <= 0) continue;

            const int tilesX = (srcW + tileSize - 1) / tileSize;
            const int tilesY = (srcH + tileSize - 1) / tileSize;

            // Bind VT resources and per-texture uniforms
            textureObject->BindVirtualTexturing(ti, 0, 1);
            glFuncs->glUniform2f(ctx.loc_src_texture_size, (float)srcW, (float)srcH);
            glFuncs->glUniform1i(ctx.loc_render_mode, renderMode);

            struct FaceReq {
                Mesh::FacePointer f;
                std::vector<std::pair<int,int>> tiles; // (tx,ty) in .rawtile space (top-down)
            };
            std::vector<FaceReq> freqs;
            freqs.reserve(faces.size());

            for (auto fptr : faces) {
                vcg::Box2d srcBox;
                for (int k = 0; k < 3; ++k) srcBox.Add(WTCSh[fptr].tc[k].P());

                double minX = std::max(0.0, std::min((double)srcW - 1.0, srcBox.min.X()));
                double maxX = std::max(0.0, std::min((double)srcW - 1.0, srcBox.max.X()));
                double minY = std::max(0.0, std::min((double)srcH - 1.0, srcBox.min.Y()));
                double maxY = std::max(0.0, std::min((double)srcH - 1.0, srcBox.max.Y()));

                int tx0 = std::max(0, std::min(tilesX - 1, (int)std::floor(minX / tileSize)));
                int tx1 = std::max(0, std::min(tilesX - 1, (int)std::floor(maxX / tileSize)));

                // Convert bottom-up pixel Y to top-down file Y for tile addressing
                double fileYTop = (double)srcH - maxY;
                double fileYBottom = (double)srcH - minY;
                int ty0 = std::max(0, std::min(tilesY - 1, (int)std::floor(fileYTop / tileSize)));
                int ty1 = std::max(0, std::min(tilesY - 1, (int)std::floor(fileYBottom / tileSize)));

                FaceReq fr;
                fr.f = fptr;
                fr.tiles.reserve((tx1 - tx0 + 1) * (ty1 - ty0 + 1));
                for (int ty = ty0; ty <= ty1; ++ty) {
                    for (int tx = tx0; tx <= tx1; ++tx) {
                        fr.tiles.push_back({tx, ty});
                    }
                }
                freqs.push_back(std::move(fr));
            }

            std::vector<char> done(freqs.size(), 0);
            size_t remaining = freqs.size();

            auto t_draw_start = std::chrono::high_resolution_clock::now();
            while (remaining > 0) {
                // Build tile demand counts among remaining faces
                std::unordered_map<uint64_t, int> demand;
                demand.reserve(4096);
                for (size_t fi = 0; fi < freqs.size(); ++fi) {
                    if (done[fi]) continue;
                    for (auto &t : freqs[fi].tiles) {
                        uint64_t key = (uint64_t(uint32_t(t.first)) << 32) | uint64_t(uint32_t(t.second));
                        demand[key] += 1;
                    }
                }

                std::vector<std::pair<uint64_t, int>> dv;
                dv.reserve(demand.size());
                for (auto &d : demand) dv.push_back({d.first, d.second});
                std::sort(dv.begin(), dv.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

                int cap = textureObject->TileCacheLayers();
                if (cap <= 0) cap = 1024;
                int pinCount = std::min((int)dv.size(), cap);

                std::vector<std::pair<int,int>> pins;
                pins.reserve(pinCount);
                for (int k = 0; k < pinCount; ++k) {
                    uint32_t tx = (uint32_t)(dv[k].first >> 32);
                    uint32_t ty = (uint32_t)(dv[k].first & 0xFFFFFFFFu);
                    pins.push_back({(int)tx, (int)ty});
                }

                // Load pinned tiles (protect during eviction)
                textureObject->BeginTilePinning(ti, pins);
                for (auto &t : pins) textureObject->EnsureTileResident(ti, t.first, t.second);
                textureObject->EndTilePinning();

                // Find ready faces (all required tiles resident)
                std::vector<size_t> ready;
                ready.reserve(freqs.size());
                for (size_t fi = 0; fi < freqs.size(); ++fi) {
                    if (done[fi]) continue;
                    bool ok = true;
                    for (auto &t : freqs[fi].tiles) {
                        if (!textureObject->IsTileResident(ti, t.first, t.second)) { ok = false; break; }
                    }
                    if (ok) ready.push_back(fi);
                }

                if (ready.empty()) {
                    // Fallback: force-load tiles for one face to guarantee forward progress
                    size_t first = 0;
                    while (first < freqs.size() && done[first]) ++first;
                    if (first >= freqs.size()) break;
                    textureObject->BeginTilePinning(ti, freqs[first].tiles);
                    for (auto &t : freqs[first].tiles) textureObject->EnsureTileResident(ti, t.first, t.second);
                    textureObject->EndTilePinning();
                    ready.push_back(first);
                }

                // Build VBO for ready faces and draw them
                auto t_vbo_start = std::chrono::high_resolution_clock::now();
                glFuncs->glBindBuffer(GL_ARRAY_BUFFER, ctx.vertexbuf);
                size_t bufferSize = ready.size() * 15 * sizeof(float);
                glFuncs->glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_STREAM_DRAW);
                float *vp = (float *)glFuncs->glMapBufferRange(GL_ARRAY_BUFFER, 0, bufferSize, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
                ensure(vp != nullptr);
                for (auto fi : ready) {
                    Mesh::FacePointer fptr = freqs[fi].f;
                    for (int k = 0; k < 3; ++k) {
                        *vp++ = (float)fptr->cWT(k).U();
                        *vp++ = (float)fptr->cWT(k).V();
                        vcg::Point2d inuvPix = WTCSh[fptr].tc[k].P();
                        *vp++ = (float)(inuvPix.X() / (double)srcW);
                        *vp++ = (float)(inuvPix.Y() / (double)srcH);

                        unsigned char *colorptr = (unsigned char *) vp;
                        *colorptr++ = fptr->C()[0];
                        *colorptr++ = fptr->C()[1];
                        *colorptr++ = fptr->C()[2];
                        *colorptr++ = fptr->C()[3];
                        vp++;
                    }
                }
                glFuncs->glUnmapBuffer(GL_ARRAY_BUFFER);
                auto t_vbo_end = std::chrono::high_resolution_clock::now();
                ctx.total_vbo_s += std::chrono::duration<double>(t_vbo_end - t_vbo_start).count();

                glFuncs->glDrawArrays(GL_TRIANGLES, 0, (GLsizei)ready.size() * 3);
                CHECK_GL_ERROR();

                for (auto fi : ready) {
                    if (!done[fi]) { done[fi] = 1; remaining--; }
                }
            }
            auto t_draw_end = std::chrono::high_resolution_clock::now();
            ctx.total_draw_s += std::chrono::duration<double>(t_draw_end - t_draw_start).count();
        }

        // Read back this output tile into the final image
        glFuncs->glReadBuffer(GL_COLOR_ATTACHMENT0);
        glFuncs->glPixelStorei(GL_PACK_ALIGNMENT, 4);
        int rowPixels = textureImage->bytesPerLine() / 4;
        glFuncs->glPixelStorei(GL_PACK_ROW_LENGTH, rowPixels);
        glFuncs->glPixelStorei(GL_PACK_SKIP_ROWS, textureHeight - (otile.y + otile.h));
        glFuncs->glPixelStorei(GL_PACK_SKIP_PIXELS, otile.x);
        auto t_read_start = std::chrono::high_resolution_clock::now();
        glFuncs->glReadPixels(0, 0, otile.w, otile.h, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->bits());
        auto t_read_end = std::chrono::high_resolution_clock::now();
        ctx.total_read_s += std::chrono::duration<double>(t_read_end - t_read_start).count();
    }

    // Reset pixel store state
    glFuncs->glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glFuncs->glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glFuncs->glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (filter) vcg::PullPush(*textureImage, qRgba(0, 0, 0, 255));

    return textureImage;
}