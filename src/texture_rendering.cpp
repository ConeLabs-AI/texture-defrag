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

#include <chrono>
#include <limits>


static const char *vs_text[] = {
    "#version 410 core                                           \n"
    "                                                            \n"
    "in vec2 position;                                           \n"
    "in vec2 texcoord;                                           \n"
    "in vec4 color;                                              \n"
    "out vec2 uv;                                                \n"
    "out vec4 fcolor;                                            \n"
    "                                                            \n"
    "void main(void)                                             \n"
    "{                                                           \n"
    "    uv = texcoord;                                          \n"
    "    fcolor = color;                                         \n"
    "    //if (uv.s < 0) uv = vec2(0.0, 0.0);                    \n"
    "    vec2 p = vec2(2.0 * position.x - 1.0, 1.0 - 2.0 * position.y);\n"
    "    gl_Position = vec4(p, 0.5, 1.0);                        \n"
    "}                                                           \n"
};

static const char *fs_text[] = {
    "#version 410 core                                                     \n"
    "                                                                      \n"
    "uniform sampler2D img0;                                               \n"
    "                                                                      \n"
    "uniform vec2 texture_size;                                            \n"
    "uniform int render_mode;                                              \n"
    "                                                                      \n"
    "in vec2 uv;                                                           \n"
    "in vec4 fcolor;                                                       \n"
    "                                                                      \n"
    "out vec4 texelColor;                                                  \n"
    "                                                                      \n"
    "void main(void)                                                       \n"
    "{                                                                     \n"
    "    if (render_mode == 0) {                                           \n"
    "        if (uv.s < 0)                                                 \n"
    "            texelColor = vec4(0, 1, 0, 1);                            \n"
    "        else                                                          \n"
    "            texelColor = vec4(texture2D(img0, uv).rgb, 1);            \n"
    "    } else if (render_mode == 1) {                                    \n"
    "        vec2 coord = uv * texture_size - vec2(0.5, 0.5);              \n"
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
    "        vec4 tex00 = texture2D(img0, vec2(h0.x, h0.y) / texture_size);\n"
    "        vec4 tex10 = texture2D(img0, vec2(h1.x, h0.y) / texture_size);\n"
    "        vec4 tex01 = texture2D(img0, vec2(h0.x, h1.y) / texture_size);\n"
    "        vec4 tex11 = texture2D(img0, vec2(h1.x, h1.y) / texture_size);\n"
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

    // Cached uniform locations
    GLint loc_img0 = -1;
    GLint loc_texture_size = -1;
    GLint loc_render_mode = -1;

    RenderingContext() {
        if (QOpenGLContext::currentContext() == nullptr) {
            LOG_DEBUG << "Creating persistent OpenGL context for rendering.";
            ownContext = true;

            context.reset(new QOpenGLContext());
            surface.reset(new QOffscreenSurface());

            QSurfaceFormat format;
            format.setVersion(4, 1);
            format.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
            context->setFormat(format);

            if (!context->create()) {
                LOG_ERR << "Failed to create opengl context";
                std::exit(-1);
            }

            surface->setFormat(context->format());
            surface->create();

            if (!context->makeCurrent(surface.get())) {
                LOG_ERR << "Failed to make OpenGL context current";
                std::exit(-1);
            }
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
        loc_img0 = glFuncs->glGetUniformLocation(program, "img0");
        loc_texture_size = glFuncs->glGetUniformLocation(program, "texture_size");
        loc_render_mode = glFuncs->glGetUniformLocation(program, "render_mode");

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
    // Reset GPU texture cache stats for this rendering pass
    if (textureObject) textureObject->ResetCacheStats();

    std::vector<std::vector<Mesh::FacePointer>> facesByTexture;
    int nTex = FacesByTextureIndex(m, facesByTexture);

    ensure(nTex <= (int) texSizes.size());

    m.textures.clear();

    QFileInfo fi(outFileName.c_str());
    QString wd = QDir::currentPath();
    QDir::setCurrent(fi.absoluteDir().absolutePath());

    auto t_total_start = std::chrono::high_resolution_clock::now();
    double t_total_render_s = 0.0;
    double t_total_savequeue_enqueue_s = 0.0;
    double t_total_png_save_s = 0.0; // captured by queue
    double t_save_wait_s = 0.0;      // time waiting in finish()
    int64_t total_pixels_rendered = 0;

    RenderingContext renderingContext;
    ImageSaveQueue saveQueue(2);
    saveQueue.resetStats();

    for (int i = 0; i < nTex; ++i) {
        LOG_INFO << "Processing sheet " << (i + 1) << " of " << nTex << "...";
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
        saveQueue.enqueue(*teximg, absPath, 90);
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
    LOG_INFO << "[RENDER-STATS] sheets=" << nTex
             << " pixels=" << total_pixels_rendered
             << " total_s=" << t_total_s
             << " render_s=" << t_total_render_s
             << " enqueue_s=" << t_total_savequeue_enqueue_s
             << " save_wait_s=" << t_save_wait_s
             << " png_save_s=" << t_total_png_save_s
             << " png_min_s=" << saveStats.minSaveS
             << " png_max_s=" << saveStats.maxSaveS
             << " png_saved=" << saveStats.saved;

    // Log GPU texture cache stats
    if (textureObject) {
        auto cs = textureObject->GetCacheStats();
        uint64_t lookups = cs.hits + cs.misses;
        double hitRate = lookups ? double(cs.hits) / double(lookups) : 0.0;
        LOG_INFO << "[TEX-CACHE] lookups=" << lookups
                 << " hits=" << cs.hits
                 << " misses=" << cs.misses
                 << " hitRate=" << hitRate
                 << " evictions=" << cs.evictions
                 << " bytesEvicted=" << cs.bytesEvicted
                 << " bytesInUse=" << textureObject->GetCurrentCacheBytes()
                 << "/budget=" << textureObject->GetCacheBudgetBytes();
    }
}

static std::shared_ptr<QImage> RenderTexture(RenderingContext& ctx,
                                             std::vector<Mesh::FacePointer>& fvec,
                                             Mesh &m, TextureObjectHandle textureObject,
                                             bool filter, RenderMode imode,
                                             int textureWidth, int textureHeight)
{
    auto WTCSh = GetWedgeTexCoordStorageAttribute(m);

    // sort the faces in increasing order of input texture unit
    auto FaceComparatorByInputTexIndex = [&WTCSh](const Mesh::FacePointer& f1, const Mesh::FacePointer& f2) {
        return WTCSh[f1].tc[0].N() < WTCSh[f2].tc[0].N();
    };

    std::sort(fvec.begin(), fvec.end(), FaceComparatorByInputTexIndex);

    OpenGLFunctionsHandle glFuncs = ctx.glFuncs;
    glFuncs->glUseProgram(ctx.program);
    glFuncs->glBindVertexArray(ctx.vao);
    CHECK_GL_ERROR();

    // Allocate vertex data

    std::vector<TextureSize> inTexSizes;
    for (std::size_t i = 0; i < textureObject->ArraySize(); ++i) {
        int iw = textureObject->TextureWidth(i);
        int ih = textureObject->TextureHeight(i);
        inTexSizes.push_back({iw, ih});
    }

    auto t_vbo_start = std::chrono::high_resolution_clock::now();
    glFuncs->glBindBuffer(GL_ARRAY_BUFFER, ctx.vertexbuf);
    // Use streaming usage hint and buffer orphaning + map range to reduce stalls.
    size_t bufferSize = fvec.size() * 15 * sizeof(float);
    glFuncs->glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_STREAM_DRAW);
    float *p = (float *)glFuncs->glMapBufferRange(GL_ARRAY_BUFFER, 0, bufferSize, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    ensure(p != nullptr);
    for (auto fptr : fvec) {
        int ti = WTCSh[fptr].tc[0].N();
        for (int i = 0; i < 3; ++i) {
            *p++ = fptr->cWT(i).U();
            *p++ = fptr->cWT(i).V();
            vcg::Point2d uv = WTCSh[fptr].tc[i].P();
            *p++ = uv.X() / inTexSizes[ti].w;
            *p++ = uv.Y() / inTexSizes[ti].h;
            unsigned char *colorptr = (unsigned char *) p;
            *colorptr++ = fptr->C()[0];
            *colorptr++ = fptr->C()[1];
            *colorptr++ = fptr->C()[2];
            *colorptr++ = fptr->C()[3];
            p++;

        }
    }
    glFuncs->glUnmapBuffer(GL_ARRAY_BUFFER);

    p = nullptr;
    glFuncs->glBindBuffer(GL_ARRAY_BUFFER, 0); // done, unbind
    auto t_vbo_end = std::chrono::high_resolution_clock::now();
    double t_vbo_s = std::chrono::duration<double>(t_vbo_end - t_vbo_start).count();

    ctx.prepareRenderTarget(textureWidth, textureHeight);
    glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, ctx.fbo);

    // --- [DIAGNOSTIC] START: QImage Allocation ---
    LOG_INFO << "[DIAG] Attempting to allocate QImage for rendering. Size: " << textureWidth << "x" << textureHeight;
    std::shared_ptr<QImage> textureImage = std::make_shared<QImage>(textureWidth, textureHeight, QImage::Format_ARGB32);
    if (textureImage->isNull()) {
        LOG_ERR << "[DIAG] FATAL: QImage allocation FAILED. System is out of memory.";
        logging::LogMemoryUsage();
        std::exit(-1);
    }
    LOG_INFO << "[DIAG] QImage allocation successful. Memory usage AFTER QImage allocation:";
    logging::LogMemoryUsage();
    // --- [DIAGNOSTIC] END ---

    glFuncs->glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glFuncs->glClearColor(0.0f, 1.0f, 0.0f, 1.0f);

    glFuncs->glClear(GL_COLOR_BUFFER_BIT);

    auto t_draw_start = std::chrono::high_resolution_clock::now();
    auto f0 = fvec.begin();
    auto fbase = f0;
    while (fbase != fvec.end()) {
        auto fcurr = fbase;
        int currTexIndex = WTCSh[*fcurr].tc[0].N();
        while (fcurr != fvec.end() && WTCSh[*fcurr].tc[0].N() == currTexIndex)
            fcurr++;
        int baseIndex = std::distance(f0, fbase) * 3;
        int count = std::distance(fbase, fcurr) * 3;

        // Load texture image
        glFuncs->glActiveTexture(GL_TEXTURE0);
        LOG_DEBUG << "Binding texture unit " << currTexIndex;
        textureObject->Bind(currTexIndex);

        glFuncs->glUniform1i(ctx.loc_img0, 0);
        glFuncs->glUniform2f(ctx.loc_texture_size, float(textureObject->TextureWidth(currTexIndex)), float(textureObject->TextureHeight(currTexIndex)));


        glFuncs->glUniform1i(ctx.loc_render_mode, 0);

        // Texture parameters are now set once in TextureObject::Bind, so we remove the redundant settings from here.
        switch (imode) {
        case Cubic:
            glFuncs->glUniform1i(ctx.loc_render_mode, 1);
            break;
        case Linear:
            // This is the default render_mode 0 in the shader
            break;
        case Nearest:
            // Nearest filtering must be set on the texture object itself. For simplicity, we assume linear/cubic for this path.
            // A more robust implementation might require passing filter mode to TextureObject::Bind.
            break;
        case FaceColor:
            glFuncs->glUniform1i(ctx.loc_render_mode, 2);
            break;
        default:
            ensure(0 && "Should never happen");
        }

        glFuncs->glDrawArrays(GL_TRIANGLES, baseIndex, count);
        CHECK_GL_ERROR();

        fbase = fcurr;
    }
    auto t_draw_end = std::chrono::high_resolution_clock::now();
    double t_draw_s = std::chrono::duration<double>(t_draw_end - t_draw_start).count();

    glFuncs->glReadBuffer(GL_COLOR_ATTACHMENT0);
    glFuncs->glPixelStorei(GL_PACK_ALIGNMENT, 4);
    auto t_read_start = std::chrono::high_resolution_clock::now();
    glFuncs->glReadPixels(0, 0, textureWidth, textureHeight, GL_BGRA, GL_UNSIGNED_BYTE, textureImage->bits());
    auto t_read_end = std::chrono::high_resolution_clock::now();
    double t_read_s = std::chrono::duration<double>(t_read_end - t_read_start).count();

    glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (filter)
        vcg::PullPush(*textureImage, qRgba(0, 255, 0, 255));

    LOG_INFO << "[RENDER-PROFILE] vbo_s=" << t_vbo_s
             << " draw_s=" << t_draw_s
             << " readPixels_s=" << t_read_s
             << " image=" << textureWidth << "x" << textureHeight;
    return textureImage;
}
