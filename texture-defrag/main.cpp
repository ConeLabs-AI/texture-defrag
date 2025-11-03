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

#include "mesh.h"

#include "timer.h"
#include "texture_object.h"
#include "texture_optimization.h"
#include "packing.h"
#include "logging.h"
#include "utils.h"
#include "mesh_attribute.h"
#include "seam_remover.h"
#include "texture_rendering.h"

#include <wrap/io_trimesh/io_mask.h>
#include <wrap/system/qgetopt.h>

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/update/color.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>

#include <omp.h>

#include <QApplication>
#include <QImage>
#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QOpenGLContext>
#include <QSurfaceFormat>
#include <QOffscreenSurface>
#include <QOpenGLFunctions>

#include <cmath>

struct Args {
    double m = 2.0;
    double b = 0.2;
    double d = 0.5;
    double g = 0.025;
    double u = 0.0;
    double a = 5.0;
    double t = 0.0;
    std::string infile = "";
    std::string outfile = "";
    int r = 4;
    int l = 0;
    double c = 8.0; // texture GPU cache budget in GB
    double p = 8.0; // packing rasterization cache budget in GB
};

void PrintArgsUsage(const char *binary);
bool ParseOption(const std::string& option, const std::string& argument, Args *args);
Args ParseArgs(int argc, char *argv[]);

void EnsureOpenGLContextOrExit(std::unique_ptr<QOpenGLContext>& context, std::unique_ptr<QOffscreenSurface>& surface);

int main(int argc, char *argv[])
{
    // Make sure the executable directory is added to Qt's library path
    QApplication app(argc, argv);

    // Persistent OpenGL context and offscreen surface for the whole application lifetime
    std::unique_ptr<QOpenGLContext> mainContext;
    std::unique_ptr<QOffscreenSurface> mainSurface;

    AlgoParameters ap;

    Args args = ParseArgs(argc, argv);

    ap.matchingThreshold = args.m;
    ap.boundaryTolerance = args.b;
    ap.distortionTolerance = args.d;
    ap.globalDistortionThreshold = args.g;
    ap.UVBorderLengthReduction = args.u;
    ap.offsetFactor = args.a;
    ap.timelimit = args.t;
    ap.rotationNum = args.r;

    LOG_INIT(args.l);

    LOG_INFO << "Verifying OpenGL context availability...";
    EnsureOpenGLContextOrExit(mainContext, mainSurface);

#ifdef _OPENMP
    LOG_INFO << "OpenMP is enabled.";
    LOG_INFO << "Number of available processors: " << omp_get_num_procs();
    LOG_INFO << "Max threads: " << omp_get_max_threads();
#else
    LOG_INFO << "OpenMP is not enabled.";
#endif

    Mesh m;
    TextureObjectHandle textureObject;
    int loadMask;

    Timer t;
    std::map<std::string, double> timings;

    if (LoadMesh(args.infile.c_str(), m, textureObject, loadMask) == false) {
        LOG_ERR << "Failed to open mesh";
        std::exit(-1);
    }
    timings["Load mesh"] = t.TimeSinceLastCheck();

    // Configure GPU texture cache budget
    if (textureObject) {
        textureObject->SetCacheBudgetGB(args.c);
        LOG_INFO << "Texture GPU cache budget configured to " << args.c << " GB";
    }

    // Configure packing rasterization cache budget
    {
        std::size_t rasterCacheBytes = (args.p <= 0.0)
            ? 0
            : static_cast<std::size_t>(args.p * 1024.0 * 1024.0 * 1024.0);
        SetRasterizerCacheMaxBytes(rasterCacheBytes);
        LOG_INFO << "Packing rasterization cache budget configured to " << args.p << " GB";
    }

    LOG_INFO << "[DIAG] Input mesh loaded: " << m.FN() << " faces, " << m.VN() << " vertices.";

    ensure(loadMask & tri::io::Mask::IOM_WEDGTEXCOORD);
    tri::UpdateTopology<Mesh>::FaceFace(m);

    tri::UpdateNormal<Mesh>::PerFaceNormalized(m);
    tri::UpdateNormal<Mesh>::PerVertexNormalized(m);

    ScaleTextureCoordinatesToImage(m, textureObject);

    LOG_VERBOSE << "Preparing mesh...";

    int vndupIn;
    PrepareMesh(m, &vndupIn);
    ComputeWedgeTexCoordStorageAttribute(m);

    GraphHandle graph = ComputeGraph(m, textureObject);
    timings["Mesh preparation & Graph computation"] = t.TimeSinceLastCheck();

    std::map<RegionID, bool> flipped;
    for (auto& c : graph->charts)
        flipped[c.first] = c.second->UVFlipped();

    double inputMP = textureObject->GetResolutionInMegaPixels();
    int inputCharts = graph->Count();
    double inputUVLen = graph->BorderUV();

    // ensure all charts are oriented coherently, and then store the wtc attribute
    ReorientCharts(graph);

    std::map<ChartHandle, int> anchorMap;
    AlgoStateHandle state = InitializeState(graph, ap);

    GreedyOptimization(graph, state, ap);
    timings["Greedy optimization"] = t.TimeSinceLastCheck();
    int vndupOut;

    std::string savename = args.outfile;
    if (savename == "")
        savename = "out_" + m.name;
    if (savename.substr(savename.size() - 3, 3) == "fbx")
        savename.append(".obj");

    Finalize(graph, savename, &vndupOut);
    timings["Finalize"] = t.TimeSinceLastCheck();

    double zeroResamplingFraction = 0;

    bool colorize = true;

    if (colorize)
        tri::UpdateColor<Mesh>::PerFaceConstant(m, vcg::Color4b(91, 130, 200, 255));

    LOG_INFO << "Rotating charts...";
    double zeroResamplingMeshArea = 0;
    for (auto entry : graph->charts) {
        ChartHandle chart = entry.second;
        double zeroResamplingChartArea;
        int anchor = RotateChartForResampling(chart, state->changeSet, flipped, colorize, &zeroResamplingChartArea);
        if (anchor != -1) {
            anchorMap[chart] = anchor;
            zeroResamplingMeshArea += zeroResamplingChartArea;
        }
    }
    timings["Chart rotation"] = t.TimeSinceLastCheck();
    zeroResamplingFraction = zeroResamplingMeshArea / graph->Area3D();

    LOG_INFO << "[VALIDATION] Checking graph and mesh integrity post-optimization...";
    int emptyCharts = 0;
    for (auto const& [id, chart] : graph->charts) {
        if (chart == nullptr) {
            LOG_ERR << "[VALIDATION] CRITICAL: Found null chart handle in graph!";
        } else if (chart->fpVec.empty()) {
            emptyCharts++;
        }
        for (auto fptr : chart->fpVec) {
            if (fptr == nullptr || fptr->IsD()) {
                 LOG_ERR << "[VALIDATION] CRITICAL: Chart " << id << " contains an invalid face pointer!";
            }
        }
    }
    if(emptyCharts > 0) LOG_WARN << "[VALIDATION] Found " << emptyCharts << " charts with zero faces.";

    int nonManifoldVerts = vcg::tri::Clean<Mesh>::CountNonManifoldVertexFF(m);
    if (nonManifoldVerts > 0) {
        LOG_WARN << "[VALIDATION] Mesh has " << nonManifoldVerts << " non-manifold vertices after optimization.";
    } else {
        LOG_INFO << "[VALIDATION] Mesh is manifold. Integrity check passed.";
    }

    state.reset();

    int outputCharts = graph->Count();
    double outputUVLen = graph->BorderUV();

    // pack the atlas

    // first discard zero-area charts
    std::vector<ChartHandle> chartsToPack;
    for (auto& entry : graph->charts) {
        double a = entry.second->AreaUV();
        if (std::isfinite(a) && a > 0.0) {
            chartsToPack.push_back(entry.second);
        } else {
            for (auto fptr : entry.second->fpVec) {
                for (int j = 0; j < fptr->VN(); ++j) {
                    fptr->V(j)->T().P() = Point2d::Zero();
                    fptr->V(j)->T().N() = 0;
                    fptr->WT(j).P() = Point2d::Zero();
                    fptr->WT(j).N() = 0;
                }
            }
        }
    }

    LOG_INFO << "Packing atlas of size " << chartsToPack.size();

    std::vector<TextureSize> texszVec;
    int npacked = Pack(chartsToPack, textureObject, texszVec, ap, anchorMap);
    timings["Packing"] = t.TimeSinceLastCheck();

    LOG_INFO << "Packed " << npacked << " charts in " << timings["Packing"] << " seconds";

    LOG_INFO << "[DIAG] Packing function finished.";
    if (npacked < (int) chartsToPack.size()) {
        LOG_ERR << "[VALIDATION] Not all charts were packed! Expected " << chartsToPack.size() << ", got " << npacked;
        // The original code exits here, which is correct. This just adds a clearer log.
        std::exit(-1);
    }

    int64_t totalNewTexturePixels = 0;
    for (const auto& sz : texszVec) {
        totalNewTexturePixels += (int64_t)sz.w * sz.h;
    }
    double totalNewTextureMB = (totalNewTexturePixels * 4.0) / (1024.0 * 1024.0);
    LOG_INFO << "[DIAG] Total texture memory to be allocated by rendering: " << totalNewTextureMB << " MB";

    if (npacked < (int) chartsToPack.size()) {
        LOG_ERR << "Not all charts were packed (" << chartsToPack.size() << " charts, " << npacked << " packed)";
        std::exit(-1);
    }

    LOG_INFO << "Trimming texture...";

    TrimTexture(m, texszVec, false);
    timings["Texture trimming"] = t.TimeSinceLastCheck();

    LOG_INFO << "Shifting charts...";

    IntegerShift(m, chartsToPack, texszVec, anchorMap, flipped);
    timings["Chart shifting"] = t.TimeSinceLastCheck();

    LOG_INFO << "Rendering texture...";

    RenderTextureAndSave(savename, m, textureObject, texszVec, false, RenderMode::Linear);
    timings["Texture rendering"] = t.TimeSinceLastCheck();

    double outputMP;
    {
        int64_t totArea = 0;
        for (const auto& sz : texszVec) {
            totArea += (int64_t)sz.w * sz.h;
        }
        outputMP = totArea / 1000000.0;
    }

    LOG_INFO << "InputVert " << m.VN();
    LOG_INFO << "InputVertDup " << vndupIn;
    LOG_INFO << "OutputVertDup " << vndupOut;
    LOG_INFO << "InputCharts " << inputCharts;
    LOG_INFO << "OutputCharts " << outputCharts;
    LOG_INFO << "InputUVLen " << inputUVLen;
    LOG_INFO << "OutputUVLen " << outputUVLen;
    LOG_INFO << "InputMP " << inputMP;
    LOG_INFO << "OutputMP " << outputMP;
    LOG_INFO << "RelativeMPChange " << ((outputMP - inputMP) / inputMP);
    LOG_INFO << "ZeroResamplingFraction " << zeroResamplingFraction;

    LOG_INFO << "Saving mesh file...";

    if (SaveMesh(savename.c_str(), m, {}, true) == false)
        LOG_ERR << "Model not saved correctly";
    timings["Saving mesh"] = t.TimeSinceLastCheck();

    LOG_INFO << "--- Timings ---";
    for (const auto& timing : timings) {
        LOG_INFO << timing.first << ": " << timing.second << "s";
    }
    LOG_INFO << "Processing took " << t.TimeElapsed() << " seconds";

    return 0;
}

void PrintArgsUsage(const char *binary) {
    Args def;
    std::cout << "Usage: " << binary << " MESHFILE [-mbdgutao]" << std::endl;
    std::cout << std::endl;
    std::cout << "MESHFILE specifies the input mesh file (supported formats are obj, ply and fbx)" << std::endl;
    std::cout << std::endl;
    std::cout << "-m  <val>      " << "Matching error tolerance when attempting merge operations." << " (default: " << def.m << ")" << std::endl;
    std::cout << "-b  <val>      " << "Maximum tolerance on the seam-length to chart-perimeter ratio when attempting merge operations. Range is [0,1]." << " (default: " << def.b << ")" << std::endl;
    std::cout << "-d  <val>      " << "Local ARAP distortion tolerance when performing the local UV optimization." << " (default: " << def.d << ")" << std::endl;
    std::cout << "-g  <val>      " << "Global ARAP distortion tolerance when performing the local UV optimization." << " (default: " << def.g << ")" << std::endl;
    std::cout << "-u  <val>      " << "UV border reduction target in percentage relative to the input. Range is [0,1]." << " (default: " << def.u << ")" << std::endl;
    std::cout << "-a  <val>      " << "Alpha parameter to control the UV optimization area size." << " (default: " << def.a << ")" << std::endl;
    std::cout << "-t  <val>      " << "Time-limit for the atlas clustering (in seconds)." << " (default: " << def.t << ")" << std::endl;
    std::cout << "-o  <val>      " << "Output mesh file. Supported formats are obj and ply." << " (default: out_MESHFILE" << ")" << std::endl;
    std::cout << "-r  <val>      " << "Number of rotations to try (e.g., 4 for 0/90/180/270, 1 for no rotation). If > 1, must be multiple of 4." << " (default: " << def.r << ")" << std::endl;
    std::cout << "-l  <val>      " << "Logging level. 0 for minimal verbosity, 1 for verbose output, 2 for debug output." << " (default: " << def.l << ")" << std::endl;
    std::cout << "-c  <val>      " << "Texture GPU cache budget in GB. Set 0 for unlimited." << " (default: " << def.c << ")" << std::endl;
    std::cout << "-p  <val>      " << "Packing rasterization cache budget in GB. Set 0 for unlimited." << " (default: " << def.p << ")" << std::endl;
}

bool ParseOption(const std::string& option, const std::string& argument, Args *args)
{
    ensure(option.size() == 2);
    if (option[1] == 'o') {
        args->outfile = argument;
        return true;
    }
    if (option[1] == 'l') {
        args->l = std::stoi(argument);
        if (args->l >= 0)
            return true;
        else {
            std::cerr << "Logging level must be positive" << std::endl << std::endl;
            return false;
        }
    }
    try {
        switch (option[1]) {
            case 'm': args->m = std::stod(argument); break;
            case 'b': args->b = std::stod(argument); break;
            case 'd': args->d = std::stod(argument); break;
            case 'g': args->g = std::stod(argument); break;
            case 'u': args->u = std::stod(argument); break;
            case 'a': args->a = std::stod(argument); break;
            case 't': args->t = std::stod(argument); break;
            case 'r': args->r = std::stoi(argument); break;
            case 'c': args->c = std::stod(argument); break;
            case 'p': args->p = std::stod(argument); break;
            default:
                std::cerr << "Unrecognized option " << option << std::endl << std::endl;
                return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error while parsing option `" << option << " " << argument << "`"; std::cerr << ": " << e.what() << std::endl << std::endl;;
        return false;
    }
    return true;
}

Args ParseArgs(int argc, char *argv[])
{
    if (argc < 2) {
        PrintArgsUsage(argv[0]);
        std::exit(-1);
    }

    Args args;

    for (int i = 0; i < argc; ++i) {
        std::string argi(argv[i]);
        if (argi[0] == '-' && argi.size() == 2) {
            i++;
            if (i >= argc) {
                std::cerr << "Missing argument for option " << argi << std::endl << std::endl;
                PrintArgsUsage(argv[0]);
                std::exit(-1);
            } else {
                if (!ParseOption(argi, std::string(argv[i]), &args)) {
                    PrintArgsUsage(argv[0]);
                    std::exit(-1);
                }
            }
        } else {
            args.infile = argi;
        }
    }

    if (args.infile == "") {
        std::cerr << "Missing input mesh argument" << std::endl << std::endl;
        PrintArgsUsage(argv[0]);
        std::exit(-1);
    }

    return args;
}

void EnsureOpenGLContextOrExit(std::unique_ptr<QOpenGLContext>& context, std::unique_ptr<QOffscreenSurface>& surface)
{
    QSurfaceFormat format;
    format.setVersion(4, 1);
    format.setProfile(QSurfaceFormat::CoreProfile);

    context.reset(new QOpenGLContext());
    context->setFormat(format);
    if (!context->create()) {
        LOG_ERR << "Failed to create OpenGL context. Ensure a headless backend is available (e.g., set QT_QPA_PLATFORM=offscreen QT_OPENGL=egl) or a valid X/GLX display.";
        std::exit(-1);
    }

    surface.reset(new QOffscreenSurface());
    surface->setFormat(context->format());
    surface->create();
    if (!surface->isValid()) {
        LOG_ERR << "Failed to create offscreen surface for OpenGL.";
        std::exit(-1);
    }

    if (!context->makeCurrent(surface.get())) {
        LOG_ERR << "Failed to make OpenGL context current on offscreen surface.";
        std::exit(-1);
    }

    QOpenGLFunctions *f = context->functions();
    if (f) {
        f->initializeOpenGLFunctions();
        const char *vendor = reinterpret_cast<const char *>(f->glGetString(GL_VENDOR));
        const char *renderer = reinterpret_cast<const char *>(f->glGetString(GL_RENDERER));
        const char *version = reinterpret_cast<const char *>(f->glGetString(GL_VERSION));
        LOG_INFO << "[GL] Vendor: " << (vendor ? vendor : "unknown")
                 << " Renderer: " << (renderer ? renderer : "unknown")
                 << " Version: " << (version ? version : "unknown");
    }
}