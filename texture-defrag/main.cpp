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
#include "seam_straightening.h"
#include "texture_conversion.h"

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
#include <algorithm>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <QApplication>
#include <QImage>
#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QFile>
#include <QOpenGLContext>
#include <QSurfaceFormat>
#include <QOffscreenSurface>
#include <QOpenGLFunctions>

#include <cmath>

struct Args {
    double matchingThreshold = 2.0;
    double boundaryTolerance = 0.2;
    double localDistortionTolerance = 0.5;
    double globalDistortionThreshold = 0.025;
    double maxErrorTexels = 0.0;
    double borderReductionTarget = 0.0;
    double alpha = 5.0;
    double timelimit = 0.0;
    double straightenTolerancePixels = 2.0;
    std::string objective = "uv-border"; // or "seam-edges"
    std::string infile = "";
    std::string outfile = "";
    int rotationNum = 4;
    int loggingLevel = 0;
    double gpuCacheGB = 8.0; // texture GPU cache budget in GB
    double packingCacheGB = 8.0; // packing rasterization cache budget in GB
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

    ap.matchingThreshold = args.matchingThreshold;
    ap.boundaryTolerance = args.boundaryTolerance;
    ap.distortionTolerance = args.localDistortionTolerance;
    ap.globalDistortionThreshold = args.globalDistortionThreshold;
    ap.maxErrorTexels = args.maxErrorTexels;
    ap.UVBorderLengthReduction = args.borderReductionTarget;
    ap.offsetFactor = args.alpha;
    ap.timelimit = args.timelimit;
    ap.rotationNum = args.rotationNum;
    ap.minimizeSeamEdges = (args.objective == "seam-edges");

    LOG_INIT(args.loggingLevel);

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
        textureObject->SetCacheBudgetGB(args.gpuCacheGB);
        LOG_INFO << "Texture GPU cache budget configured to " << args.gpuCacheGB << " GB";
    }

    // Configure packing rasterization cache budget
    {
        std::size_t rasterCacheBytes = (args.packingCacheGB <= 0.0)
            ? 0
            : static_cast<std::size_t>(args.packingCacheGB * 1024.0 * 1024.0 * 1024.0);
        SetRasterizerCacheMaxBytes(rasterCacheBytes);
        LOG_INFO << "Packing rasterization cache budget configured to " << args.packingCacheGB << " GB";
    }

    LOG_INFO << "[DIAG] Input mesh loaded: " << m.FN() << " faces, " << m.VN() << " vertices.";

    ensure(loadMask & tri::io::Mask::IOM_WEDGTEXCOORD);
    tri::UpdateTopology<Mesh>::FaceFace(m);

    tri::UpdateNormal<Mesh>::PerFaceNormalized(m);
    tri::UpdateNormal<Mesh>::PerVertexNormalized(m);

    // Pre-scan: enforce normalized UVs in [0,1] at source. If any UV in a chart violates this,
    // skip the entire chart by zeroing its UVs so it is excluded downstream.
    {
        // Ensure topology is available for connected-component traversal
        tri::UpdateTopology<Mesh>::FaceFace(m);

        int skippedChartsNonNormalized = 0;
        {
            GraphHandle preGraph = ComputeGraph(m, textureObject);
            for (auto &entry : preGraph->charts) {
                auto chart = entry.second;
                bool chartOk = true;
                for (auto fptr : chart->fpVec) {
                    for (int i = 0; i < 3; ++i) {
                        const auto &p = fptr->cWT(i).P();
                        if (!std::isfinite(p.X()) || !std::isfinite(p.Y()) || p.X() < 0.0 || p.X() > 1.0 || p.Y() < 0.0 || p.Y() > 1.0) {
                            chartOk = false;
                            break;
                        }
                    }
                    if (!chartOk) break;
                }
                if (!chartOk) {
                    // Skip chart: zero out its UVs so it becomes zero-area and is ignored later
                    for (auto fptr : chart->fpVec) {
                        for (int j = 0; j < fptr->VN(); ++j) {
                            fptr->V(j)->T().P() = Point2d::Zero();
                            fptr->V(j)->T().N() = 0;
                            fptr->WT(j).P() = Point2d::Zero();
                            fptr->WT(j).N() = 0;
                        }
                    }
                    skippedChartsNonNormalized++;
                }
            }
        }

        if (skippedChartsNonNormalized > 0) {
            LOG_WARN << "[VALIDATION] Skipped " << skippedChartsNonNormalized
                     << " charts due to non-normalized or non-finite UVs (expected [0,1]).";
        }
    }

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

    // [DIAG] Log UV/3D area ratio stats before optimization
    {
        std::vector<double> ratios;
        ratios.reserve(graph->charts.size());
        double minR = std::numeric_limits<double>::infinity();
        double maxR = 0.0;
        for (auto &entry : graph->charts) {
            auto chart = entry.second;
            double a3d = chart->Area3D();
            double auv = chart->AreaUV();
            if (a3d <= 0 || !std::isfinite(auv)) continue;
            double r = auv / a3d;
            ratios.push_back(r);
            minR = std::min(minR, r);
            maxR = std::max(maxR, r);
        }
        if (!ratios.empty()) {
            size_t mid = ratios.size() / 2;
            std::nth_element(ratios.begin(), ratios.begin() + mid, ratios.end());
            double med = ratios[mid];
            LOG_INFO << "[DIAG] BEFORE optimization UV/3D area ratio stats: min=" << minR
                     << " median=" << med << " max=" << maxR
                     << " (charts=" << ratios.size() << ")";
        }
    }

    // ensure all charts are oriented coherently, and then store the wtc attribute
    ReorientCharts(graph);

    std::map<ChartHandle, int> anchorMap;
    AlgoStateHandle state = InitializeState(graph, ap);

    GreedyOptimization(graph, state, ap);
    timings["Greedy optimization"] = t.TimeSinceLastCheck();

    double maxRes = (textureObject) ? (double)textureObject->MaxSize() : 1024.0;
    double straightenToleranceEpsilon = args.straightenTolerancePixels;

    if (args.straightenTolerancePixels > 0) {
        LOG_INFO << "Straightening seams...";
        {
            UVDefrag::SeamStraighteningParameters ssp;
            ssp.initialTolerance = straightenToleranceEpsilon;
            LOG_INFO << "[DIAG] Seam straightening texel tolerance: " << args.straightenTolerancePixels << " pixels -> epsilon=" << ssp.initialTolerance << " (resolution=" << maxRes << ")";
            UVDefrag::IntegrateSeamStraightening(graph, ssp);
        }
        timings["Seam straightening"] = t.TimeSinceLastCheck();
    } else {
        LOG_INFO << "Skipping seam straightening (tolerance is zero).";
    }

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
    std::map<ChartHandle, float> chartMultipliers;
    for (auto entry : graph->charts) {
        ChartHandle chart = entry.second;
        double zeroResamplingChartArea;
        int anchor = RotateChartForResampling(chart, {}, flipped, colorize, &zeroResamplingChartArea);
        
        double chartArea3D = chart->Area3D();
        double resampledFraction = 1.0; 
        if (chartArea3D > 1e-12) {
            resampledFraction = 1.0 - (zeroResamplingChartArea / chartArea3D);
        }
        
        if (!std::isfinite(resampledFraction) || resampledFraction > 1.0) resampledFraction = 1.0;
        if (resampledFraction < 0.0) resampledFraction = 0.0;
        
        // Multiplier: fully rigid (fraction=0) -> 1.0, fully resampled (fraction=1) -> sqrt(2)
        float mul = static_cast<float>(std::sqrt(1.0 + resampledFraction));
        
        if (!std::isfinite(mul) || mul < 1.0f) mul = 1.0f;
        if (mul > 1.4143f) mul = 1.4143f; // slightly more than sqrt(2)
        
        chartMultipliers[chart] = mul;

        if (anchor != -1) {
            anchorMap[chart] = anchor;
            zeroResamplingMeshArea += zeroResamplingChartArea;
        }
    }
    timings["Chart rotation"] = t.TimeSinceLastCheck();
    zeroResamplingFraction = zeroResamplingMeshArea / graph->Area3D();

    LOG_INFO << "[VALIDATION] Checking graph and mesh integrity post-optimization...";
    int emptyCharts = 0;
    for (auto const& pair : graph->charts) {
        RegionID id = pair.first;
        ChartHandle chart = pair.second;
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

    // [DIAG] Log UV/3D area ratio stats after optimization (before packing)
    double medianExpansion = 1.0;
    {
        std::vector<double> ratios;
        ratios.reserve(graph->charts.size());
        double minR = std::numeric_limits<double>::infinity();
        double maxR = 0.0;
        for (auto &entry : graph->charts) {
            auto chart = entry.second;
            double a3d = chart->Area3D();
            double auv = chart->AreaUV();
            if (a3d <= 0 || !std::isfinite(auv)) continue;
            double r = auv / a3d;
            ratios.push_back(r);
            minR = std::min(minR, r);
            maxR = std::max(maxR, r);
        }
        if (!ratios.empty()) {
            size_t mid = ratios.size() / 2;
            std::nth_element(ratios.begin(), ratios.begin() + mid, ratios.end());
            medianExpansion = ratios[mid];
            LOG_INFO << "[DIAG] AFTER optimization UV/3D area ratio stats: min=" << minR
                     << " median=" << medianExpansion << " max=" << maxR
                     << " (charts=" << ratios.size() << ")";
        }
    }

    // pack the atlas

    // First discard zero-area charts AND charts with exploded/invalid UVs
    // This catches cases where optimization produced runaway UV coordinates
    std::vector<ChartHandle> chartsToPack;
    int skippedZeroArea = 0;
    int skippedExplodedUV = 0;
    int skippedNonFiniteUV = 0;
    
    // Dynamic UV dimension limit: 4x the max texture dimension
    double maxTexDim = (double)std::max(1024, textureObject->MaxSize());
    double maxReasonableUVDim = 4.0 * maxTexDim;
    
    // Expansion factor threshold: 1000x median is definitely an outlier
    double maxExpansion = (medianExpansion > 1e-9) ? medianExpansion * 1000.0 : 1e20;
    LOG_INFO << "[DIAG] Packing validation thresholds: maxReasonableUVDim=" << maxReasonableUVDim
             << ", maxExpansion=" << maxExpansion;
    
    for (auto& entry : graph->charts) {
        ChartHandle chart = entry.second;
        double a = chart->AreaUV();
        
        // Check for zero/invalid area
        if (!std::isfinite(a) || a <= 0.0) {
            skippedZeroArea++;
            for (auto fptr : chart->fpVec) {
                for (int j = 0; j < fptr->VN(); ++j) {
                    fptr->V(j)->T().P() = Point2d::Zero();
                    fptr->V(j)->T().N() = 0;
                    fptr->WT(j).P() = Point2d::Zero();
                    fptr->WT(j).N() = 0;
                }
            }
            continue;
        }
        
        // Check for exploded/non-finite UV bounding box
        vcg::Box2d uvBox = chart->UVBox();
        bool hasNonFinite = !std::isfinite(uvBox.min.X()) || !std::isfinite(uvBox.min.Y()) ||
                            !std::isfinite(uvBox.max.X()) || !std::isfinite(uvBox.max.Y());
        
        // Use dynamic limit based on texture size
        bool hasExploded = uvBox.DimX() > maxReasonableUVDim || uvBox.DimY() > maxReasonableUVDim;
        
        // Also check expansion factor (UV area vs 3D area) to catch warped charts
        if (!hasExploded && !hasNonFinite && chart->Area3D() > 1e-9) {
            double exp = chart->AreaUV() / chart->Area3D();
            if (exp > maxExpansion) {
                hasExploded = true;
                LOG_WARN << "[DIAG] Chart " << chart->id << " has exploded expansion factor: " << exp
                         << " (threshold=" << maxExpansion << "). Treating as exploded.";
            }
        }
        
        if (hasNonFinite) {
            skippedNonFiniteUV++;
            for (auto fptr : chart->fpVec) {
                for (int j = 0; j < fptr->VN(); ++j) {
                    fptr->V(j)->T().P() = Point2d::Zero();
                    fptr->V(j)->T().N() = 0;
                    fptr->WT(j).P() = Point2d::Zero();
                    fptr->WT(j).N() = 0;
                }
            }
            continue;
        }
        
        if (hasExploded) {
            skippedExplodedUV++;
            double a3d = chart->Area3D();
            double auv = chart->AreaUV();
            LOG_WARN << "[DIAG] Chart " << chart->id << " has exploded UV box: "
                     << uvBox.DimX() << "x" << uvBox.DimY() << " (max=" << maxReasonableUVDim << ")"
                     << ", AreaUV=" << auv << ", Area3D=" << a3d
                     << ", UV/3D ratio=" << (a3d > 0 ? auv / a3d : -1.0) << ". Skipping.";
            for (auto fptr : chart->fpVec) {
                for (int j = 0; j < fptr->VN(); ++j) {
                    fptr->V(j)->T().P() = Point2d::Zero();
                    fptr->V(j)->T().N() = 0;
                    fptr->WT(j).P() = Point2d::Zero();
                    fptr->WT(j).N() = 0;
                }
            }
            continue;
        }
        
        chartsToPack.push_back(chart);
    }
    
    if (skippedZeroArea > 0) {
        LOG_INFO << "[VALIDATION] Skipped " << skippedZeroArea << " charts with zero/invalid area.";
    }
    if (skippedNonFiniteUV > 0) {
        LOG_WARN << "[VALIDATION] Skipped " << skippedNonFiniteUV << " charts with non-finite UV coordinates.";
    }
    if (skippedExplodedUV > 0) {
        LOG_WARN << "[VALIDATION] Skipped " << skippedExplodedUV << " charts with exploded UV coordinates"
                 << " (dim > " << maxReasonableUVDim << " or expansion > " << maxExpansion << ").";
    }

    LOG_INFO << "Packing atlas of size " << chartsToPack.size();

    std::vector<TextureSize> texszVec;
    int npacked = Pack(chartsToPack, textureObject, texszVec, ap, chartMultipliers);
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
    double totalNewTextureGB = totalNewTextureMB / 1024.0;
    LOG_INFO << "[DIAG] Total texture memory to be allocated by rendering: " << totalNewTextureMB << " MB (" << totalNewTextureGB << " GB)";
    
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

    if (textureObject) {
        LOG_INFO << "Cleaning up intermediate .rawtile files...";
        for (size_t i = 0; i < textureObject->ArraySize(); ++i) {
            std::string rawPath = TextureConversion::GetRawTilePath(textureObject->texInfoVec[i].path);
            if (QFile::exists(QString::fromStdString(rawPath))) {
                QFile::remove(QString::fromStdString(rawPath));
            }
        }
    }

    std::stringstream timingsSS;
    timingsSS << "--- Timings ---";
    for (const auto& timing : timings) {
        timingsSS << "\n" << timing.first << ": " << timing.second << "s";
    }
    LOG_INFO << timingsSS.str();
    LOG_INFO << "Processing took " << t.TimeElapsed() << " seconds";

    return 0;
}

void PrintArgsUsage(const char *binary) {
    Args def;
    std::cout << "Usage: " << binary << " MESHFILE [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "MESHFILE specifies the input mesh file (supported formats are obj, ply and fbx)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m, --matching-threshold <val>  " << "Matching error tolerance for merge operations." << " (default: " << def.matchingThreshold << ")" << std::endl;
    std::cout << "  -b, --boundary-tolerance <val>  " << "Max tolerance on seam-length to perimeter ratio [0,1]." << " (default: " << def.boundaryTolerance << ")" << std::endl;
    std::cout << "  -d, --local-distortion <val>    " << "Local ARAP distortion tolerance for UV optimization." << " (default: " << def.localDistortionTolerance << ")" << std::endl;
    std::cout << "  -g, --global-distortion <val>   " << "Global ARAP distortion threshold." << " (default: " << def.globalDistortionThreshold << ")" << std::endl;
    std::cout << "  -e, --max-error-texels <val>    " << "Absolute distortion tolerance in texels (0 disables)." << " (default: " << def.maxErrorTexels << ")" << std::endl;
    std::cout << "  -u, --reduction-target <val>    " << "UV border reduction target percentage [0,1]." << " (default: " << def.borderReductionTarget << ")" << std::endl;
    std::cout << "  -a, --alpha <val>               " << "Alpha parameter for UV optimization area size." << " (default: " << def.alpha << ")" << std::endl;
    std::cout << "  -t, --time-limit <val>          " << "Time-limit for atlas clustering (seconds)." << " (default: " << def.timelimit << ")" << std::endl;
    std::cout << "  -k, --straighten-pixels <val>   " << "Allowable deviation in pixels for seam straightening." << " (default: " << def.straightenTolerancePixels << ")" << std::endl;
    std::cout << "  -j, --objective <val>           " << "Merge objective: uv-border|seam-edges." << " (default: " << def.objective << ")" << std::endl;
    std::cout << "  -o, --output <val>              " << "Output mesh file (obj or ply)." << " (default: out_MESHFILE" << ")" << std::endl;
    std::cout << "  -r, --rotations <val>           " << "Number of rotations to try (must be multiple of 4)." << " (default: " << def.rotationNum << ")" << std::endl;
    std::cout << "  -l, --loglevel <val>            " << "Logging level (0: minimal, 1: verbose, 2: debug)." << " (default: " << def.loggingLevel << ")" << std::endl;
    std::cout << "  -c, --gpu-cache <val>           " << "Texture GPU cache budget in GB (0: unlimited)." << " (default: " << def.gpuCacheGB << ")" << std::endl;
    std::cout << "  -p, --packing-cache <val>       " << "Packing rasterization cache budget in GB (0: unlimited)." << " (default: " << def.packingCacheGB << ")" << std::endl;
}

bool ParseOption(const std::string& option, const std::string& argument, Args *args)
{
    try {
        if (option == "-m" || option == "--matching-threshold") args->matchingThreshold = std::stod(argument);
        else if (option == "-b" || option == "--boundary-tolerance") args->boundaryTolerance = std::stod(argument);
        else if (option == "-d" || option == "--local-distortion") args->localDistortionTolerance = std::stod(argument);
        else if (option == "-g" || option == "--global-distortion") args->globalDistortionThreshold = std::stod(argument);
        else if (option == "-e" || option == "--max-error-texels") args->maxErrorTexels = std::stod(argument);
        else if (option == "-u" || option == "--reduction-target") args->borderReductionTarget = std::stod(argument);
        else if (option == "-a" || option == "--alpha") args->alpha = std::stod(argument);
        else if (option == "-t" || option == "--time-limit") args->timelimit = std::stod(argument);
        else if (option == "-k" || option == "--straighten-pixels") args->straightenTolerancePixels = std::stod(argument);
        else if (option == "-j" || option == "--objective") {
            args->objective = argument;
            if (args->objective != "uv-border" && args->objective != "seam-edges") {
                std::cerr << "Objective must be one of: uv-border, seam-edges" << std::endl << std::endl;
                return false;
            }
        }
        else if (option == "-o" || option == "--output") args->outfile = argument;
        else if (option == "-r" || option == "--rotations") args->rotationNum = std::stoi(argument);
        else if (option == "-l" || option == "--loglevel") {
            args->loggingLevel = std::stoi(argument);
            if (args->loggingLevel < 0) {
                std::cerr << "Logging level must be positive" << std::endl << std::endl;
                return false;
            }
        }
        else if (option == "-c" || option == "--gpu-cache") args->gpuCacheGB = std::stod(argument);
        else if (option == "-p" || option == "--packing-cache") args->packingCacheGB = std::stod(argument);
        else {
            std::cerr << "Unrecognized option " << option << std::endl << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error while parsing option `" << option << " " << argument << "`: " << e.what() << std::endl << std::endl;
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

    for (int i = 1; i < argc; ++i) {
        std::string argi(argv[i]);
        if (argi[0] == '-') {
            std::string option = argi;
            i++;
            if (i >= argc) {
                std::cerr << "Missing argument for option " << option << std::endl << std::endl;
                PrintArgsUsage(argv[0]);
                std::exit(-1);
            }
            if (!ParseOption(option, std::string(argv[i]), &args)) {
                PrintArgsUsage(argv[0]);
                std::exit(-1);
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
