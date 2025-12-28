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

#include "packing.h"
#include "texture_object.h"
#include "mesh_graph.h"
#include "logging.h"
#include "utils.h"
#include "mesh_attribute.h"
#include "timer.h"

#include <vcg/space/rasterized_outline2_packer.h>
#include <wrap/qt/outline2_rasterizer.h>
#include <vcg/complex/algorithms/outline_support.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <set>
#include <chrono>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <queue>

namespace vcg {
/**
 * Composition of two Similarity2 transformations: A * B
 * Resulting transformation R(p) = A(B(p))
 * Similarity2 transformation: T(S(R(p))) = tra + sca * (Rotate(p, rotRad))
 * 
 * A(p) = a.tra + a.sca * Rotate(p, a.rotRad)
 * B(p) = b.tra + b.sca * Rotate(p, b.rotRad)
 * A(B(p)) = a.tra + a.sca * Rotate(b.tra + b.sca * Rotate(p, b.rotRad), a.rotRad)
 *         = a.tra + a.sca * Rotate(b.tra, a.rotRad) + a.sca * b.sca * Rotate(Rotate(p, b.rotRad), a.rotRad)
 *         = (a.tra + a.sca * Rotate(b.tra, a.rotRad)) + (a.sca * b.sca) * Rotate(p, a.rotRad + b.rotRad)
 */
template <class SCALAR_TYPE>
Similarity2<SCALAR_TYPE> operator*(const Similarity2<SCALAR_TYPE> &a, const Similarity2<SCALAR_TYPE> &b) {
    Similarity2<SCALAR_TYPE> res;
    res.rotRad = a.rotRad + b.rotRad;
    res.sca = a.sca * b.sca;
    Point2<SCALAR_TYPE> rotatedTra = b.tra;
    rotatedTra.Rotate(a.rotRad);
    res.tra = a.tra + rotatedTra * a.sca;
    return res;
}
}

typedef vcg::RasterizedOutline2Packer<float, QtOutline2Rasterizer> RasterizationBasedPacker;

struct PackingStats {
    // Timings
    double mesoTime = 0;
    double parallelTime = 0;
    double individualTime = 0;
    double totalTime = 0;

    // Chart counts
    size_t totalCharts = 0;
    size_t macroCharts = 0;
    size_t microCharts = 0;
    size_t mesoBins = 0;

    // Area/Efficiency
    double totalMicroArea = 0;
    double totalMesoBinArea = 0;
    double totalPackableArea = 0;

    // Skip counts
    int skippedEmpty = 0;
    int skippedInvalid = 0;
    int skippedTooLarge = 0;

    // Sheet counts
    int estimatedSheets = 0;
    int actualSheets = 0;
    int parallelBins = 0;
    int individualSheets = 0;

    // Warnings
    int microChartsOverMaxDim = 0;
    int outlineExtractionWarnings = 0;
    int infiniteGrowthWarnings = 0;
    int individualPackFailures = 0;

    // Cache
    long long cacheLookups = 0;
    long long cacheHits = 0;
    long long cacheMisses = 0;
    double cacheHitRate = 0;
};

void SetRasterizerCacheMaxBytes(std::size_t bytes)
{
    QtOutline2Rasterizer::setCacheMaxBytes(bytes);
}


int Pack(const std::vector<ChartHandle>& charts, TextureObjectHandle textureObject, std::vector<TextureSize>& texszVec, const struct AlgoParameters& params, const std::map<ChartHandle, float>& chartMultipliers)
{
    Timer totalTimer;
    PackingStats stats;
    stats.totalCharts = charts.size();

    using Packer = RasterizedOutline2Packer<float, QtOutline2Rasterizer>;
    auto rpack_params = Packer::Parameters();
    
    QtOutline2Rasterizer::statsSnapshot(true);

    texszVec.clear();

    std::vector<Outline2f> outlines;
    outlines.reserve(charts.size());
    std::vector<float> chartScaleMul;
    chartScaleMul.reserve(charts.size());
    std::vector<double> chartAreasOriginal;
    chartAreasOriginal.reserve(charts.size());

    for (auto c : charts) {
        float mul = 1.4142f; // Default to sqrt(2) if not found
        auto it = chartMultipliers.find(c);
        if (it != chartMultipliers.end()) {
            mul = it->second;
        }
        chartScaleMul.push_back(mul);
        Outline2f outline = ExtractOutline2f(*c, &stats.outlineExtractionWarnings);
        double originalAreaAbs = std::abs(vcg::tri::OutlineUtil<float>::Outline2Area(outline));
        chartAreasOriginal.push_back(originalAreaAbs);
        for (auto &p : outline) {
            p.X() *= mul;
            p.Y() *= mul;
        }
        outlines.push_back(outline);
    }

    std::vector<double> chartAreas(outlines.size(), 0.0);
    for (size_t i = 0; i < outlines.size(); ++i) {
        chartAreas[i] = std::abs(vcg::tri::OutlineUtil<float>::Outline2Area(outlines[i]));
    }

    int packingSize = 16384;
    std::vector<std::pair<double,double>> trs = textureObject->ComputeRelativeSizes();

    std::vector<Point2i> containerVec;
    for (auto rs : trs) {
        vcg::Point2i container(packingSize * rs.first, packingSize * rs.second);
        containerVec.push_back(container);
    }

    int64_t packingArea = 0;
    int64_t textureArea = 0;
    for (unsigned i = 0; i < containerVec.size(); ++i) {
        packingArea += (int64_t)containerVec[i].X() * containerVec[i].Y();
        textureArea += (int64_t)textureObject->TextureWidth(i) * textureObject->TextureHeight(i);
    }
    double targetUVArea = 0.0;
    for (size_t i = 0; i < chartAreasOriginal.size(); ++i) {
        // 1:1 charts keep original area; resampled charts get mul^2 (i.e., 2.0) area
        double mul = chartScaleMul[i];
        targetUVArea += chartAreasOriginal[i] * double(mul) * double(mul);
    }
    double packingScale = 1.0;

    Timer mesoTimer;
    const int MICRO_CHART_FACE_THRESHOLD = 5;
    const int MESO_BIN_MAX_DIM = 2048; // in packing grid units

    std::vector<unsigned> macroIndices;
    std::vector<unsigned> microIndices;

    for (unsigned i = 0; i < (unsigned)charts.size(); ++i) {
        if (charts[i]->FN() < (size_t)MICRO_CHART_FACE_THRESHOLD) {
            microIndices.push_back(i);
        } else {
            macroIndices.push_back(i);
        }
    }
    stats.microCharts = microIndices.size();
    stats.macroCharts = macroIndices.size();

    struct MesoBin {
        std::vector<unsigned> chartIndices;
        std::vector<vcg::Similarity2f> relativeTransforms;
        Outline2f outline;
        double areaCharts = 0.0;
    };
    std::vector<MesoBin> mesoBins;

    if (!microIndices.empty()) {
        // Simple shelf packer for micro charts
        // Sort micro charts by height (bbox DimY) for shelf packing
        std::sort(microIndices.begin(), microIndices.end(), [&](unsigned a, unsigned b) {
            vcg::Box2f bbA, bbB;
            for (const auto& p : outlines[a]) bbA.Add(p);
            for (const auto& p : outlines[b]) bbB.Add(p);
            return bbA.DimY() > bbB.DimY();
        });

        auto createNewBin = [&]() {
            mesoBins.emplace_back();
            return &mesoBins.back();
        };

        MesoBin* currentBin = createNewBin();
        float currentX = 0, currentY = 0, shelfHeight = 0;
        float padding = 4.0f / (float)packingScale; // 4 pixels in UV space

        for (unsigned idx : microIndices) {
            vcg::Box2f bb;
            for (const auto& p : outlines[idx]) bb.Add(p);
            float w = bb.DimX() + padding;
            float h = bb.DimY() + padding;

            // Enforce integer pixel placement inside the bin to avoid sub-pixel bleeding
            auto snapToPixel = [&](float val) {
                return std::ceil(val * (float)packingScale) / (float)packingScale;
            };

            if (snapToPixel(currentX + w) * packingScale > MESO_BIN_MAX_DIM) {
                currentX = 0;
                currentY = snapToPixel(currentY + shelfHeight);
                shelfHeight = 0;
            }

            if (snapToPixel(currentY + h) * packingScale > MESO_BIN_MAX_DIM) {
                currentBin = createNewBin();
                currentX = 0;
                currentY = 0;
                shelfHeight = 0;
            }

            if (w * packingScale > MESO_BIN_MAX_DIM || h * packingScale > MESO_BIN_MAX_DIM) {
                stats.microChartsOverMaxDim++;
            }

            vcg::Similarity2f relTr;
            relTr.tra = vcg::Point2f(currentX - bb.min.X(), currentY - bb.min.Y());
            relTr.sca = 1.0f;
            
            currentBin->chartIndices.push_back(idx);
            currentBin->relativeTransforms.push_back(relTr);
            currentBin->areaCharts += chartAreas[idx];

            currentX = snapToPixel(currentX + w);
            shelfHeight = std::max(shelfHeight, h);
        }

        for (auto& bin : mesoBins) {
            vcg::Box2f binBB;
            for (size_t i = 0; i < bin.chartIndices.size(); ++i) {
                unsigned idx = bin.chartIndices[i];
                for (const auto& p : outlines[idx]) {
                    binBB.Add(bin.relativeTransforms[i] * p);
                }
            }
            // Reset transforms so bin starts at (0,0)
            vcg::Point2f offset = -binBB.min;
            for (auto& tr : bin.relativeTransforms) {
                tr.tra += offset;
            }
            binBB.Offset(offset);

            bin.outline.push_back(vcg::Point2f(0, 0));
            bin.outline.push_back(vcg::Point2f(binBB.max.X(), 0));
            bin.outline.push_back(vcg::Point2f(binBB.max.X(), binBB.max.Y()));
            bin.outline.push_back(vcg::Point2f(0, binBB.max.Y()));
        }

        stats.mesoTime = mesoTimer.TimeElapsed();
        stats.mesoBins = mesoBins.size();
        for (const auto& bin : mesoBins) {
            stats.totalMesoBinArea += std::abs(vcg::tri::OutlineUtil<float>::Outline2Area(bin.outline));
        }
        for (unsigned idx : microIndices) stats.totalMicroArea += chartAreas[idx];
    }

    struct Packable {
        Outline2f outline;
        int originalChartIdx = -1; // If >= 0, it's a macro chart
        int mesoBinIdx = -1;       // If >= 0, it's a meso bin
    };
    std::vector<Packable> packables;
    packables.reserve(macroIndices.size() + mesoBins.size());
    for (unsigned idx : macroIndices) {
        Packable p;
        p.outline = outlines[idx];
        p.originalChartIdx = (int)idx;
        packables.push_back(p);
    }
    for (int i = 0; i < (int)mesoBins.size(); ++i) {
        Packable p;
        p.outline = mesoBins[i].outline;
        p.mesoBinIdx = i;
        packables.push_back(p);
    }

    std::vector<Outline2f> packableOutlines;
    packableOutlines.reserve(packables.size());
    std::vector<double> packableAreas(packables.size());
    for (size_t i = 0; i < packables.size(); ++i) {
        packableOutlines.push_back(packables[i].outline);
        packableAreas[i] = std::abs(vcg::tri::OutlineUtil<float>::Outline2Area(packables[i].outline));
    }

    rpack_params.costFunction = Packer::Parameters::LowestHorizon;
    rpack_params.doubleHorizon = false;
    rpack_params.innerHorizon = false;
    rpack_params.permutations = (charts.size() < 50);
    rpack_params.rotationNum = params.rotationNum;
    rpack_params.gutterWidth = 4;
    rpack_params.minmax = false;

    // Partitioned Parallel Packing Step
    struct ParallelItem {
        int packableIndex; // Index into the 'packables' vector
        double area;
    };
    std::vector<ParallelItem> items;
    const float QIMAGE_MAX_DIM = 32766.0f;

    int skippedEmpty = 0;
    int skippedInvalid = 0;
    int skippedTooLarge = 0;

    for (size_t i = 0; i < packables.size(); ++i) {
        if (packables[i].outline.empty()) {
            stats.skippedEmpty++;
            continue;
        }

        vcg::Box2f bbox;
        for (const auto& p : packables[i].outline) bbox.Add(p);

        if (!std::isfinite(bbox.DimX()) || !std::isfinite(bbox.DimY()) || bbox.DimX() < 0 || bbox.DimY() < 0) {
            stats.skippedInvalid++;
            continue;
        }

        float w = bbox.DimX() * (float)packingScale;
        float h = bbox.DimY() * (float)packingScale;
        float diagonal = std::sqrt(w * w + h * h);

        if (diagonal > QIMAGE_MAX_DIM) {
            stats.skippedTooLarge++;
            continue;
        }

        double a = std::abs(vcg::tri::OutlineUtil<float>::Outline2Area(packables[i].outline));
        items.push_back({(int)i, a});
    }

    // 2. Sort Descending by Area (LPT Heuristic)
    std::sort(items.begin(), items.end(), [](const ParallelItem& a, const ParallelItem& b) {
        return a.area > b.area;
    });

    // 3. Estimate Sheets & Distribute
    double totalPackableArea = 0;
    for (const auto& item : items) totalPackableArea += item.area;
    stats.totalPackableArea = totalPackableArea;

    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif
    
    double gridArea = (double)packingSize * (double)packingSize;
    // Assume 85% packing efficiency to estimate number of sheets
    int estimatedSheets = (int)std::ceil(totalPackableArea / (gridArea * 0.85));
    // Ensure we have at least as many sheets as threads to maximize concurrency
    int numSheets = std::max(numThreads, estimatedSheets);
    stats.estimatedSheets = estimatedSheets;
    stats.parallelBins = numSheets;

    struct SheetBin {
        std::vector<int> packableIndices;
        double fill = 0;
    };
    std::vector<SheetBin> sheets(numSheets);
    
    // Priority queue to distribute items to the emptiest bin
    using BinPair = std::pair<double, int>;
    std::priority_queue<BinPair, std::vector<BinPair>, std::greater<BinPair>> pq;
    for (int i = 0; i < numSheets; ++i) pq.push({0.0, i});

    for (const auto& item : items) {
        auto best = pq.top();
        pq.pop();
        sheets[best.second].packableIndices.push_back(item.packableIndex);
        sheets[best.second].fill += item.area;
        pq.push({sheets[best.second].fill, best.second});
    }

    // 4. Parallel Pack
    struct ThreadResult {
        std::vector<TextureSize> sizes;
        std::vector<vcg::Similarity2f> transforms;
        std::vector<int> containerIndices;
        std::vector<int> originalPackableIndices; // Track which packable index this result belongs to
    };
    std::vector<ThreadResult> results(numSheets);

    Timer parallelTimer;
    int sheetsProcessed = 0;
    int chartsProcessedInBins = 0;
    int lastReportedProgress = -1;
    int totalPackablesCount = (int)packables.size();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numSheets; ++i) {
        if (sheets[i].packableIndices.empty()) {
            #pragma omp atomic
            sheetsProcessed++;
            continue;
        }

        // 1. Setup Input
        std::vector<Outline2f> subsetOutlines;
        std::vector<int> subsetIndices;
        subsetOutlines.reserve(sheets[i].packableIndices.size());
        subsetIndices.reserve(sheets[i].packableIndices.size());
        for (int pIdx : sheets[i].packableIndices) {
            subsetOutlines.push_back(packables[pIdx].outline);
            subsetIndices.push_back(pIdx);
        }

        // 2. Setup Containers - Start with ONE
        std::vector<vcg::Point2i> localCont = { vcg::Point2i(packingSize, packingSize) };
        std::vector<vcg::Similarity2f> localTr;
        std::vector<int> localPolyToCont;

        // 3. Dynamic Packing Loop
        bool fullyPacked = false;
        int safetyCounter = 0;

        while (!fullyPacked) {
            localTr.clear();
            localPolyToCont.clear();

            // Call Packer with FIXED scale 1.0
            // We set bypassCache to false now because we will make it thread-local later
            int nPacked = RasterizationBasedPacker::PackBestEffortAtScale(
                subsetOutlines, 
                localCont, 
                localTr, 
                localPolyToCont, 
                rpack_params, 
                1.0f, 
                false
            );

            if (nPacked == (int)subsetOutlines.size()) {
                fullyPacked = true;
            } else {
                // overflow detected: add another texture page
                localCont.push_back(vcg::Point2i(packingSize, packingSize));
                
                // Safety break for giant/degenerate charts
                if (++safetyCounter > 50) {
                     #pragma omp critical
                     stats.infiniteGrowthWarnings++;

                     // Fallback: Try one last time with downscaling just for this bucket
                     RasterizationBasedPacker::PackBestEffortAtScale(
                        subsetOutlines, localCont, localTr, localPolyToCont, rpack_params, 0.5f, false);
                     fullyPacked = true; 
                }
            }
        }

        // 4. Save Results
        int maxUsedCont = -1;
        for (int contIdx : localPolyToCont) {
            if (contIdx > maxUsedCont) maxUsedCont = contIdx;
        }

        for (int k = 0; k <= maxUsedCont; ++k) {
            TextureSize tsz;
            tsz.w = packingSize;
            tsz.h = packingSize;
            results[i].sizes.push_back(tsz);
        }

        for (size_t k = 0; k < localPolyToCont.size(); ++k) {
            int contIdx = localPolyToCont[k];
            if (contIdx != -1) {
                results[i].transforms.push_back(localTr[k]);
                results[i].containerIndices.push_back(contIdx);
                results[i].originalPackableIndices.push_back(subsetIndices[k]);
            }
        }

        #pragma omp critical
        {
            sheetsProcessed++;
            chartsProcessedInBins += subsetIndices.size();
            int progress = (chartsProcessedInBins * 100) / totalPackablesCount;
            if (progress / 10 > lastReportedProgress / 10) {
                LOG_INFO << "[PACKING] Progress: " << progress << "% (" << chartsProcessedInBins << "/" << totalPackablesCount << " items)";
                lastReportedProgress = progress;
            }
        }
    }
    stats.parallelTime = parallelTimer.TimeElapsed();

    // 5. Global Reconstruction (Serialized)
    std::vector<int> packableContainerIndices(packables.size(), -1);
    std::vector<vcg::Similarity2f> packableTransforms(packables.size(), vcg::Similarity2f{});
    std::vector<Point2i> atlasContainerSizes;
    
    int globalTexOffset = 0;
    for (int i = 0; i < numSheets; ++i) {
        // Append sizes
        texszVec.insert(texszVec.end(), results[i].sizes.begin(), results[i].sizes.end());
        for (size_t k = 0; k < results[i].sizes.size(); ++k) {
            atlasContainerSizes.push_back(Point2i(packingSize, packingSize));
        }

        // Map transforms back using the stored original indices
        for (size_t k = 0; k < results[i].transforms.size(); ++k) {
            int pIdx = results[i].originalPackableIndices[k];
            int localC = results[i].containerIndices[k];
            
            if (pIdx >= 0 && pIdx < (int)packableContainerIndices.size()) {
                packableContainerIndices[pIdx] = globalTexOffset + localC;
                packableTransforms[pIdx] = results[i].transforms[k];
            }
        }
        globalTexOffset += results[i].sizes.size();
    }

    // After parallel packing, finalize the container indices and transforms for all charts
    std::vector<int> chartContainerIndices(charts.size(), -1);
    std::vector<vcg::Similarity2f> chartPackingTransforms(charts.size(), vcg::Similarity2f{});

    for (int i = 0; i < (int)packables.size(); ++i) {
        int cIdx = packables[i].originalChartIdx;
        int bIdx = packables[i].mesoBinIdx;
        int contIdx = packableContainerIndices[i];
        if (contIdx >= 0) {
            if (cIdx >= 0) {
                chartContainerIndices[cIdx] = contIdx;
                chartPackingTransforms[cIdx] = packableTransforms[i];
            } else if (bIdx >= 0) {
                const auto& bin = mesoBins[bIdx];
                for (size_t j = 0; j < bin.chartIndices.size(); ++j) {
                    unsigned chartIdx = bin.chartIndices[j];
                    chartContainerIndices[chartIdx] = contIdx;
                    chartPackingTransforms[chartIdx] = packableTransforms[i] * bin.relativeTransforms[j];
                }
            }
        }
    }

    int totPackedCharts = 0;
    for (int idx : chartContainerIndices) if (idx >= 0) totPackedCharts++;
    int nc = (int)atlasContainerSizes.size();

    // Helper function to round up to nearest power of two
    auto roundUpToPowerOfTwo = [](int v) -> int {
        if (v <= 0) return 1;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    };
    
    // Handle any remaining unpacked charts (both macro and micro)
    std::vector<unsigned> remainingUnpacked;
    for (unsigned ci = 0; ci < charts.size(); ++ci) {
        if (chartContainerIndices[ci] == -1) {
            remainingUnpacked.push_back(ci);
        }
    }
    
    if (!remainingUnpacked.empty()) {
        Timer individualTimer;
        for (unsigned ci : remainingUnpacked) {
            float mul = chartScaleMul[ci];
            ChartHandle chartptr = charts[ci];
            auto outline = ExtractOutline2d(*chartptr, &stats.outlineExtractionWarnings);
            
            vcg::Box2d bb;
            for (const auto& p : outline) bb.Add(p);
            
            int padding = 8;
            int requiredWidth = (int)std::ceil(bb.DimX() * mul) + padding;
            int requiredHeight = (int)std::ceil(bb.DimY() * mul) + padding;
            
            requiredWidth = roundUpToPowerOfTwo(requiredWidth);
            requiredHeight = roundUpToPowerOfTwo(requiredHeight);
            
            const int MAX_QIMAGE_SIZE = 32767;
            requiredWidth = std::min(requiredWidth, MAX_QIMAGE_SIZE);
            requiredHeight = std::min(requiredHeight, MAX_QIMAGE_SIZE);
            
            vcg::Point2i containerSize(requiredWidth, requiredHeight);
            
            std::vector<std::vector<vcg::Point2f>> singleOutlineVec;
            singleOutlineVec.emplace_back();
            singleOutlineVec.back().reserve(outline.size());
            for (const auto &p : outline) {
                singleOutlineVec.back().push_back(vcg::Point2f(static_cast<float>(p.X()*mul), static_cast<float>(p.Y()*mul)));
            }
 
            std::vector<vcg::Similarity2f> singleTrVec;
            std::vector<int> singlePolyToContainer;
 
             float scale = 1.0;
             bool packSuccess = Packer::PackAtFixedScale(singleOutlineVec, {containerSize}, singleTrVec, singlePolyToContainer, rpack_params, scale);
             while (!packSuccess && scale > 0.1) {
                 scale *= 0.9;
                 packSuccess = Packer::PackAtFixedScale(singleOutlineVec, {containerSize}, singleTrVec, singlePolyToContainer, rpack_params, scale);
             }
 
             if (packSuccess && !singlePolyToContainer.empty() && singlePolyToContainer[0] != -1) {
                 chartContainerIndices[ci] = nc;
                 chartPackingTransforms[ci] = singleTrVec[0];
 
                TextureSize tsz;
                float textureScaleLocal = 1.0f / scale;
                float w_f = requiredWidth * textureScaleLocal;
                float h_f = requiredHeight * textureScaleLocal;
                
                if (!std::isfinite(w_f) || !std::isfinite(h_f) || w_f <= 0 || h_f <= 0 || w_f > MAX_QIMAGE_SIZE || h_f > MAX_QIMAGE_SIZE) {
                    stats.individualPackFailures++;
                    continue;
                }
                
                tsz.w = (int)std::max(1.0f, std::ceil(w_f));
                tsz.h = (int)std::max(1.0f, std::ceil(h_f));
                texszVec.push_back(tsz);
                atlasContainerSizes.push_back(containerSize);
                nc++;
                totPackedCharts++;
                stats.individualSheets++;
             } else {
                stats.individualPackFailures++;
             }
        }
        stats.individualTime = individualTimer.TimeElapsed();
    }

    for (unsigned i = 0; i < charts.size(); ++i) {
        for (auto fptr : charts[i]->fpVec) {
            int ic = chartContainerIndices[i];
            if (ic < 0) {
                for (int j = 0; j < fptr->VN(); ++j) {
                    fptr->V(j)->T().P() = Point2d::Zero();
                    fptr->V(j)->T().N() = 0;
                    fptr->WT(j).P() = Point2d::Zero();
                    fptr->WT(j).N() = 0;
                }
            }
            else {
                Point2i gridSize = atlasContainerSizes[ic];
                for (int j = 0; j < fptr->VN(); ++j) {
                    Point2d uv = fptr->WT(j).P();
                    float mul = chartScaleMul[i];
                    Point2f p = chartPackingTransforms[i] * (Point2f(uv[0] * mul, uv[1] * mul));
                    p.X() /= (double) gridSize.X();
                    p.Y() /= (double) gridSize.Y();
                    fptr->V(j)->T().P() = Point2d(p.X(), p.Y());
                    fptr->V(j)->T().N() = ic;
                    fptr->WT(j).P() = fptr->V(j)->T().P();
                    fptr->WT(j).N() = fptr->V(j)->T().N();
                }
            }
        }
    }

    for (auto c : charts)
        c->ParameterizationChanged();

    // Log rasterizer cache stats for this packing pass
    {
        auto s = QtOutline2Rasterizer::statsSnapshot(false);
        long long lookups = static_cast<long long>(s.hits) + static_cast<long long>(s.misses);
        stats.cacheLookups = lookups;
        stats.cacheHits = s.hits;
        stats.cacheMisses = s.misses;
        stats.cacheHitRate = (lookups > 0) ? (double(s.hits) / double(lookups)) : 0.0;
    }

    stats.totalTime = totalTimer.TimeElapsed();
    stats.actualSheets = (int)texszVec.size();

    LOG_INFO << "------- Packing Phase Statistics -------";
    LOG_INFO << "Charts: Total=" << stats.totalCharts << ", Macro=" << stats.macroCharts << ", Micro=" << stats.microCharts;
    LOG_INFO << "Meso-packing: " << stats.mesoBins << " bins created (" << stats.mesoTime << "s)";
    if (stats.mesoBins > 0) {
        LOG_INFO << "Meso-efficiency: " << (stats.totalMesoBinArea > 0 ? (stats.totalMicroArea / stats.totalMesoBinArea) : 0) 
                 << " (MicroUV: " << stats.totalMicroArea << " / BinUV: " << stats.totalMesoBinArea << ")";
    }
    LOG_INFO << "Parallel Packing: " << stats.parallelBins << " bins, " << stats.actualSheets - stats.individualSheets << " sheets (" << stats.parallelTime << "s)";
    LOG_INFO << "Individual Packing: " << stats.individualSheets << " sheets (" << stats.individualTime << "s)";
    LOG_INFO << "Total Sheets: " << stats.actualSheets << " (Estimated: " << stats.estimatedSheets << ")";
    LOG_INFO << "Total Time: " << stats.totalTime << "s";
    
    if (stats.skippedEmpty > 0 || stats.skippedInvalid > 0 || stats.skippedTooLarge > 0 || 
        stats.microChartsOverMaxDim > 0 || stats.infiniteGrowthWarnings > 0 || stats.individualPackFailures > 0) {
        LOG_INFO << "------- Packing Warnings Summary -------";
        if (stats.skippedEmpty > 0) LOG_WARN << "Skipped " << stats.skippedEmpty << " items with empty outlines.";
        if (stats.skippedInvalid > 0) LOG_WARN << "Skipped " << stats.skippedInvalid << " items due to invalid/non-finite UV bounding box.";
        if (stats.skippedTooLarge > 0) LOG_WARN << "Skipped " << stats.skippedTooLarge << " items because their scaled diagonal exceeds QImage limits.";
        if (stats.microChartsOverMaxDim > 0) LOG_WARN << stats.microChartsOverMaxDim << " micro charts were larger than MESO_BIN_MAX_DIM.";
        if (stats.infiniteGrowthWarnings > 0) LOG_WARN << stats.infiniteGrowthWarnings << " parallel bins triggered infinite growth safety.";
        if (stats.individualPackFailures > 0) LOG_WARN << stats.individualPackFailures << " individual chart packing failures.";
    }

    LOG_INFO << "[PACK-CACHE] lookups=" << stats.cacheLookups
             << " hits=" << stats.cacheHits
             << " misses=" << stats.cacheMisses
             << " hitRate=" << stats.cacheHitRate;
    LOG_INFO << "----------------------------------------";

    return totPackedCharts;
}

Outline2f ExtractOutline2f(FaceGroup& chart, int* warningCounter)
{
    Outline2d outline2d = ExtractOutline2d(chart, warningCounter);
    Outline2f outline2f;
    for (auto& p : outline2d) {
        outline2f.push_back(vcg::Point2f(p.X(), p.Y()));
    }
    return outline2f;
}

Outline2d ExtractOutline2d(FaceGroup& chart, int* warningCounter)
{
    //ensure(chart.numMerges == 0);

    std::vector<Outline2d> outline2Vec;
    Outline2d outline;

    for (auto fptr : chart.fpVec)
        fptr->ClearV();

    for (auto fptr : chart.fpVec) {
        for (int i = 0; i < 3; ++i) {
            if (!fptr->IsV() && face::IsBorder(*fptr, i)) {
                face::Pos<Mesh::FaceType> p(fptr, i);
                face::Pos<Mesh::FaceType> startPos(p.F(), p.E(), p.V());
                ensure(p.IsBorder());
                do {
                    ensure(p.IsManifold());
                    p.F()->SetV();
                    vcg::Point2d uv = p.F()->WT(p.VInd()).P();
                    outline.push_back(uv);
                    p.NextB();
                }
                while (p != startPos);
                outline2Vec.push_back(outline);
                outline.clear();
            }
        }
    }

    int i;
    vcg::Box2d box = chart.UVBox();
    bool useChartBBAsOutline = false;

    std::size_t maxsz = 0;
    for (std::size_t i = 0; i < outline2Vec.size(); ++i) {
        maxsz = std::max(maxsz, outline2Vec[i].size());
    }

    if (maxsz == 0) {
        useChartBBAsOutline = true;
    } else {
        // Pick the largest finite, positive-area loop; ignore degenerate/non-finite ones
        int best = -1;
        double bestArea = 0.0;
        bool seenNonFinite = false;
        for (int k = 0; k < (int)outline2Vec.size(); ++k) {
            std::vector<vcg::Point2d> uniq;
            uniq.reserve(outline2Vec[k].size());
            bool localNonFinite = false;
            for (const auto &p : outline2Vec[k]) {
                if (!std::isfinite(p.X()) || !std::isfinite(p.Y())) { localNonFinite = true; break; }
                if (uniq.empty() || (p - uniq.back()).Norm() > 1e-15) uniq.push_back(p);
            }
            if (localNonFinite) { seenNonFinite = true; continue; }
            if (uniq.size() < 3) continue;
            double a = std::abs(vcg::tri::OutlineUtil<double>::Outline2Area(uniq));
            if (!std::isfinite(a) || a <= 0.0) continue;
            if (a > bestArea) { bestArea = a; best = k; }
        }

        if (best < 0) {
            if (seenNonFinite) {
                if (warningCounter) (*warningCounter)++;
            }
            useChartBBAsOutline = true;
        } else {
            i = best;
            if (tri::OutlineUtil<double>::Outline2Area(outline2Vec[i]) < 0)
                tri::OutlineUtil<double>::ReverseOutline2(outline2Vec[i]);
            vcg::Box2d outlineBox;
            for (const auto& p : outline2Vec[i])
                outlineBox.Add(p);
            if (outlineBox.DimX() < box.DimX() || outlineBox.DimY() < box.DimY())
                useChartBBAsOutline = true;
        }
    }

    if (useChartBBAsOutline) {
        outline.clear();
        outline.push_back(Point2d(box.min.X(), box.min.Y()));
        outline.push_back(Point2d(box.max.X(), box.min.Y()));
        outline.push_back(Point2d(box.max.X(), box.max.Y()));
        outline.push_back(Point2d(box.min.X(), box.max.Y()));
        return outline;
    } else {
        return outline2Vec[i];
    }

}

void IntegerShift(Mesh& m, const std::vector<ChartHandle>& chartsToPack, const std::vector<TextureSize>& texszVec, const std::map<ChartHandle, int>& anchorMap, const std::map<RegionID, bool>& flippedInput)
{
    // compute grid-preserving translation
    // for each chart
    //   - find an anchor vertex (i.e. find a vertex that belonged to the
    //     source chart that determined the integer translation of the final chart.
    //   - compute the displacement of this anchor vertex wrt the integer pixel coordinates
    //     both in its original configuration (t0) and in the final, packed chart (t1)
    //   - compute the translation vector t = t0 - t1
    //   - apply the translation t to the entire chart

    ensure(HasWedgeTexCoordStorageAttribute(m));
    auto wtcsh = GetWedgeTexCoordStorageAttribute(m);

    std::vector<double> angle = { 0, M_PI_2, M_PI, (M_PI_2 + M_PI) };

    auto Rotate = [] (vcg::Point2d p, double theta) -> vcg::Point2d { return p.Rotate(theta); };

    int degenerateAnchorEdges = 0;
    int unexpectedRotationIndices = 0;

    for (auto c : chartsToPack) {
        auto it = anchorMap.find(c);
        if (it != anchorMap.end()) {
            Mesh::FacePointer fptr = &(m.face[it->second]);
            bool flipped = flippedInput.at(fptr->initialId);

            vcg::Point2d d0 = wtcsh[fptr].tc[1].P() - wtcsh[fptr].tc[0].P();
            vcg::Point2d d1 = fptr->cWT(1).P() - fptr->cWT(0).P();

            if (flipped)
                d0.X() *= -1;

            double minResidual = 2 * M_PI;
            int minResidualIndex = -1;
            // Guard against degenerate/invalid vectors which can yield NaN residuals
            double len0 = std::hypot(d0.X(), d0.Y());
            double len1 = std::hypot(d1.X(), d1.Y());
            if (std::isfinite(len0) && std::isfinite(len1) && len0 > 1e-12 && len1 > 1e-12) {
                for (int i = 0; i < 4; ++i) {
                    double residual = VecAngle(Rotate(d0, angle[i]), d1);
                    if (std::isfinite(residual) && residual < minResidual) {
                        minResidual = residual;
                        minResidualIndex = i;
                    }
                }
            }
            if (minResidualIndex == -1) {
                degenerateAnchorEdges++;
                minResidualIndex = 0;
            }

            int ti = fptr->cWT(0).N();
            if (ti >= (int) texszVec.size()) {
                LOG_ERR << "[VALIDATION] IntegerShift: texture index " << ti
                        << " out of bounds for texszVec.size()=" << texszVec.size() << ". Aborting.";
                std::exit(-1);
            }
            vcg::Point2d textureSize(texszVec[ti].w, texszVec[ti].h);

            vcg::Point2d u0 = wtcsh[fptr].tc[0].P();
            vcg::Point2d u1 = fptr->cWT(0).P();

            double unused;
            double dx = std::modf(u0.X(), &unused);
            double dy = std::modf(u0.Y(), &unused);

            if (flipped)
                dx = 1 - dx;

            switch(minResidualIndex) {
            case 0:
                break;
            case 1:
                std::swap(dx, dy);
                dx = 1 - dx;
                break;
            case 2:
                dx = 1 - dx;
                dy = 1 - dy;
                break;
            case 3:
                std::swap(dx, dy);
                dy = 1 - dy;
                break;
            default:
                unexpectedRotationIndices++;
                // Fallback to no rotation adjustment
                minResidualIndex = 0;
                break;
            }

            double dx1 = std::modf(u1.X() * textureSize.X(), &unused);
            double dy1 = std::modf(u1.Y() * textureSize.Y(), &unused);
            vcg::Point2d t(0, 0);
            t.X() = (dx - dx1) / textureSize.X();
            t.Y() = (dy - dy1) / textureSize.Y();

            for (auto fptr : c->fpVec) {
                for (int i = 0; i < 3; ++i) {
                    fptr->WT(i).P() += t;
                    fptr->V(i)->T().P() = fptr->WT(i).P();
                }
            }
        }
    }

    if (degenerateAnchorEdges > 0 || unexpectedRotationIndices > 0) {
        LOG_INFO << "------- Integer Shift Statistics -------";
        if (degenerateAnchorEdges > 0) LOG_WARN << "Degenerate anchor edges: " << degenerateAnchorEdges;
        if (unexpectedRotationIndices > 0) LOG_WARN << "Unexpected rotation indices: " << unexpectedRotationIndices;
        LOG_INFO << "----------------------------------------";
    }
}
