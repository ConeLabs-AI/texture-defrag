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

#include <vcg/complex/algorithms/outline_support.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <set>
#include <chrono>
#include <vcg/space/rasterized_outline2_packer.h>
#include <wrap/qt/outline2_rasterizer.h>
// #include <wrap/qt/Outline2ToQImage.h>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

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

void SetRasterizerCacheMaxBytes(std::size_t bytes)
{
    QtOutline2Rasterizer::setCacheMaxBytes(bytes);
}


int Pack(const std::vector<ChartHandle>& charts, TextureObjectHandle textureObject, std::vector<TextureSize>& texszVec, const struct AlgoParameters& params, const std::map<ChartHandle, float>& chartMultipliers)
{
    using Packer = RasterizedOutline2Packer<float, QtOutline2Rasterizer>;
    auto rpack_params = Packer::Parameters();
    
    // Reset rasterizer cache stats for this packing run
    QtOutline2Rasterizer::statsSnapshot(true);
    
    // Pack the atlas

    texszVec.clear();

    std::vector<Outline2f> outlines;
    outlines.reserve(charts.size());
    std::vector<float> chartScaleMul;
    chartScaleMul.reserve(charts.size());
    std::vector<double> chartAreasOriginal;
    chartAreasOriginal.reserve(charts.size());

    for (auto c : charts) {
        // Determine per-chart scale from the pre-computed multipliers
        float mul = 1.4142f; // Default to sqrt(2) if not found
        auto it = chartMultipliers.find(c);
        if (it != chartMultipliers.end()) {
            mul = it->second;
        }
        chartScaleMul.push_back(mul);
        // Save the outline of the parameterization for this portion of the mesh and apply per-chart scaling
        Outline2f outline = ExtractOutline2f(*c);
        // Track original area before scaling
        double originalAreaAbs = std::abs(vcg::tri::OutlineUtil<float>::Outline2Area(outline));
        chartAreasOriginal.push_back(originalAreaAbs);
        for (auto &p : outline) {
            p.X() *= mul;
            p.Y() *= mul;
        }
        outlines.push_back(outline);
    }

    // Precompute per-chart absolute UV area (after per-chart scaling)
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

    // compute the scale factor for the packing
    int packingArea = 0;
    int textureArea = 0;
    for (unsigned i = 0; i < containerVec.size(); ++i) {
        packingArea += containerVec[i].X() * containerVec[i].Y();
        textureArea += textureObject->TextureWidth(i) * textureObject->TextureHeight(i);
    }
    // Adjust target area: preserve 1:1 charts, give sqrt(2) more area to resampled charts
    double targetUVArea = 0.0;
    for (size_t i = 0; i < chartAreasOriginal.size(); ++i) {
        // 1:1 charts keep original area; resampled charts get mul^2 (i.e., 2.0) area
        double mul = chartScaleMul[i];
        targetUVArea += chartAreasOriginal[i] * double(mul) * double(mul);
    }
    // If we do not have outline areas, fall back to original texture area target
    double packingScale = 1.0;
    if (targetUVArea > 0.0) {
        // The virtual grid area is packingArea; scale outlines by packingScale so that placed area ~= packingArea
        // outlines are already multiplied per-chart; Rasterizer also takes packingScale. We set packingScale so that
        // targetUVArea * packingScale^2 ~= packingArea => packingScale = sqrt(packingArea / targetUVArea)
        packingScale = std::sqrt((double)packingArea / targetUVArea);
    } else {
        packingScale = (textureArea > 0) ? std::sqrt(packingArea / (double)textureArea) : 1.0;
    }

    if (!std::isfinite(packingScale) || packingScale <= 0) {
        LOG_WARN << "[DIAG] Invalid packingScale computed: " << packingScale
                 << ". Resetting to 1.0. (packingArea=" << packingArea << ", textureArea=" << textureArea << ")";
        packingScale = 1.0;
    }

    // Guard: if packingScale is extremely small, output sizes will explode. Enforce a minimum.
    // With grid size up to 16384 and QImage limit of 32767, we need textureScale <= 2, so packingScale >= 0.5
    const double MIN_PACKING_SCALE = 0.5;
    if (packingScale < MIN_PACKING_SCALE) {
        LOG_ERR << "[VALIDATION] packingScale=" << packingScale << " is below minimum " << MIN_PACKING_SCALE
                << ". This indicates UV areas are far too large relative to packing grid."
                << " (targetUVArea=" << targetUVArea << ", packingArea=" << packingArea << "). Aborting.";
        LOG_ERR << "[VALIDATION] This usually means input UVs are not normalized to [0,1] or optimization exploded UV coordinates.";
        std::exit(-1);
    }

    LOG_INFO << "[DIAG] Packing scale factor: " << packingScale
             << " (packingArea=" << packingArea << ", targetUVArea=" << targetUVArea << ", textureArea=" << textureArea << ")";

    // Hierarchical Packing Step: Meso-pack micro charts into macro-tiles (bins)
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

        int microChartsOverMaxDim = 0;
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
                microChartsOverMaxDim++;
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
        if (microChartsOverMaxDim > 0) {
            LOG_WARN << "[DIAG] " << microChartsOverMaxDim << " micro charts were larger than MESO_BIN_MAX_DIM and may overflow their bins.";
        }

        // Finalize bins: compute their rectangular outlines and reset coordinates to (0,0)
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

        double totalBinArea = 0;
        for (const auto& bin : mesoBins) {
            totalBinArea += std::abs(vcg::tri::OutlineUtil<float>::Outline2Area(bin.outline));
        }
        double totalMicroArea = 0;
        for (unsigned idx : microIndices) totalMicroArea += chartAreas[idx];

        LOG_INFO << "[MESO-STATS] Micro charts: " << microIndices.size() 
                 << " packed into " << mesoBins.size() << " bins (" << mesoTimer.TimeElapsed() << "s)";
        LOG_INFO << "[MESO-STATS] Bins efficiency: " << (totalBinArea > 0 ? (totalMicroArea / totalBinArea) : 0) 
                 << " (MicroUV: " << totalMicroArea << " / BinUV: " << totalBinArea << ")";
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
    //rpack_params.permutations = false;
    rpack_params.permutations = (charts.size() < 50);
    rpack_params.rotationNum = params.rotationNum;
    rpack_params.gutterWidth = 4;
    rpack_params.minmax = false; // not used

    int totPackedItems = 0;

    std::vector<int> packableContainerIndices(packables.size(), -1); // -1 means not packed
    std::vector<vcg::Similarity2f> packableTransforms(packables.size(), vcg::Similarity2f{});

    auto selectSubset = [&](const std::vector<unsigned>& eligible,
                            double targetUVArea) -> std::vector<unsigned> {
        if (eligible.empty()) return {};

        std::vector<unsigned> indices = eligible;
        static thread_local std::mt19937 rng(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<unsigned> selected;
        selected.reserve(indices.size());
        double accum = 0.0;
        for (unsigned idx : indices) {
            if (accum >= targetUVArea) break;
            selected.push_back(idx);
            accum += packableAreas[idx];
        }
        if (selected.empty() && !indices.empty()) {
            selected.push_back(indices[0]);
        }

        std::sort(selected.begin(), selected.end(), [&](unsigned a, unsigned b){
            return packableAreas[a] > packableAreas[b];
        });
        return selected;
    };

    unsigned nc = 0; // current container index
    while (totPackedItems < (int) packables.size()) {
        if (nc >= containerVec.size())
            containerVec.push_back(vcg::Point2i(packingSize, packingSize));

        std::vector<unsigned> pending;
        for (unsigned i = 0; i < packableContainerIndices.size(); ++i) {
            if (packableContainerIndices[i] == -1) {
                pending.push_back(i);
            }
        }

        if (pending.empty())
            break;

        const float QIMAGE_MAX_DIM = 32766.0f;
        std::vector<unsigned> eligible;
        eligible.reserve(pending.size());

        int skippedEmpty = 0;
        int skippedInvalid = 0;
        int skippedTooLarge = 0;
        for(unsigned origIdx : pending) {
            if (packableOutlines[origIdx].empty()) {
                skippedEmpty++;
                packableContainerIndices[origIdx] = -2; // Mark as skipped
                totPackedItems++;
                continue;
            }

            vcg::Box2f bbox;
            for(const auto& p : packableOutlines[origIdx]) bbox.Add(p);

            if (!std::isfinite(bbox.DimX()) || !std::isfinite(bbox.DimY()) || bbox.DimX() < 0 || bbox.DimY() < 0) {
                skippedInvalid++;
                packableContainerIndices[origIdx] = -4; 
                totPackedItems++;
                continue;
            }

            float w = bbox.DimX() * (float)packingScale;
            float h = bbox.DimY() * (float)packingScale;
            float diagonal = std::sqrt(w * w + h * h);

            if (diagonal > QIMAGE_MAX_DIM) {
                skippedTooLarge++;
                packableContainerIndices[origIdx] = -3;
                totPackedItems++;
                continue;
            }
            eligible.push_back(origIdx);
        }
        if (skippedEmpty > 0) LOG_WARN << "[DIAG] Skipped " << skippedEmpty << " items with empty outlines.";
        if (skippedInvalid > 0) LOG_WARN << "[DIAG] Skipped " << skippedInvalid << " items due to invalid/non-finite UV bounding box.";
        if (skippedTooLarge > 0) LOG_WARN << "[DIAG] Skipped " << skippedTooLarge << " items because their scaled diagonal exceeds QImage limits.";

        if (eligible.empty()) {
            if (totPackedItems < (int) packables.size()) continue;
            else break;
        }

        double gridArea = double(containerVec[nc].X()) * double(containerVec[nc].Y());
        double targetUVArea = 5 * (gridArea / (packingScale * packingScale));

        std::vector<unsigned> selectedIdx = selectSubset(eligible, targetUVArea);
        std::vector<Outline2f> outlines_iter;
        outlines_iter.reserve(selectedIdx.size());
        for (unsigned idx : selectedIdx) outlines_iter.push_back(packableOutlines[idx]);

        LOG_VERBOSE << "[DIAG] Subset selection: pending=" << pending.size()
                 << ", eligible=" << eligible.size()
                 << ", selected=" << outlines_iter.size()
                 << ", targetUVArea≈" << targetUVArea
                 << ", grid=" << containerVec[nc].X() << "x" << containerVec[nc].Y();

        if (outlines_iter.empty())
            continue;

        const int MAX_SIZE = 20000;
        std::vector<vcg::Similarity2f> transforms;
        std::vector<int> polyToContainer;
        int n = 0;
        int packAttempts = 0;
        const int MAX_PACK_ATTEMPTS = 50;
        do {
            if (++packAttempts > MAX_PACK_ATTEMPTS) {
                LOG_ERR << "[DIAG] FATAL: Packing loop exceeded " << MAX_PACK_ATTEMPTS << " attempts. Aborting.";
                std::exit(-1);
            }
            transforms.clear();
            polyToContainer.clear();
            LOG_VERBOSE << "Packing " << outlines_iter.size() << " items into grid of size " << containerVec[nc].X() << " " << containerVec[nc].Y() << " (Attempt " << packAttempts << ")";
            n = RasterizationBasedPacker::PackBestEffortAtScale(outlines_iter, {containerVec[nc]}, transforms, polyToContainer, rpack_params, packingScale);
            LOG_VERBOSE << "[DIAG] Packing attempt finished. Items packed: " << n << ".";
            if (n == 0 && !outlines_iter.empty()) {
                containerVec[nc].X() *= 1.1;
                containerVec[nc].Y() *= 1.1;
            }
        } while (n == 0 && !outlines_iter.empty() && containerVec[nc].X() <= MAX_SIZE && containerVec[nc].Y() <= MAX_SIZE);

        if (n > 0) totPackedItems += n;

        if (n == 0) // no items were packed, stop
            break;
        else {
            double textureScale = 1.0 / packingScale;
            double w_d = static_cast<double>(containerVec[nc].X()) * textureScale;
            double h_d = static_cast<double>(containerVec[nc].Y()) * textureScale;
            if (!std::isfinite(w_d) || !std::isfinite(h_d) || w_d <= 0.0 || h_d <= 0.0) {
                LOG_ERR << "[VALIDATION] Invalid scaled container size (w=" << w_d << ", h=" << h_d << "). Aborting.";
                std::exit(-1);
            }
            const int MAX_QIMAGE_SIZE = 32767;
            if (w_d > MAX_QIMAGE_SIZE || h_d > MAX_QIMAGE_SIZE) {
                LOG_ERR << "[VALIDATION] Computed output container exceeds QImage limit: " << w_d << "x" << h_d << ". Aborting.";
                std::exit(-1);
            }
            TextureSize outTsz;
            outTsz.w = static_cast<int>(std::ceil(w_d));
            outTsz.h = static_cast<int>(std::ceil(h_d));
            texszVec.push_back(outTsz);
            for (unsigned i = 0; i < outlines_iter.size(); ++i) {
                if (polyToContainer[i] != -1) {
                    ensure(polyToContainer[i] == 0);
                    int packableInd = selectedIdx[i];
                    ensure(packableContainerIndices[packableInd] == -1);
                    packableContainerIndices[packableInd] = nc;
                    packableTransforms[packableInd] = transforms[i];
                }
            }
        }
        nc++;
    }

    // After main packing loop, finalize the container indices and transforms for all charts
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
    
    int totPackedCharts = (int)charts.size() - (int)remainingUnpacked.size();
    if (!remainingUnpacked.empty()) {
        LOG_INFO << "[DIAG] Creating individual containers for " << remainingUnpacked.size() << " unpacked charts.";
        
        for (unsigned ci : remainingUnpacked) {
            float mul = chartScaleMul[ci];
            ChartHandle chartptr = charts[ci];
            auto outline = ExtractOutline2d(*chartptr);
            
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
                    LOG_ERR << "[VALIDATION] Invalid individual container size for chart " << ci << ". Skipping.";
                    continue;
                }
                
                tsz.w = (int)std::max(1.0f, std::ceil(w_f));
                tsz.h = (int)std::max(1.0f, std::ceil(h_f));
                texszVec.push_back(tsz);
                containerVec.push_back(containerSize);
                nc++;
                totPackedCharts++;
             }
        }
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
                Point2i gridSize = containerVec[ic];
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
        double hitRate = (lookups > 0) ? (double(s.hits) / double(lookups)) : 0.0;
        LOG_INFO << "[PACK-CACHE] lookups=" << lookups
                 << " hits=" << s.hits
                 << " misses=" << s.misses
                 << " hitRate=" << hitRate
                 << " inserts=" << s.inserts
                 << " evictions=" << s.evictions
                 << " bytes=" << s.bytesCurrent << "/" << s.bytesMax;
    }

    return totPackedCharts;
}

Outline2f ExtractOutline2f(FaceGroup& chart)
{
    Outline2d outline2d = ExtractOutline2d(chart);
    Outline2f outline2f;
    for (auto& p : outline2d) {
        outline2f.push_back(vcg::Point2f(p.X(), p.Y()));
    }
    return outline2f;
}

Outline2d ExtractOutline2d(FaceGroup& chart)
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
                face::Pos<Mesh::FaceType> startPos = p;
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
                LOG_WARN << "[DIAG] Outline extraction: encountered non-finite UVs for chart " << chart.id << ". Falling back to UV bounding box.";
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
                LOG_WARN << "[DIAG] IntegerShift: degenerate anchor edge or invalid residuals for chart. Falling back to rotation 0.";
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
                LOG_WARN << "[DIAG] IntegerShift: unexpected rotation index " << minResidualIndex
                         << " (residual=" << minResidual << ") - falling back to 0.";
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
}
