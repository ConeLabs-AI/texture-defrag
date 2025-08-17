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

typedef vcg::RasterizedOutline2Packer<float, QtOutline2Rasterizer> RasterizationBasedPacker;

void SetRasterizerCacheMaxBytes(std::size_t bytes)
{
    QtOutline2Rasterizer::setCacheMaxBytes(bytes);
}


int Pack(const std::vector<ChartHandle>& charts, TextureObjectHandle textureObject, std::vector<TextureSize>& texszVec, const struct AlgoParameters& params, const std::map<ChartHandle, int>& anchorMap)
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
        // Determine per-chart scale: 1.0 for 1:1 copy (present in anchorMap), sqrt(2) for resampled charts
        float mul = (anchorMap.find(c) != anchorMap.end()) ? 1.0f : static_cast<float>(std::sqrt(2.0));
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

    LOG_INFO << "[DIAG] Packing scale factor: " << packingScale
             << " (packingArea=" << packingArea << ", textureArea=" << textureArea << ")";


    rpack_params.costFunction = Packer::Parameters::LowestHorizon;
    rpack_params.doubleHorizon = false;
    rpack_params.innerHorizon = false;
    //rpack_params.permutations = false;
    rpack_params.permutations = (charts.size() < 50);
    rpack_params.rotationNum = params.rotationNum;
    rpack_params.gutterWidth = 4;
    rpack_params.minmax = false; // not used

    int totPacked = 0;

    std::vector<int> containerIndices(outlines.size(), -1); // -1 means not packed to any container

    std::vector<vcg::Similarity2f> packingTransforms(outlines.size(), vcg::Similarity2f{});

    auto selectSubset = [&](const std::vector<unsigned>& eligible,
                            double targetUVArea) -> std::vector<unsigned> {
        if (eligible.empty()) return {};

        // Randomly shuffle eligible charts to obtain a representative subset
        std::vector<unsigned> indices = eligible;
        static thread_local std::mt19937 rng(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<unsigned> selected;
        selected.reserve(indices.size());
        double accum = 0.0;
        for (unsigned idx : indices) {
            if (accum >= targetUVArea) break;
            selected.push_back(idx);
            accum += chartAreas[idx];
        }
        if (selected.empty() && !indices.empty()) {
            selected.push_back(indices[0]);
        }

        // Ensure the final subset is sorted by size (area) for packing
        std::sort(selected.begin(), selected.end(), [&](unsigned a, unsigned b){
            return chartAreas[a] > chartAreas[b];
        });
        return selected;
    };

    unsigned nc = 0; // current container index
    while (totPacked < (int) charts.size()) {
        if (nc >= containerVec.size())
            containerVec.push_back(vcg::Point2i(packingSize, packingSize));

        // Build list of yet-unpacked chart indices
        std::vector<unsigned> pending;
        for (unsigned i = 0; i < containerIndices.size(); ++i) {
            if (containerIndices[i] == -1) {
                pending.push_back(i);
            }
        }

        if (pending.empty())
            break;

        const float QIMAGE_MAX_DIM = 32766.0f; // QImage limit is 32767
        std::vector<unsigned> eligible;
        eligible.reserve(pending.size());

        for(unsigned origIdx : pending) {
            if (outlines[origIdx].empty()) {
                LOG_WARN << "[DIAG] Skipping empty outline for original chart index " << origIdx;
                containerIndices[origIdx] = -2; // Mark as skipped
                totPacked++;
                continue;
            }

            vcg::Box2f bbox;
            for(const auto& p : outlines[origIdx]) bbox.Add(p);

            if (!std::isfinite(bbox.DimX()) || !std::isfinite(bbox.DimY()) || bbox.DimX() < 0 || bbox.DimY() < 0) {
                LOG_WARN << "[DIAG] Skipping chart with original index " << origIdx
                         << " due to invalid/non-finite UV bounding box. This chart will not be packed.";
                containerIndices[origIdx] = -4; // Mark as skipped due to invalid bbox
                totPacked++;
                continue;
            }

            float w = bbox.DimX() * packingScale;
            float h = bbox.DimY() * packingScale;
            float diagonal = std::sqrt(w * w + h * h);

            if (diagonal > QIMAGE_MAX_DIM) {
                LOG_WARN << "[DIAG] Skipping chart with original index " << origIdx
                         << " because its scaled diagonal (" << diagonal
                         << ") exceeds QImage limits. This chart will not be packed.";
                containerIndices[origIdx] = -3; // Mark as skipped due to size
                totPacked++;
                continue;
            }
            eligible.push_back(origIdx);
        }

        if (eligible.empty()) {
            if (totPacked < (int) charts.size()) continue;
            else break;
        }

        // Target subset with total UV area ~= 5x current grid area (convert to UV by dividing scale^2)
        double gridArea = double(containerVec[nc].X()) * double(containerVec[nc].Y());
        double targetUVArea = 5 * (gridArea / (packingScale * packingScale));

        std::vector<unsigned> selectedIdx = selectSubset(eligible, targetUVArea);
        // Prepare outlines list for the selected subset
        std::vector<Outline2f> outlines_iter;
        outlines_iter.reserve(selectedIdx.size());
        for (unsigned idx : selectedIdx) outlines_iter.push_back(outlines[idx]);

        LOG_INFO << "[DIAG] Subset selection: pending=" << pending.size()
                 << ", eligible=" << eligible.size()
                 << ", selected=" << outlines_iter.size()
                 << ", targetUVArea≈" << targetUVArea
                 << ", grid=" << containerVec[nc].X() << "x" << containerVec[nc].Y();

        if (outlines_iter.empty())
            continue;

        if (!outlines_iter.empty()) {
            size_t largest_outline_idx = 0;
            double max_area = 0;
            for(size_t i = 0; i < outlines_iter.size(); ++i) {
                vcg::Box2f bbox;
                for(const auto& p : outlines_iter[i]) bbox.Add(p);
                if (bbox.Area() > max_area) {
                    max_area = bbox.Area();
                    largest_outline_idx = i;
                }
            }
            size_t original_batch_idx = selectedIdx[largest_outline_idx];
            LOG_INFO << "[DIAG] Largest chart in this packing batch is index " << original_batch_idx
                     << " with UV area " << chartAreas[original_batch_idx];
        }
        const int MAX_SIZE = 20000;
        std::vector<vcg::Similarity2f> transforms;
        std::vector<int> polyToContainer;
        int n = 0;
        int packAttempts = 0;
        const int MAX_PACK_ATTEMPTS = 50;
        do {
            if (++packAttempts > MAX_PACK_ATTEMPTS) {
                LOG_ERR << "[DIAG] FATAL: Packing loop exceeded " << MAX_PACK_ATTEMPTS << " attempts. Aborting.";
                LOG_ERR << "[DIAG] This likely indicates an un-packable chart or runaway logic.";
                LOG_ERR << "[DIAG] Current target grid size: " << containerVec[nc].X() << "x" << containerVec[nc].Y();
                std::exit(-1);
            }
            transforms.clear();
            polyToContainer.clear();
            LOG_INFO << "Packing " << outlines_iter.size() << " charts into grid of size " << containerVec[nc].X() << " " << containerVec[nc].Y() << " (Attempt " << packAttempts << ")";
            n = RasterizationBasedPacker::PackBestEffortAtScale(outlines_iter, {containerVec[nc]}, transforms, polyToContainer, rpack_params, packingScale);
            LOG_INFO << "[DIAG] Packing attempt finished. Charts packed: " << n << ".";
            const auto& prof = RasterizationBasedPacker::LastProfile();
            LOG_INFO << "[PACK-PROF] polys=" << prof.polys_considered
                     << " placed=" << prof.placed_count << " not_placed=" << prof.not_placed_count
                     << " rasterize=" << prof.rasterize_s << "s (" << prof.rasterize_calls << " calls)"
                     << " candY(b/e)=" << prof.candidateY_build_s << "/" << prof.evaluate_drop_y_s << "s cols=" << prof.candidateY_cols_evaluated
                     << " candX(b/e)=" << prof.candidateX_build_s << "/" << prof.evaluate_drop_x_s << "s rows=" << prof.candidateX_rows_evaluated
                     << " place=" << prof.place_s << "s trans=" << prof.transform_s << "s total=" << prof.total_s << "s";
            if (n == 0 && !outlines_iter.empty()) {
                LOG_WARN << "[DIAG] Failed to pack any of the " << outlines_iter.size() << " charts in this batch.";
                containerVec[nc].X() *= 1.1;
                containerVec[nc].Y() *= 1.1;
            }
        } while (n == 0 && !outlines_iter.empty() && containerVec[nc].X() <= MAX_SIZE && containerVec[nc].Y() <= MAX_SIZE);

        if (n > 0) totPacked += n;

        if (n == 0) // no charts were packed, stop
            break;
        else {
            double textureScale = 1.0 / packingScale;
            texszVec.push_back({(int) (containerVec[nc].X() * textureScale), (int) (containerVec[nc].Y() * textureScale)});
            for (unsigned i = 0; i < outlines_iter.size(); ++i) {
                if (polyToContainer[i] != -1) {
                    ensure(polyToContainer[i] == 0); // We only use a single container
                    int outlineInd = selectedIdx[i];
                    ensure(containerIndices[outlineInd] == -1);
                    containerIndices[outlineInd] = nc;
                    packingTransforms[outlineInd] = transforms[i];
                }
            }
        }
        nc++;
    }

    // Helper function to round up to nearest power of two
    auto roundUpToPowerOfTwo = [](int v) -> int {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    };
    
    // Handle any remaining unpacked charts by giving them individual containers
    std::vector<unsigned> remainingUnpacked;
    for (unsigned ci = 0; ci < charts.size(); ++ci) {
        if (containerIndices[ci] == -1) {
            remainingUnpacked.push_back(ci);
        }
    }
    
    if (!remainingUnpacked.empty()) {
        LOG_INFO << "[DIAG] Creating individual containers for " << remainingUnpacked.size() << " unpacked charts.";
        
        for (unsigned ci : remainingUnpacked) {
            ChartHandle chartptr = charts[ci];
            auto outline = ExtractOutline2d(*chartptr);
            
            // Compute bounding box of the outline
            vcg::Box2d bb;
            for (const auto& p : outline) {
                bb.Add(p);
            }
            
            // Calculate container size based on bounding box and per-chart scale
            int padding = 8; // 8 pixel padding
            float mul = chartScaleMul[ci];
            int requiredWidth = (int)std::ceil(bb.DimX() * mul) + padding;
            int requiredHeight = (int)std::ceil(bb.DimY() * mul) + padding;
            
            // Round up to nearest power of two
            requiredWidth = roundUpToPowerOfTwo(requiredWidth);
            requiredHeight = roundUpToPowerOfTwo(requiredHeight);
            
            // Respect QImage's 32767 pixel limit
            const int MAX_QIMAGE_SIZE = 32767;
            requiredWidth = std::min(requiredWidth, MAX_QIMAGE_SIZE);
            requiredHeight = std::min(requiredHeight, MAX_QIMAGE_SIZE);
            
            // Create a new container for this chart
            vcg::Point2i containerSize(requiredWidth, requiredHeight);
            
            LOG_INFO << "[DIAG] Creating container " << requiredWidth << "x" << requiredHeight 
                     << " for chart " << ci << " (BB: " << bb.DimX() << "x" << bb.DimY() << ")";
            
            // Pack just this single chart into its dedicated container
            std::vector<std::vector<vcg::Point2f>> singleOutlineVec;
            singleOutlineVec.emplace_back();
            singleOutlineVec.back().reserve(outline.size());
            float mul = chartScaleMul[ci];
            for (const auto &p : outline) {
                singleOutlineVec.back().push_back(vcg::Point2f(static_cast<float>(p.X()*mul), static_cast<float>(p.Y()*mul)));
            }
 
            std::vector<vcg::Similarity2f> singleTrVec;
            std::vector<int> singlePolyToContainer;
 
             // Try to pack at scale 1.0 first
             bool packSuccess = Packer::PackAtFixedScale(
                 singleOutlineVec,
                 {containerSize},
                 singleTrVec,
                 singlePolyToContainer,
                 rpack_params,
                 1.0
             );
 
             // If it fails at scale 1.0, try with decreasing scales
             float scale = 1.0;
             while (!packSuccess && scale > 0.1) {
                 scale *= 0.9;
                 packSuccess = Packer::PackAtFixedScale(
                     singleOutlineVec,
                     {containerSize},
                     singleTrVec,
                     singlePolyToContainer,
                     rpack_params,
                     scale
                 );
             }
 
             if (packSuccess && !singlePolyToContainer.empty() && singlePolyToContainer[0] != -1) {
                 // Successfully packed the chart
                 containerIndices[ci] = nc; // Assign to the current container index
                 packingTransforms[ci] = singleTrVec[0];
 
                // Create a new texture/atlas for this chart
                TextureSize tsz;
                float textureScaleLocal = 1.0f / scale;
                tsz.w = (int)std::max(1.0f, std::floor(requiredWidth * textureScaleLocal));
                tsz.h = (int)std::max(1.0f, std::floor(requiredHeight * textureScaleLocal));
                texszVec.push_back(tsz);
                containerVec.push_back(containerSize); // Add the container to the list
                nc++; // Increment container count for next individual container
 
                 totPacked++;
 
                 LOG_INFO << "[DIAG] Successfully packed chart " << ci << " into individual container " << (nc-1) << " with scale: " << scale;
             } else {
                 LOG_ERR << "[DIAG] Failed to pack chart " << ci << " even in individual container";
             }
         }
     }

    for (unsigned i = 0; i < charts.size(); ++i) {
        for (auto fptr : charts[i]->fpVec) {
            int ic = containerIndices[i];
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
                    Point2f p = packingTransforms[i] * (Point2f(uv[0] * mul, uv[1] * mul));
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

    return totPacked;
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
        i = (outline2Vec.size() == 1) ? 0 : tri::OutlineUtil<double>::LargestOutline2(outline2Vec);
        if (tri::OutlineUtil<double>::Outline2Area(outline2Vec[i]) < 0)
            tri::OutlineUtil<double>::ReverseOutline2(outline2Vec[i]);
        vcg::Box2d outlineBox;
        for (const auto& p : outline2Vec[i])
            outlineBox.Add(p);
        if (outlineBox.DimX() < box.DimX() || outlineBox.DimY() < box.DimY())
            useChartBBAsOutline = true;
    }

    if (useChartBBAsOutline) {
        // --- [DIAGNOSTIC] START: Outline Failure Details ---
        LOG_WARN << "[DIAG] Failed to compute outline for chart " << chart.id
                 << ". It has " << chart.FN() << " faces."
                 << " BBox Area: " << box.Area()
                 << ". Falling back to UV bounding box.";
        // --- [DIAGNOSTIC] END ---
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
            for (int i = 0; i < 4; ++i) {
                double residual = VecAngle(Rotate(d0, angle[i]), d1);
                if (residual < minResidual) {
                    minResidual = residual;
                    minResidualIndex = i;
                }
            }

            int ti = fptr->cWT(0).N();
            ensure(ti < (int) texszVec.size());
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
                ensure(0 && "VERY BAD");
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
