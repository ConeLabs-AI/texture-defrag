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

#include "texture_optimization.h"

#include "mesh.h"
#include "mesh_graph.h"
#include "timer.h"
#include "texture_rendering.h"
#include "arap.h"
#include "mesh_attribute.h"
#include "logging.h"
#include "math_utils.h"

#include "shell.h"

#include <vcg/math/histogram.h>

#include <vector>
#include <algorithm>
#include <cmath>

#include <Eigen/SVD>


static void MirrorU(ChartHandle chart);

void ReorientCharts(GraphHandle graph)
{
    for (auto entry : graph->charts) {
        ChartHandle chart = entry.second;
        if (chart->UVFlipped())
            MirrorU(chart);
    }
}

int RotateChartForResampling(ChartHandle chart, const std::set<Mesh::FacePointer>& /*changeSet*/, const std::map<RegionID, bool> &flippedInput, bool colorize, double *zeroResamplingArea)
{
    Mesh& m = chart->mesh;
    auto wtcsh = GetWedgeTexCoordStorageAttribute(m);
    *zeroResamplingArea = 0;

    struct RigidCluster {
        double area3D = 0;
        double sinSum = 0;
        double cosSum = 0;
        int count = 0;
        Mesh::FacePointer anchor = nullptr;
    };

    // We cluster rotations into bins. Since we only care about rotations that 
    // can be recovered via 90-degree increments in packing, we can cluster 
    // the "base" rotation angle (mod 90).
    // Actually, the paper says "maximal set ... sharing the same transformation matrix".
    // That means same rotation angle theta.
    
    std::vector<RigidCluster> clusters;
    const double ANGLE_TOLERANCE = 2.0 * M_PI / 180.0; // 2 degrees
    const double RIGID_TOLERANCE = 0.05; // 5% deviation in singular values

    // Per-face storage to track cluster membership for later colorization
    std::map<Mesh::FacePointer, int> faceClusterIdx;

    for (auto fptr : chart->fpVec) {
        double areaUV_original;
        double area3D = Area3D(*fptr);
        
        vcg::Point2d x10 = wtcsh[fptr].tc[1].P() - wtcsh[fptr].tc[0].P();
        vcg::Point2d x20 = wtcsh[fptr].tc[2].P() - wtcsh[fptr].tc[0].P();
        vcg::Point2d u10 = fptr->WT(1).P() - fptr->WT(0).P();
        vcg::Point2d u20 = fptr->WT(2).P() - fptr->WT(0).P();

        areaUV_original = std::abs(x10 ^ x20);
        if (areaUV_original <= 1e-12) continue;

        Eigen::Matrix2d Jf = ComputeTransformationMatrix(x10, x20, u10, u20);
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(Jf, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d sigma = svd.singularValues();

        // Check if nearly rigid (isometry)
        if (std::abs(sigma[0] - 1.0) < RIGID_TOLERANCE && std::abs(sigma[1] - 1.0) < RIGID_TOLERANCE) {
            Eigen::Matrix2d R = svd.matrixU() * svd.matrixV().transpose();
            if (R.determinant() < 0) {
                Eigen::Matrix2d U = svd.matrixU();
                U.col(1) *= -1;
                R = U * svd.matrixV().transpose();
            }
            
            double angle = std::atan2(R(1, 0), R(0, 0));
            
            bool found = false;
            for (int ci = 0; ci < (int)clusters.size(); ++ci) {
                auto& c = clusters[ci];
                double avgAngle = std::atan2(c.sinSum / c.count, c.cosSum / c.count);
                double diff = std::abs(angle - avgAngle);
                if (diff > M_PI) diff = 2.0 * M_PI - diff;
                if (diff < ANGLE_TOLERANCE) {
                    c.area3D += area3D;
                    c.sinSum += std::sin(angle);
                    c.cosSum += std::cos(angle);
                    c.count++;
                    if (!c.anchor || area3D > Area3D(*(c.anchor))) c.anchor = fptr;
                    faceClusterIdx[fptr] = ci;
                    found = true;
                    break;
                }
            }
            if (!found) {
                faceClusterIdx[fptr] = (int)clusters.size();
                clusters.push_back({area3D, std::sin(angle), std::cos(angle), 1, fptr});
            }
        }
    }

    if (clusters.empty()) {
        return -1;
    }

    // Pick the cluster with the largest 3D area
    int bestClusterIdx = 0;
    double maxArea = -1.0;
    for (int i = 0; i < (int)clusters.size(); ++i) {
        if (clusters[i].area3D > maxArea) {
            maxArea = clusters[i].area3D;
            bestClusterIdx = i;
        }
    }
    const auto& bestCluster = clusters[bestClusterIdx];

    *zeroResamplingArea = bestCluster.area3D;
    double rotAngle = std::atan2(bestCluster.sinSum / bestCluster.count, bestCluster.cosSum / bestCluster.count);
    Mesh::FacePointer anchorFp = bestCluster.anchor;

    // rotate the uvs
    for (auto fptr : chart->fpVec) {
        for (int i = 0; i < 3; ++i) {
            fptr->WT(i).P().Rotate(-rotAngle); // Rotate back to align with grid
            fptr->V(i)->T().P() = fptr->WT(i).P();
        }
        if (colorize && faceClusterIdx.count(fptr) && faceClusterIdx[fptr] == bestClusterIdx) {
            fptr->C() = vcg::Color4b(85, 246, 85, 255);
        }
    }

    return tri::Index(chart->mesh, anchorFp);
}

void TrimTexture(Mesh& m, std::vector<TextureSize>& texszVec, bool unsafeMip)
{
    std::vector<std::vector<Mesh::FacePointer>> facesByTexture;
    unsigned ntex = FacesByTextureIndex(m, facesByTexture);

    // Validate mapping between texture indices and provided sizes
    if (ntex > texszVec.size()) {
        LOG_ERR << "[VALIDATION] TrimTexture: texture index count (" << ntex
                << ") exceeds size entries (" << texszVec.size() << "). Aborting.";
        std::exit(-1);
    }

    // Validate each target sheet size against QImage constraints and sanity
    {
        const int MAX_QIMAGE_SIZE = 32767;
        for (unsigned ti = 0; ti < ntex; ++ti) {
            if (texszVec[ti].w <= 0 || texszVec[ti].h <= 0) {
                LOG_ERR << "[VALIDATION] TrimTexture: non-positive sheet size at index " << ti
                        << ": " << texszVec[ti].w << "x" << texszVec[ti].h << ". Aborting.";
                std::exit(-1);
            }
            if (texszVec[ti].w > MAX_QIMAGE_SIZE || texszVec[ti].h > MAX_QIMAGE_SIZE) {
                LOG_ERR << "[VALIDATION] TrimTexture: sheet size at index " << ti
                        << " exceeds QImage limit: " << texszVec[ti].w << "x" << texszVec[ti].h
                        << " (max " << MAX_QIMAGE_SIZE << "). Aborting.";
                std::exit(-1);
            }
        }
    }

    LOG_INFO << "[DIAG] TrimTexture: ntex=" << ntex << ", texszVec.size()=" << texszVec.size();

    for (unsigned ti = 0; ti < ntex; ++ti) {
        vcg::Box2d uvBox;
        for (auto fptr : facesByTexture[ti]) {
            double a = AreaUV(*fptr);
            if (std::isfinite(a) && a != 0) {
                for (int i = 0; i < 3; ++i) {
                    uvBox.Add(fptr->WT(i).P());
                }
            }
        }

        if (std::min(uvBox.DimX(), uvBox.DimY()) > 0.95)
            continue;

        uvBox.min.Scale(texszVec[ti].w, texszVec[ti].h);
        uvBox.max.Scale(texszVec[ti].w, texszVec[ti].h);
        uvBox.min.X() = std::max(0, int(uvBox.min.X()) - 2);
        uvBox.min.Y() = std::max(0, int(uvBox.min.Y()) - 2);
        uvBox.max.X() = std::min(texszVec[ti].w, int(uvBox.max.X()) + 2);
        uvBox.max.Y() = std::min(texszVec[ti].h, int(uvBox.max.Y()) + 2);

        if (!unsafeMip) {
            // pad the bbox so that MIP artifacts only occur at level above
            const int MAX_SAFE_MIP_LEVEL = 5;
            const int MOD_VAL = (1 << MAX_SAFE_MIP_LEVEL);

            int bboxw = uvBox.max.X() - uvBox.min.X();
            int bboxh = uvBox.max.Y() - uvBox.min.Y();

            int incw = MOD_VAL - (bboxw % MOD_VAL);
            int inch = MOD_VAL - (bboxh % MOD_VAL);

            uvBox.max.X() += incw;
            uvBox.max.Y() += inch;
        }

        if (!std::isfinite(uvBox.DimX()) || !std::isfinite(uvBox.DimY()) || uvBox.DimX() <= 0 || uvBox.DimY() <= 0) {
            LOG_WARN << "[DIAG] TrimTexture: invalid UV bbox for texture index " << ti << ", skipping trim.";
            continue;
        }

        double uscale = texszVec[ti].w / uvBox.DimX();
        double vscale = texszVec[ti].h / uvBox.DimY();

        vcg::Point2d t(uvBox.min.X() / texszVec[ti].w, uvBox.min.Y() / texszVec[ti].h);

        for (auto fptr : facesByTexture[ti]) {
            double a = AreaUV(*fptr);
            if (std::isfinite(a) && a != 0) {
                for (int i = 0; i < 3; ++i) {
                    // Translate to new origin and scale to [0,1] domain of the trimmed texture
                    fptr->WT(i).P() -= t;
                    fptr->WT(i).P().Scale(uscale, vscale);

                    // Clamp to [0,1) to avoid precision spill outside and texture wrap
                    const double oneMinus = std::nextafter(1.0, 0.0);
                    auto &uvP = fptr->WT(i).P();
                    if (uvP.X() < 0.0) uvP.X() = 0.0; else if (uvP.X() > oneMinus) uvP.X() = oneMinus;
                    if (uvP.Y() < 0.0) uvP.Y() = 0.0; else if (uvP.Y() > oneMinus) uvP.Y() = oneMinus;

                    fptr->V(i)->T().P() = fptr->WT(i).P();
                }
            }
        }

        // sanity check
        {
            vcg::Box2d uvBoxCheck;
            for (auto fptr : facesByTexture[ti]) {
                double a = AreaUV(*fptr);
                if (std::isfinite(a) && a != 0) {
                    for (int i = 0; i < 3; ++i) {
                        uvBoxCheck.Add(fptr->WT(i).P());
                    }
                }
            }

            const double EPS = 1e-9;
            ensure(uvBoxCheck.min.X() >= -EPS);
            ensure(uvBoxCheck.min.Y() >= -EPS);
            ensure(uvBoxCheck.max.X() <= 1.0 + EPS);
            ensure(uvBoxCheck.max.Y() <= 1.0 + EPS);
        }

        // resize
        texszVec[ti].w = (int) uvBox.DimX();
        texszVec[ti].h = (int) uvBox.DimY();
    }
}

// -- static functions ---------------------------------------------------------

static void MirrorU(ChartHandle chart)
{
    double u_old = chart->UVBox().min.X();
    for (auto fptr : chart->fpVec) {
        for (int i = 0; i < 3; ++i)
            fptr->WT(i).U() *= -1;
    }
    chart->ParameterizationChanged();
    double u_new = chart->UVBox().min.X();
    for (auto fptr : chart->fpVec) {
        for (int i = 0; i < 3; ++i) {
            fptr->WT(i).U() += (u_old - u_new);
            fptr->V(i)->T().U() = fptr->WT(i).U();
        }
    }
    chart->ParameterizationChanged();
}