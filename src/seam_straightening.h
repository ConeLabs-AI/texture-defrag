#ifndef SEAM_STRAIGHTENING_H
#define SEAM_STRAIGHTENING_H

#include "mesh_graph.h"
#include "types.h"
#include <vector>
#include <map>
#include <Eigen/Core>

/**
 * Based on the technique described in the paper:
 * Seamless:
 * Seam erasure and seam-aware decoupling of shape from mesh resolution
 * SONGRUN LIU∗ and ZACHARY FERGUSON∗, George Mason University
 * ALEC JACOBSON, University of Toronto
 * YOTAM GINGOLD, George Mason University
 */
namespace UVDefrag {

struct SeamStraighteningParameters {
    double initialTolerance = 0.01; // Recommended in "Seamless" paper - likely too high for large atlases
    double geometricTolerance = 1e-6; // Tolerance for welding 3D vertices at seams
    int maxWarpAttempts = 5;
    bool colorize = true;
    bool adaptiveTolerance = true; // Scale tolerance by sqrt(chart_median_area / global_median_area)
    double minTolerancePixels = 0.25;
    double maxTolerancePixels = 16.0;
};

void IntegrateSeamStraightening(GraphHandle graph, const SeamStraighteningParameters& params = SeamStraighteningParameters());

} // namespace UVDefrag

#endif // SEAM_STRAIGHTENING_H