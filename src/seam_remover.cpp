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

#include "seam_remover.h"
#include "mesh.h"
#include "mesh_attribute.h"
#include "mesh_graph.h"
#include "matching.h"
#include "intersection.h"
#include "shell.h"
#include "arap.h"
#include "timer.h"
#include "logging.h"
#include "seams.h"
#include "texture_rendering.h"


#include <cmath>
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <string>
#include <atomic>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/append.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#endif


constexpr double PENALTY_MULTIPLIER = 2.0;


struct Perf {
    double t_init;
    double t_seamdata;
    double t_alignmerge;
    double t_optimization_area;
    double t_optimize;
    double t_optimize_build;
    double t_optimize_arap;
    double t_check_before;
    double t_check_after;
    double t_accept;
    double t_reject;
    Timer timer;
};


static void InsertNewClusterInQueue(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph, const AlgoParameters& params);
static CostInfo ComputeCost(ClusteredSeamHandle csh, GraphHandle graph, const AlgoParameters& params, double penalty);
static inline double GetPenalty(ClusteredSeamHandle csh, AlgoStateHandle state);
static inline bool Valid(const WeightedSeam& ws, ConstAlgoStateHandle state);
static inline void PurgeQueue(AlgoStateHandle state);
static void ComputeSeamData(SeamData& sd, ClusteredSeamHandle csh, GraphHandle graph, AlgoStateHandle state);
static OffsetMap AlignAndMerge(ClusteredSeamHandle csh, SeamData& sd, const MatchingTransform& mi, const AlgoParameters& params);
static void ComputeOptimizationArea(SeamData& sd, Mesh& mesh, OffsetMap& om);
static std::unordered_set<Mesh::VertexPointer> ComputeVerticesWithinOffsetThreshold(Mesh& m, const OffsetMap& om, const SeamData& sd);
static CheckStatus CheckBoundaryAfterAlignment(SeamData& sd);
static CheckStatus CheckAfterLocalOptimization(SeamData& sd, AlgoStateHandle state, const AlgoParameters& params);
static CheckStatus OptimizeChart(SeamData& sd, GraphHandle graph, bool fixIntersectingEdges);
static void AcceptMove(const SeamData& sd, AlgoStateHandle state, GraphHandle graph, const AlgoParameters& params);
static void RejectMove(const SeamData& sd, AlgoStateHandle state, GraphHandle graph, CheckStatus status);
static void EraseSeam(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph);
static void InvalidateCluster(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph, CheckStatus status, double penaltyMultiplier);
static void RestoreChartAttributes(ChartHandle c, Mesh& m, std::vector<int>::const_iterator itvi,  std::vector<vcg::Point2d>::const_iterator ittc);
static CostInfo ReduceSeam(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph, const AlgoParameters& params);


Perf perf = {};

static void LogArapExplosionDiagnostics(Mesh& shell) {
    int numComponents = tri::Clean<Mesh>::CountConnectedComponents(shell);
    LOG_WARN << "  Diagnostic: Shell has " << numComponents << " components";

    // Simple BFS to count fixed vertices per component
    tri::UpdateFlags<Mesh>::VertexClearV(shell);
    int compIdx = 0;
    for (auto& startV : shell.vert) {
        if (startV.IsD() || startV.IsV()) continue;
        
        int vCount = 0;
        int fixedCount = 0;
        std::vector<Mesh::VertexPointer> q;
        q.push_back(&startV);
        startV.SetV();
        
        while(!q.empty()) {
            Mesh::VertexPointer v = q.back(); q.pop_back();
            vCount++;
            if (v->IsS()) fixedCount++;
            
            if (v->VFp() != nullptr) {
                face::VFIterator<MeshFace> vfi(v);
                for (; !vfi.End(); ++vfi) {
                    MeshFace* f = vfi.F();
                    for (int i = 0; i < 3; ++i) {
                        Mesh::VertexPointer nv = f->V(i);
                        if (!nv->IsD() && !nv->IsV()) {
                            nv->SetV();
                            q.push_back(nv);
                        }
                    }
                }
            }
        }
        LOG_WARN << "  Component " << compIdx++ << ": " << vCount << " vertices, " << fixedCount << " fixed";
    }
}

#define PERF_TIMER_RESET (perf = {}, perf.timer.Reset())
#define PERF_TIMER_START double perf_timer_t0 = perf.timer.TimeElapsed()
#define PERF_TIMER_ACCUMULATE(field) perf.field += perf.timer.TimeElapsed() - perf_timer_t0
#define PERF_TIMER_ACCUMULATE_FROM_PREVIOUS(field) perf.field += perf.timer.TimeSinceLastCheck()

//static int statsCheck[10] = {};
//static int feasibility[6] = {};

static std::vector<int> statsCheck(CheckStatus::_END, 0);
static std::vector<int> feasibility(CostInfo::MatchingValue::_END, 0);

static vcg::Color4b statusColor[] = {
    vcg::Color4b::White, // PASS=0,
    vcg::Color4b::Gray , // FAIL_LOCAL_OVERLAP,
    vcg::Color4b::Red, // FAIL_GLOBAL_OVERLAP_BEFORE,
    vcg::Color4b::Green, // FAIL_GLOBAL_OVERLAP_AFTER_OPT, // border of the optimization area self-intersects
    vcg::Color4b::LightGreen, // FAIL_GLOBAL_OVERLAP_AFTER_BND, // border of the optimzation area hit the fixed border
    vcg::Color4b::LightBlue, // FAIL_DISTORTION_LOCAL,
    vcg::Color4b::Blue, // FAIL_DISTORTION_LOCAL,
    vcg::Color4b::LightRed, // FAIL_TOPOLOGY
    vcg::Color4b::Yellow, // FAIL_NUMERICAL_ERROR
    vcg::Color4b::White, // UNKNOWN
    vcg::Color4b(176, 0, 255, 255) // FAIL_GLOBAL_OVERLAP_UNFIXABLE
};

static vcg::Color4b mvColor[] = {
    vcg::Color4b::White, //   FEASIBLE=0,
    vcg::Color4b::Black, //   ZERO_AREA,
    vcg::Color4b::Cyan, //   UNFEASIBLE_BOUNDARY,
    vcg::Color4b::Magenta //   UNFEASIBLE_MATCHING,
};

static int accept = 0;
static int reject = 0;

static int num_retry = 0;
static int retry_success = 0;

double mincost = 100000;
double maxcost = -1;

double min_energy = 10000000000;
double max_energy = 0;

static void ClearGlobals()
{
    for (unsigned i = 0; i < statsCheck.size(); ++i)
        statsCheck[i] = 0;
    for (unsigned i = 0; i < feasibility.size(); ++i)
        feasibility[i] = 0;
    accept = 0;
    reject = 0;

    num_retry = 0;
    retry_success = 0;
}

static void LogMemoryUsage()
{
#if defined(__APPLE__)
    // macOS implementation
    mach_port_t host_port = mach_host_self();
    mach_msg_type_number_t host_size = sizeof(vm_statistics64_data_t) / sizeof(integer_t);
    vm_size_t pagesize;
    host_page_size(host_port, &pagesize);

    vm_statistics64_data_t vm_stat;
    if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
        LOG_WARN << "Failed to fetch macOS vm statistics";
        return;
    }

    uint64_t total_mem_val;
    size_t len = sizeof(total_mem_val);
    if (sysctlbyname("hw.memsize", &total_mem_val, &len, NULL, 0) != 0) {
        LOG_WARN << "Failed to fetch macOS total memory";
        return;
    }

    uint64_t used_memory = (vm_stat.active_count + vm_stat.inactive_count + vm_stat.wire_count) * (uint64_t)pagesize;

    double used_mem_gb = (double)used_memory / (1024.0 * 1024.0 * 1024.0);
    double total_mem_gb = (double)total_mem_val / (1024.0 * 1024.0 * 1024.0);

    LOG_INFO << "System RAM: " << std::fixed << std::setprecision(2) << used_mem_gb << " / " << total_mem_gb << " GB used";

#elif defined(__linux__)
    // Linux implementation
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        LOG_WARN << "Could not open /proc/meminfo to read memory stats";
        return;
    }

    std::string line;
    long long mem_total = -1, mem_available = -1;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            try { mem_total = std::stoll(line.substr(10)); } catch (...) {}
        }
        if (line.rfind("MemAvailable:", 0) == 0) {
            try { mem_available = std::stoll(line.substr(13)); } catch (...) {}
        }
    }

    if (mem_total != -1 && mem_available != -1) {
        long long used_mem = mem_total - mem_available; // in kB
        double used_mem_gb = (double)used_mem / (1024.0 * 1024.0);
        double total_mem_gb = (double)mem_total / (1024.0 * 1024.0);
        LOG_INFO << "System RAM: " << std::fixed << std::setprecision(2) << used_mem_gb << " / " << total_mem_gb << " GB used";
    } else {
        LOG_WARN << "Could not parse MemTotal/MemAvailable from /proc/meminfo";
    }

#elif defined(_WIN32)
    // Windows implementation
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        double total_mem_gb = (double)statex.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
        double used_mem_gb = (double)(statex.ullTotalPhys - statex.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);

        LOG_INFO << "System RAM: " << std::fixed << std::setprecision(2) << used_mem_gb << " / " << total_mem_gb << " GB used";
    } else {
        LOG_WARN << "Windows GlobalMemoryStatusEx failed.";
    }
#else
    LOG_WARN << "Memory usage logging not implemented for this platform.";
#endif
}

void LogExecutionStats()
{
    LogMemoryUsage();
    LOG_INFO    << "======== EXECUTION STATS ========";
    LOG_INFO    << "INIT       " << std::fixed << std::setprecision(3) << perf.t_init / perf.timer.TimeElapsed()                                << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_init << " secs";
    LOG_INFO    << "SEAM       " << std::fixed << std::setprecision(3) << perf.t_seamdata / perf.timer.TimeElapsed()                            << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_seamdata << " secs";
    LOG_INFO    << "MERGE      " << std::fixed << std::setprecision(3) << perf.t_alignmerge / perf.timer.TimeElapsed()                          << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_alignmerge << " secs";
    LOG_INFO    << "AREA OPT   " << std::fixed << std::setprecision(3) << perf.t_optimization_area / perf.timer.TimeElapsed()                   << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_optimization_area << " secs";
    LOG_INFO    << "OPTIMIZE   " << std::fixed << std::setprecision(3) << perf.t_optimize / perf.timer.TimeElapsed()                            << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_optimize << " secs";
    LOG_VERBOSE << "  BUILD    " << std::fixed << std::setprecision(3) << perf.t_optimize_build / perf.timer.TimeElapsed()                      << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_optimize_build << " secs";
    LOG_VERBOSE << "  ARAP     " << std::fixed << std::setprecision(3) << perf.t_optimize_arap / perf.timer.TimeElapsed()                       << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_optimize_arap << " secs";
    LOG_INFO    << "CHECK      " << std::fixed << std::setprecision(3) << (perf.t_check_before + perf.t_check_after) / perf.timer.TimeElapsed() << " , " << std::defaultfloat << std::setprecision(6)<< (perf.t_check_before + perf.t_check_after) << " secs";
    LOG_VERBOSE << "  BEFORE   " << std::fixed << std::setprecision(3) << perf.t_check_before / perf.timer.TimeElapsed()                        << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_check_before << " secs";
    LOG_VERBOSE << "  AFTER    " << std::fixed << std::setprecision(3) << perf.t_check_after / perf.timer.TimeElapsed()                         << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_check_after << " secs";
    LOG_INFO    << "ACCEPT     " << std::fixed << std::setprecision(3) << perf.t_accept / perf.timer.TimeElapsed()                              << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_accept << " secs";
    LOG_INFO    << "  count:                    " << accept;
    LOG_INFO    << "  with retry:               " << retry_success;
    LOG_VERBOSE << "  min energy:               " << min_energy;
    LOG_VERBOSE << "  max energy:               " << max_energy;
    LOG_INFO    << "REJECT     " << std::fixed << std::setprecision(3) << perf.t_reject / perf.timer.TimeElapsed()                              << " , " << std::defaultfloat << std::setprecision(6)<< perf.t_reject << " secs";
    LOG_INFO    << "  count:                    " << reject;
    LOG_INFO    << "  with retry:               " << num_retry - retry_success;
    LOG_VERBOSE << "  local overlaps            " << statsCheck[FAIL_LOCAL_OVERLAP];
    LOG_VERBOSE << "  global overlaps before    " << statsCheck[FAIL_GLOBAL_OVERLAP_BEFORE];
    LOG_VERBOSE << "  global overlaps after opt " << statsCheck[FAIL_GLOBAL_OVERLAP_AFTER_OPT];
    LOG_VERBOSE << "  global overlaps after bnd " << statsCheck[FAIL_GLOBAL_OVERLAP_AFTER_BND];
    LOG_VERBOSE << "  global overlaps unfixable " << statsCheck[FAIL_GLOBAL_OVERLAP_UNFIXABLE];
    LOG_VERBOSE << "  distortion (local)        " << statsCheck[FAIL_DISTORTION_LOCAL];
    LOG_VERBOSE << "  distortion (global)       " << statsCheck[FAIL_DISTORTION_GLOBAL];
    LOG_VERBOSE << "  topology                  " << statsCheck[FAIL_TOPOLOGY];
    LOG_VERBOSE << "  numerical error           " << statsCheck[FAIL_NUMERICAL_ERROR];
    LOG_VERBOSE << "    FEASIBILITY";
    LOG_VERBOSE << "      feasible              " << feasibility[CostInfo::FEASIBLE];
    LOG_VERBOSE << "      unfeasible boundary   " << feasibility[CostInfo::UNFEASIBLE_BOUNDARY];
    LOG_VERBOSE << "      unfeasible matching   " << feasibility[CostInfo::UNFEASIBLE_MATCHING];
    LOG_INFO    << "TOTAL      " << std::fixed << std::setprecision(3) << perf.timer.TimeElapsed() / perf.timer.TimeElapsed()          << " , " << std::defaultfloat << std::setprecision(6)<< perf.timer.TimeElapsed() << " secs";
    LOG_VERBOSE << "Minimum computed cost is " << mincost;
    LOG_VERBOSE << "Maximum computed cost is " << maxcost;
    LOG_INFO    << "===================================";
}

static void PrintStateInfo(AlgoStateHandle state, GraphHandle graph, const AlgoParameters& params)
{
    std::set<ClusteredSeamHandle> moveSet;

    for (auto& entry : state->chartSeamMap) {
        for (auto csh : entry.second) {
            moveSet.insert(csh);
        }
    }

    LOG_VERBOSE << "Status of the residual " << moveSet.size() << " operations:";

    int nstat[100] = {};
    int mstat[100] = {};
    for (auto csh : moveSet) {
        auto it = state->status.find(csh);
        ensure(it != state->status.end());
        ensure(it->second != PASS);
        CostInfo ci = ComputeCost(csh, graph, params, GetPenalty(csh, state));
        nstat[state->status[csh]]++;
        mstat[ci.mvalue]++;
    }

    LOG_VERBOSE << "PASS                          " << nstat[CheckStatus::PASS];
    LOG_VERBOSE << "FAIL_LOCAL_OVERLAP            " << nstat[CheckStatus::FAIL_LOCAL_OVERLAP];
    LOG_VERBOSE << "FAIL_GLOBAL_OVERLAP_BEFORE    " << nstat[CheckStatus::FAIL_GLOBAL_OVERLAP_BEFORE];
    LOG_VERBOSE << "FAIL_GLOBAL_OVERLAP_AFTER_OPT " << nstat[CheckStatus::FAIL_GLOBAL_OVERLAP_AFTER_OPT];
    LOG_VERBOSE << "FAIL_GLOBAL_OVERLAP_AFTER_BND " << nstat[CheckStatus::FAIL_GLOBAL_OVERLAP_AFTER_BND];
    LOG_VERBOSE << "FAIL_GLOBAL_OVERLAP_UNFIXABLE " << nstat[CheckStatus::FAIL_GLOBAL_OVERLAP_UNFIXABLE];
    LOG_VERBOSE << "FAIL_DISTORTION_LOCAL         " << nstat[CheckStatus::FAIL_DISTORTION_LOCAL];
    LOG_VERBOSE << "FAIL_DISTORTION_GLOBAL        " << nstat[CheckStatus::FAIL_DISTORTION_GLOBAL];
    LOG_VERBOSE << "FAIL_TOPOLOGY                 " << nstat[CheckStatus::FAIL_TOPOLOGY];
    LOG_VERBOSE << "FAIL_NUMERICAL_ERROR          " << nstat[CheckStatus::FAIL_NUMERICAL_ERROR];
    LOG_VERBOSE << "UNKNOWN                       " << nstat[CheckStatus::UNKNOWN];
    LOG_VERBOSE << "  - FEASIBLE                         " << mstat[CostInfo::MatchingValue::FEASIBLE];
    LOG_VERBOSE << "  - ZERO_AREA                        " << mstat[CostInfo::MatchingValue::ZERO_AREA];
    LOG_VERBOSE << "  - UNFEASIBLE_BOUNDARY              " << mstat[CostInfo::MatchingValue::UNFEASIBLE_BOUNDARY];
    LOG_VERBOSE << "  - UNFEASIBLE_MATCHING              " << mstat[CostInfo::MatchingValue::UNFEASIBLE_MATCHING];
    LOG_VERBOSE << "  - REJECTED                         " << mstat[CostInfo::MatchingValue::REJECTED];

}

void PrepareMesh(Mesh& m, int *vndup)
{
    int dupVert = tri::Clean<Mesh>::RemoveDuplicateVertex(m);
    if (dupVert > 0)
        LOG_INFO << "Removed " << dupVert << " duplicate vertices";

    int zeroArea = tri::Clean<Mesh>::RemoveZeroAreaFace(m);
    if (zeroArea > 0)
        LOG_INFO << "Removed " << zeroArea << " zero area faces";

    tri::UpdateTopology<Mesh>::FaceFace(m);

    // orient faces coherently
    bool wasOriented, isOrientable;
    tri::Clean<Mesh>::OrientCoherentlyMesh(m, wasOriented, isOrientable);

    tri::UpdateTopology<Mesh>::FaceFace(m);

    int numRemovedFaces = tri::Clean<Mesh>::RemoveNonManifoldFace(m);
    if (numRemovedFaces > 0)
        LOG_INFO << "Removed " << numRemovedFaces << " non-manifold faces";

    tri::Allocator<Mesh>::CompactEveryVector(m);
    tri::UpdateTopology<Mesh>::FaceFace(m);

    Compute3DFaceAdjacencyAttribute(m);

    CutAlongSeams(m);
    tri::Allocator<Mesh>::CompactEveryVector(m);

    *vndup = m.VN();

    tri::UpdateTopology<Mesh>::FaceFace(m);
    while (tri::Clean<Mesh>::SplitNonManifoldVertex(m, 0))
        ;
    tri::UpdateTopology<Mesh>::VertexFace(m);

    tri::Allocator<Mesh>::CompactEveryVector(m);
}

AlgoStateHandle InitializeState(GraphHandle graph, const AlgoParameters& algoParameters)
{
    PERF_TIMER_RESET;
    PERF_TIMER_START;

    AlgoStateHandle state = std::make_shared<AlgoState>();
    ARAP::ComputeEnergyFromStoredWedgeTC(graph->mesh, &state->arapNum, &state->arapDenom);
    state->inputUVBorderLength = 0;
    state->currentUVBorderLength = 0;

    BuildSeamMesh(graph->mesh, state->sm);
    std::vector<SeamHandle> seams = GenerateSeams(state->sm);

    // disconnecting seams are (initially) clustered by chart adjacency
    // non-disconnecting seams are not clustered (segment granularity)

    std::vector<ClusteredSeamHandle> cshvec = ClusterSeamsByChartId(seams);
    int ndisconnecting = 0;
    int nself = 0;

    for (auto csh : cshvec) {
        ChartPair charts = GetCharts(csh, graph);
        if (charts.first == charts.second)
            nself++;
        else
            ndisconnecting++;
        InsertNewClusterInQueue(csh, state, graph, algoParameters);
    }
    LOG_INFO << "Found " << ndisconnecting << " disconnecting seams";
    LOG_INFO << "Found " << nself << " non-disconnecting seams";

    // sanity check
    for (auto& entry : state->chartSeamMap) {
        ensure(entry.second.size() >= (graph->GetChart(entry.first)->adj.size()));
    }

    for (const auto& ch : graph->charts) {
        state->inputUVBorderLength += ch.second->BorderUV();
        state->currentUVBorderLength += ch.second->BorderUV();
    }

    PERF_TIMER_ACCUMULATE(t_init);
    return state;
}

void GreedyOptimization(GraphHandle graph, AlgoStateHandle state, const AlgoParameters& params)
{
    ClearGlobals();

    Timer t;
    Timer tglobal;

    PrintStateInfo(state, graph, params);

    LOG_INFO << "Atlas energy before optimization is " << ARAP::ComputeEnergyFromStoredWedgeTC(graph->mesh, nullptr, nullptr);
    LOG_INFO << "Starting greedy optimization loop with " << state->queue.size() << " operations...";

    // Use a relative logging frequency (e.g., every 5% of estimated total, or min 5000)
    // to avoid excessive logs on very large models (1M+ charts).
    size_t nCharts = graph->Count();
    int logInterval = std::max(5000, (int)(nCharts / 20));
    LOG_INFO << "Log frequency set to every " << logInterval << " iterations (based on " << nCharts << " charts).";

    int k = 0;
    while (state->queue.size() > 0) {
        if (k < 100) {
            LOG_INFO << "Iteration " << k << " - Queue size: " << state->queue.size();
        }

        if (state->queue.size() > 2 * state->cost.size())
            PurgeQueue(state);

        if (state->queue.size() == 0) {
            LOG_INFO << "Queue is empty, interrupting.";
            break;
        }

        if (params.timelimit > 0 && t.TimeElapsed() > params.timelimit) {
            LOG_INFO << "Timelimit hit, interrupting.";
            break;
        }

        if (params.UVBorderLengthReduction > (state->currentUVBorderLength / state->inputUVBorderLength)) {
            LOG_INFO << "Target UV border reduction reached, interrupting.";
            break;
        }

        WeightedSeam ws = state->queue.top();
        state->queue.pop();
        if (!Valid(ws, state)) {
            if (k < 1000 && (state->queue.size() % 1000 == 0)) {
                LOG_INFO << "  [Iter " << k << "] Skipping invalid queue entry, queue size: " << state->queue.size();
            }
            continue;
        }

        if (ws.second == Infinity()) {
            // sanity check
            for (auto& entry : state->cost)
                ensure(!std::isfinite(entry.second));
            LOG_INFO << "Queue is empty, interrupting.";
            break;
        } else {
            Timer iterTimer;
            ++k;
            if ((k % logInterval) == 0) {
                LOG_INFO << "Logging execution stats after " << k << " iterations";
                LogExecutionStats();
            }

            SeamData sd;
            ComputeSeamData(sd, ws.first, graph, state);

            if (k <= 100) {
                LOG_INFO << "  [Iter " << k << "] Processing charts " << sd.a->id << " and " << sd.b->id
                         << " (areas UV: " << sd.a->AreaUV() << ", " << sd.b->AreaUV() << ")";
            }

            LOG_DEBUG << "  Chart ids are " << sd.a->id << " " << sd.b->id << " (areas = " << sd.a->AreaUV() << ", " << sd.b->AreaUV() << ")";

            OffsetMap om = AlignAndMerge(ws.first, sd, state->transform[ws.first], params);

            ComputeOptimizationArea(sd, graph->mesh, om);

            if (k <= 100) {
                LOG_INFO << "  [Iter " << k << "] Optimization area size: " << sd.optimizationArea.size() << " faces";
            }

            // when merging two charts, check if they collide outside the optimization area

            CheckStatus status = (sd.a != sd.b) ? CheckBoundaryAfterAlignment(sd) : PASS;

            if (k <= 100 && status != PASS) {
                LOG_INFO << "  [Iter " << k << "] CheckBoundaryAfterAlignment failed: " << status;
            }

            if (status == PASS) {
                if (k <= 100) LOG_INFO << "  [Iter " << k << "] Using OptimizeChart";
                status = OptimizeChart(sd, graph, false);

                if (status == PASS) {
                    status = CheckAfterLocalOptimization(sd, state, params);

                    if (k <= 100 && status != PASS) {
                        LOG_INFO << "  [Iter " << k << "] CheckAfterLocalOptimization failed: " << status;
                    }
                } else if (k <= 100) {
                    LOG_INFO << "  [Iter " << k << "] Optimization failed: " << status;
                }
            }

            int retryCount = 0;
            const int MAX_RETRIES = 10;
            while ((status == FAIL_GLOBAL_OVERLAP_AFTER_OPT || status == FAIL_GLOBAL_OVERLAP_AFTER_BND) && retryCount < MAX_RETRIES) {
                retryCount++;
                LOG_DEBUG << "Global overlaps detected after ARAP optimization, fixing edges";
                if (k <= 100) LOG_INFO << "  [Iter " << k << "] Global overlap detected, retrying with edge fixing (" << retryCount << ")";
                CheckStatus iterStatus = OptimizeChart(sd, graph, true);
                if (iterStatus == _END)
                    break;
                if (iterStatus != PASS) {
                    status = iterStatus;
                    break;
                }
                status = CheckAfterLocalOptimization(sd, state, params);
            }

            if (retryCount == MAX_RETRIES) {
                LOG_DEBUG << "Reached MAX_RETRIES for edge fixing, giving up";
            }

            statsCheck[status]++;

            if (status == PASS) {
                AcceptMove(sd, state, graph, params);
                ColorizeSeam(sd.csh, vcg::Color4b(255, 69, 0, 255));
                accept++;
                LOG_DEBUG << "Accepted operation";
                if (k <= 100) LOG_INFO << "  [Iter " << k << "] ACCEPTED";
            } else {
                RejectMove(sd, state, graph, status);
                reject++;
                LOG_DEBUG << "Rejected operation";
                if (k <= 100) LOG_INFO << "  [Iter " << k << "] REJECTED (status: " << status << ")";
            }

            if (iterTimer.TimeElapsed() > 5.0) {
                LOG_WARN << "  [Iter " << k << "] Long iteration detected: " << std::fixed << std::setprecision(2) << iterTimer.TimeElapsed() << "s "
                         << "(charts " << sd.a->id << " and " << sd.b->id << ", "
                         << sd.optimizationArea.size() << " faces)";
            }
        }
    }

    PrintStateInfo(state, graph, params);

    LogExecutionStats();

    Mesh shell;

    LOG_INFO << "Atlas energy after optimization is " << ARAP::ComputeEnergyFromStoredWedgeTC(graph->mesh, nullptr, nullptr);

}

void Finalize(GraphHandle graph, const std::string& outname, int *vndup)
{
    std::unordered_set<Mesh::ConstVertexPointer> vset;
    for (const MeshFace& f : graph->mesh.face)
        for (int i = 0; i < 3; ++i)
            vset.insert(f.cV(i));

    *vndup = (int) vset.size();

    // The following two calls are probably not necessary as at this point the mesh has been cut along texture seams
    // and there should not be any duplicate vertex. In any case, it is safer to leave them here.
    tri::Clean<Mesh>::RemoveDuplicateVertex(graph->mesh);
    tri::Clean<Mesh>::RemoveUnreferencedVertex(graph->mesh);
    tri::UpdateTopology<Mesh>::VertexFace(graph->mesh);
}

// -- static functions ---------------------------------------------------------

static void InsertNewClusterInQueue(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph, const AlgoParameters& params)
{
    ColorizeSeam(csh, vcg::Color4b::White);

    CostInfo ci = ComputeCost(csh, graph, params, GetPenalty(csh, state));

    if (params.reduce) {
        while (ci.mvalue == CostInfo::UNFEASIBLE_MATCHING) {
            ci = ReduceSeam(csh, state, graph, params);
        }
    }

    ColorizeSeam(csh, mvColor[ci.mvalue]);

    feasibility[ci.mvalue]++;

    if (ci.cost != Infinity()) {
        mincost = std::min(mincost, ci.cost);
        maxcost = std::max(maxcost, ci.cost);
    }

    state->queue.push(std::make_pair(csh, ci.cost));
    state->cost[csh] = ci.cost;
    state->transform[csh] = ci.matching;
    state->status[csh] = UNKNOWN;
    state->mvalue[csh] = ci.mvalue;

    // initialize chart-to-seam map
    ChartPair p = GetCharts(csh, graph);
    state->chartSeamMap[p.first->id].insert(csh);
    state->chartSeamMap[p.second->id].insert(csh);

    // add cluster to endpoint map
    std::set<int> endpoints = GetEndpoints(csh);
    for (auto vi : endpoints)
        state->emap[vi].insert(csh);
}

// this function returns true if there exists a sequence of at most maxSteps operations
// that results in a UV-island being formed in the graph AND involves a and b
static bool IslandLookahead(ChartHandle a, ChartHandle b, int maxSteps)
{
    // if there are too many candidates, exit early
    if (a->adj.size() > (unsigned) maxSteps || b->adj.size() > (unsigned) maxSteps)
        return false;

    std::unordered_set<ChartHandle> nab;
    nab.insert(a->adj.begin(), a->adj.end());
    nab.insert(b->adj.begin(), b->adj.end());

    for (auto c : nab) {
        std::stack<ChartHandle> s;
        s.push(a);

        std::set<ChartHandle> visited = {c, a}; // prevent the visit from reaching c
        int steps = 0;

        // start the visit
        while (!s.empty()) {
            ChartHandle sc = s.top();
            s.pop();
            visited.insert(sc);
            for (auto ac : sc->adj) {
                if (visited.find(ac) == visited.end()) {
                    s.push(ac);
                    steps++;
                }
                if (steps > maxSteps)
                    break;
            }
            if (steps > maxSteps)
                break;
        }

        // if the stack is empty we could not advance the visit
        // this means that the visited component would only be adjacent to c
        //   => the visited component is an island whose only adjacency is c
        if (s.empty())
            return true;
    }

    return false;
}

static CostInfo ComputeCost(ClusteredSeamHandle csh, GraphHandle graph, const AlgoParameters& params, double penalty)
{
    bool swapped;
    ChartPair charts = GetCharts(csh, graph, &swapped);

    ChartHandle a = charts.first;
    ChartHandle b = charts.second;
    double aUV = a->AreaUV();
    double bUV = b->AreaUV();
    double a3D = a->Area3D();
    double b3D = b->Area3D();
    if (!std::isfinite(aUV) || !std::isfinite(bUV) || aUV <= 0 || bUV <= 0 || a3D <= 0 || b3D <= 0) {
        return { Infinity(), {}, CostInfo::ZERO_AREA };
    }

    std::vector<vcg::Point2d> bpa;
    std::vector<vcg::Point2d> bpb;

    ExtractUVCoordinates(csh, bpa, bpb, {a->id});

    MatchingTransform mi = MatchingTransform::Identity();
    // if seam is disconnecting compute the actual matching
    if (a != b)
        mi = ComputeMatchingRigidMatrix(bpa, bpb);

    std::map<RegionID, double> bmap;
    double seamLength3D = 0;

    int ne = 0;
    SeamMesh& seamMesh = csh->sm;
    for (SeamHandle sh : csh->seams) {
        for (int iedge : sh->edges) {
            SeamEdge& edge = seamMesh.edge[iedge];
            bmap[edge.fa->id] += (edge.fa->V0(edge.ea)->T().P() - edge.fa->V1(edge.ea)->T().P()).Norm();
            bmap[edge.fb->id] += (edge.fb->V0(edge.eb)->T().P() - edge.fb->V1(edge.eb)->T().P()).Norm();
            seamLength3D += (edge.fa->P0(edge.ea) - edge.fa->P1(edge.ea)).Norm();
            ne++;
        }
    }

    CostInfo ci;
    ci.matching = mi;
    ci.mvalue = CostInfo::FEASIBLE;

    if (a != b) {
        double maxSeamToBoundaryRatio = std::max(bmap[a->id] / a->BorderUV(), bmap[b->id] / b->BorderUV());
        if (maxSeamToBoundaryRatio < params.boundaryTolerance && (!params.visitComponents || !IslandLookahead(a, b, 5))) {
            ci.cost = Infinity();
            ci.mvalue = CostInfo::UNFEASIBLE_BOUNDARY;
            return ci;
        }
    }

    double totErr = MatchingErrorTotal(mi, bpa, bpb);
    double avgErr = totErr / (double) bpa.size();

    if (avgErr > params.matchingThreshold * ((bmap[a->id] + bmap[b->id]) / 2.0)) {
        ci.cost = Infinity();
        ci.mvalue = CostInfo::UNFEASIBLE_MATCHING;
        return ci;
    }

    double lossgain = avgErr * std::pow(std::min(a->BorderUV() / bmap[a->id], b->BorderUV() / bmap[b->id]), params.expb);
    double sizebonus = std::min(a->AreaUV(), b->AreaUV());

    ci.cost = lossgain * sizebonus;

    if (ci.cost == 0 && penalty > 1.0)
        ci.cost = 1;

    ci.cost *= penalty;

    if (!std::isfinite(ci.cost))
        ci.cost = Infinity();

    return ci;
}

static inline double GetPenalty(ClusteredSeamHandle csh, AlgoStateHandle state)
{
    if (state->penalty.find(csh) == state->penalty.end())
        state->penalty[csh] = 1.0;
    return state->penalty[csh];
}

static inline bool Valid(const WeightedSeam& ws, ConstAlgoStateHandle state)
{
    auto it = state->cost.find(ws.first);
    return (it != state->cost.end() && it->second == ws.second);
}

static inline void PurgeQueue(AlgoStateHandle state)
{
    if (state->queue.empty()) return;
    LOG_INFO << "Purging queue of size " << state->queue.size() << " (cost map size is " << state->cost.size() << ")";
    decltype(state->queue) newQueue;
    for (const auto& costEntry : state->cost) {
        if (std::isfinite(costEntry.second)) {
            newQueue.push(std::make_pair(costEntry.first, costEntry.second));
        }
    }
    state->queue.swap(newQueue);
    LOG_INFO << "Queue purged, new size is " << state->queue.size();
}

static void ComputeSeamData(SeamData& sd, ClusteredSeamHandle csh, GraphHandle graph, AlgoStateHandle state)
{
    PERF_TIMER_START;

    sd.csh = csh;

    ChartPair charts = GetCharts(csh, graph);
    sd.a = charts.first;
    sd.b = charts.second;

    if (state->failed[sd.a->id].count(sd.b->id) > 0)
        num_retry++;

    Mesh& m = graph->mesh;

    sd.texcoorda.reserve(3 * sd.a->FN());
    sd.vertexinda.reserve(3 * sd.a->FN());
    for (auto fptr : sd.a->fpVec) {
        sd.texcoorda.push_back(fptr->V(0)->T().P());
        sd.texcoorda.push_back(fptr->V(1)->T().P());
        sd.texcoorda.push_back(fptr->V(2)->T().P());
        sd.vertexinda.push_back(tri::Index(m, fptr->V(0)));
        sd.vertexinda.push_back(tri::Index(m, fptr->V(1)));
        sd.vertexinda.push_back(tri::Index(m, fptr->V(2)));
    }

    if (sd.a != sd.b) {
        sd.texcoordb.reserve(3 * sd.b->FN());
        sd.vertexindb.reserve(3 * sd.b->FN());
        for (auto fptr : sd.b->fpVec) {
            sd.texcoordb.push_back(fptr->V(0)->T().P());
            sd.texcoordb.push_back(fptr->V(1)->T().P());
            sd.texcoordb.push_back(fptr->V(2)->T().P());
            sd.vertexindb.push_back(tri::Index(m, fptr->V(0)));
            sd.vertexindb.push_back(tri::Index(m, fptr->V(1)));
            sd.vertexindb.push_back(tri::Index(m, fptr->V(2)));
        }
    }

    sd.inputUVBorderLength = sd.a->BorderUV();
    if (sd.a != sd.b)
        sd.inputUVBorderLength += sd.b->BorderUV();

    PERF_TIMER_ACCUMULATE(t_seamdata);
}

static void WedgeTexFromVertexTex(ChartHandle c)
{
    for (auto fptr : c->fpVec)
        for (int i = 0; i < 3; ++i)
            fptr->WT(i).P() = fptr->V(i)->T().P();
}

static OffsetMap AlignAndMerge(ClusteredSeamHandle csh, SeamData& sd, const MatchingTransform& mi, const AlgoParameters& params)
{
    PERF_TIMER_START;

    OffsetMap om;

    // align
    if (sd.a != sd.b) {
        std::unordered_set<Mesh::VertexPointer> visited;
        for (auto fptr : sd.b->fpVec) {
            for (int i = 0; i < 3; ++i) {
                if (visited.count(fptr->V(i)) == 0) {
                    visited.insert(fptr->V(i));
                    fptr->V(i)->T().P() = mi.Apply(fptr->V(i)->T().P());
                }
            }
        }
    }

    // elect representative vertices for the merge (only 1 vert must be referenced after the merge)
    SeamMesh& seamMesh = csh->sm;
    for (SeamHandle sh : csh->seams) {
        for (int iedge : sh->edges) {
            SeamEdge& edge = seamMesh.edge[iedge];

            Mesh::VertexPointer v0a = edge.fa->V0(edge.ea);
            Mesh::VertexPointer v1a = edge.fa->V1(edge.ea);
            Mesh::VertexPointer v0b = edge.fb->V0(edge.eb);
            Mesh::VertexPointer v1b = edge.fb->V1(edge.eb);

            if (v0a->P() != edge.V(0)->P())
                std::swap(v0a, v1a);
            if (v0b->P() != edge.V(0)->P())
                std::swap(v0b, v1b);

            if (sd.mrep.count(v0a) == 0)
                sd.evec[edge.V(0)].push_back(v0a);
            sd.mrep[v0a] = sd.evec[edge.V(0)].front();

            if (sd.mrep.count(v1a) == 0)
                sd.evec[edge.V(1)].push_back(v1a);
            sd.mrep[v1a] = sd.evec[edge.V(1)].front();

            if (sd.mrep.count(v0b) == 0)
                sd.evec[edge.V(0)].push_back(v0b);
            sd.mrep[v0b] = sd.evec[edge.V(0)].front();

            if (sd.mrep.count(v1b) == 0)
                sd.evec[edge.V(1)].push_back(v1b);
            sd.mrep[v1b] = sd.evec[edge.V(1)].front();
        }
    }

    // for each vertex merged, store its original vfadj fan
    for (auto& entry : sd.mrep) {
        face::VFStarVF(entry.first, sd.vfmap[entry.first].first, sd.vfmap[entry.first].second);
    }

    // update vertex references
    for (auto fptr : sd.a->fpVec) {
        for (int i = 0; i < 3; ++i)
            if (sd.mrep.count(fptr->V(i)))
                fptr->V(i) = sd.mrep[fptr->V(i)];
    }
    if (sd.a != sd.b) {
        for (auto fptr : sd.b->fpVec) {
            for (int i = 0; i < 3; ++i)
                if (sd.mrep.count(fptr->V(i)))
                    fptr->V(i) = sd.mrep[fptr->V(i)];
        }
    }

    // update topologies. face-face is trivial, for vertex-face we
    // merge the VF lists

    // face-face
    for (SeamHandle sh : csh->seams) {
        for (int iedge : sh->edges) {
            SeamEdge& edge = seamMesh.edge[iedge];
            edge.fa->FFp(edge.ea) = edge.fb;
            edge.fa->FFi(edge.ea) = edge.eb;
            edge.fb->FFp(edge.eb) = edge.fa;
            edge.fb->FFi(edge.eb) = edge.ea;
        }
    }

    //tri::UpdateTopology<Mesh>::VertexFace(graph->mesh);

    // iterate over emap, and concatenate vf adjacencies
    // ugly... essentially we have a list of vfadj fans stored in vfmap,
    // and we concatenate the end of a fan with the beginning of the next
    // note that we only change the data stored in the faces (i.e. the list) and
    // never touch the data stored on the vertices
    for (auto& entry : sd.evec) {
        if (entry.second.size() > 1) {
            std::vector<Mesh::VertexPointer>& verts = entry.second;
            for (unsigned i = 0; i < entry.second.size() - 1; ++i) {
                sd.vfmap[verts[i]].first.back()->VFp(sd.vfmap[verts[i]].second.back()) = sd.vfmap[verts[i+1]].first.front();
                sd.vfmap[verts[i]].first.back()->VFi(sd.vfmap[verts[i]].second.back()) = sd.vfmap[verts[i+1]].second.front();
            }
        }
    }

    // compute average positions and displacements
    for (auto& entry : sd.evec) {
        vcg::Point2d sumpos = vcg::Point2d::Zero();
        for (auto vp : entry.second) {
            sumpos += vp->T().P();
        }
        vcg::Point2d avg = sumpos / (double) entry.second.size();

        double maxOffset = 0;
        for (auto& vp : entry.second)
            maxOffset = std::max(maxOffset, params.offsetFactor * (vp->T().P() - avg).Norm());

        om[entry.second.front()] = maxOffset;
        entry.second.front()->T().P() = avg;
    }

    PERF_TIMER_ACCUMULATE(t_alignmerge);
    return om;
}

static void ComputeOptimizationArea(SeamData& sd, Mesh& mesh, OffsetMap& om)
{
    PERF_TIMER_START;

    std::vector<Mesh::FacePointer> fpvec;
    fpvec.insert(fpvec.end(), sd.a->fpVec.begin(), sd.a->fpVec.end());
    if (sd.a != sd.b)
        fpvec.insert(fpvec.end(), sd.b->fpVec.begin(), sd.b->fpVec.end());

    sd.verticesWithinThreshold = ComputeVerticesWithinOffsetThreshold(mesh, om, sd);
    sd.optimizationArea.clear();

    for (auto fptr : fpvec) {
        for (int i = 0; i < 3; ++i) {
            if (sd.verticesWithinThreshold.find(fptr->V(i)) != sd.verticesWithinThreshold.end()) {
                sd.optimizationArea.insert(fptr);
                break;
            }
        }
    }

    for (auto fptr : sd.optimizationArea) {
        sd.texcoordoptVert.push_back(fptr->V(0)->T().P());
        sd.texcoordoptVert.push_back(fptr->V(1)->T().P());
        sd.texcoordoptVert.push_back(fptr->V(2)->T().P());

        sd.texcoordoptWedge.push_back(fptr->WT(0).P());
        sd.texcoordoptWedge.push_back(fptr->WT(1).P());
        sd.texcoordoptWedge.push_back(fptr->WT(2).P());

    }

    {
        sd.inputNegativeArea = 0;
        sd.inputAbsoluteArea = 0;
        for (auto fptr : sd.optimizationArea) {
            vcg::Point2d uv0in = fptr->WT(0).P();
            vcg::Point2d uv1in = fptr->WT(1).P();
            vcg::Point2d uv2in = fptr->WT(2).P();

            double inputAreaUV = ((uv1in - uv0in) ^ (uv2in - uv0in)) / 2.0;
            if (inputAreaUV < 0)
                sd.inputNegativeArea += inputAreaUV;
            sd.inputAbsoluteArea += std::abs(inputAreaUV);
        }
    }

    PERF_TIMER_ACCUMULATE(t_optimization_area);
}

/* Visit vertices starting from the merged ones, subject to the distance budget
 * stored in the OffsetMap object. */
static std::unordered_set<Mesh::VertexPointer> ComputeVerticesWithinOffsetThreshold(Mesh& m, const OffsetMap& om, const SeamData& sd)
{
    // typedef for heap nodes
    typedef std::pair<Mesh::VertexPointer, double> VertexNode;

    // comparison operator for the max-heap
    auto cmp = [] (const VertexNode& v1, const VertexNode& v2) { return v1.second < v2.second; };

    std::unordered_set<Mesh::VertexPointer> vset;

    // distance budget map
    OffsetMap dist;
    // heap
    std::vector<VertexNode> h;

    for (const auto& entry : om) {
        h.push_back(std::make_pair(entry.first, entry.second));
        dist[entry.first] = entry.second;
    }

    std::make_heap(h.begin(), h.end(), cmp);

    while (!h.empty()) {
        std::pop_heap(h.begin(), h.end(), cmp);
        VertexNode node = h.back();
        h.pop_back();
        if (node.second == dist[node.first]) {
            std::vector<Mesh::FacePointer> faces;
            std::vector<int> indices;
            face::VFStarVF(node.first, faces, indices);
            for (unsigned i = 0; i < faces.size(); ++i) {
                if(faces[i]->id != sd.a->id && faces[i]->id != sd.b->id){
                    LOG_ERR << "issue at face " << tri::Index(m, faces[i]);
                }
                ensure(faces[i]->id == sd.a->id || faces[i]->id == sd.b->id);

                // if either neighboring vertex is seen with more spare distance,
                // update the distance map

                int e1 = indices[i];
                Mesh::VertexPointer v1 = faces[i]->V1(indices[i]);
                double d1 = dist[node.first] - EdgeLengthUV(*faces[i], e1);

                if (d1 >= 0 && (dist.find(v1) == dist.end() || dist[v1] < d1)) {
                    dist[v1] = d1;
                    h.push_back(std::make_pair(v1, d1));
                    std::push_heap(h.begin(), h.end(), cmp);
                }

                int e2 = (indices[i]+2)%3;
                Mesh::VertexPointer v2 = faces[i]->V2(indices[i]);
                double d2 = dist[node.first] - EdgeLengthUV(*faces[i], e2);

               if (d2 >= 0 && (dist.find(v2) == dist.end() || dist[v2] < d2)) {
                    dist[v2] = d2;
                    h.push_back(std::make_pair(v2, d2));
                    std::push_heap(h.begin(), h.end(), cmp);
                }
            }
        }
    }

    for (const auto& entry : dist)
        vset.insert(entry.first);

    LOG_DEBUG << "vset.size() == " << vset.size();

    return vset;
}

static std::vector<HalfEdge> ExtractHalfEdges(const std::vector<ChartHandle>& charts, const vcg::Box2d& box, bool internalOnly)
{
    std::vector<HalfEdge> hvec;
    for (auto ch : charts) {
        if (!ch->UVBox().Intersects(box))
            continue;
        for (auto fptr : ch->fpVec) {
            for (int i = 0; i < 3; ++i) {
                if ((!internalOnly || !face::IsBorder(*fptr, i)) && SegmentBoxIntersection(Segment(fptr->V0(i)->T().P(), fptr->V1(i)->T().P()), box))
                    hvec.push_back(HalfEdge{fptr, i});
            }
        }
    }
    return hvec;
}

static CheckStatus CheckBoundaryAfterAlignmentInner(SeamData& sd)
{
    ensure(sd.a != sd.b);

    // check if the borders of the fixed areas of a and b intersect each other
    std::vector<HalfEdge> aVec;
    for (auto fptr : sd.a->fpVec)
        if (sd.optimizationArea.find(fptr) == sd.optimizationArea.end())
            for (int i = 0; i < 3; ++i)
                if (face::IsBorder(*fptr, i) || (sd.optimizationArea.find(fptr->FFp(i)) != sd.optimizationArea.end()))
                    aVec.push_back(HalfEdge{fptr, i});

    std::vector<HalfEdge> bVec;
    for (auto fptr : sd.b->fpVec)
        if (sd.optimizationArea.find(fptr) == sd.optimizationArea.end())
            for (int i = 0; i < 3; ++i)
                if (face::IsBorder(*fptr, i) || (sd.optimizationArea.find(fptr->FFp(i)) != sd.optimizationArea.end()))
                    bVec.push_back(HalfEdge{fptr, i});

    if ((aVec.size() > 0) && (bVec.size() > 0)) {
        if (HasAnyCrossIntersection(aVec, bVec))
            return FAIL_GLOBAL_OVERLAP_BEFORE;
    }
#if 0
    vcg::Box2d box;
    for (auto fptr : sd.b->fpVec)
        for (int i = 0; i < 3; ++i)
            box.Add(fptr->V(i)->T().P());

    // also check if the edges of b overlap the edges of a (only check the edges inside the bbox of b)
    aVec = ExtractHalfEdges({sd.a}, box, false);
    bVec = ExtractHalfEdges({sd.b}, box, false);
    if ((aVec.size() > 0) && (bVec.size() > 0)) {
        if (HasAnyCrossIntersection(aVec, bVec))
            return FAIL_GLOBAL_OVERLAP_BEFORE;
    }
#endif

    return PASS;
}

static CheckStatus CheckBoundaryAfterAlignment(SeamData& sd)
{
    PERF_TIMER_START;
    LOG_DEBUG << "Running CheckBoundaryAfterAlignment()";
    CheckStatus status = CheckBoundaryAfterAlignmentInner(sd);
    PERF_TIMER_ACCUMULATE(t_check_before);
    return status;
}

static CheckStatus CheckAfterLocalOptimizationInner(SeamData& sd, AlgoStateHandle state, const AlgoParameters& params)
{
    double newArapVal = (state->arapNum + (sd.outputArapNum - sd.inputArapNum)) / state->arapDenom;
    if (newArapVal > params.globalDistortionThreshold) {
        LOG_DEBUG << "[DIAG] Rejecting move for charts " << sd.a->id
                  << (sd.a != sd.b ? ("/" + std::to_string(sd.b->id)) : "")
                  << " due to global ARAP energy " << newArapVal
                  << " > threshold " << params.globalDistortionThreshold;
        return FAIL_DISTORTION_GLOBAL;
    }

    double localDistortion = sd.outputArapNum / sd.outputArapDenom;
    if (localDistortion > params.distortionTolerance) {
        LOG_DEBUG << "[DIAG] Rejecting move for charts " << sd.a->id
                  << (sd.a != sd.b ? ("/" + std::to_string(sd.b->id)) : "")
                  << " due to local ARAP distortion " << localDistortion
                  << " > threshold " << params.distortionTolerance;
        return FAIL_DISTORTION_LOCAL;
    }

    // Check if the combined chart exceeds the physical texture limit (~32k)
    // QImage cannot handle dimensions larger than 32767 pixels. We reject any merge
    // that would create a chart exceeding this limit to prevent downstream packing failures.
    {
        vcg::Box2d fullBox;
        // Calculate the bounding box of the POTENTIAL new chart using the updated wedge coords
        for (auto fptr : sd.a->fpVec) {
            for (int k = 0; k < 3; ++k)
                fullBox.Add(fptr->WT(k).P());
        }
        if (sd.b != sd.a) {
            for (auto fptr : sd.b->fpVec) {
                for (int k = 0; k < 3; ++k)
                    fullBox.Add(fptr->WT(k).P());
            }
        }

        // Limit based on QImage (32767). We use 32000 to be safe and allow for padding.
        const double MAX_UV_DIM = 32000.0;
        if (fullBox.DimX() > MAX_UV_DIM || fullBox.DimY() > MAX_UV_DIM) {
            LOG_DEBUG << "[DIAG] Rejecting move for charts " << sd.a->id
                      << (sd.a != sd.b ? ("/" + std::to_string(sd.b->id)) : "")
                      << " due to global UV size explosion: " << fullBox.DimX() << "x" << fullBox.DimY()
                      << " (limit=" << MAX_UV_DIM << ")";
            return FAIL_DISTORTION_GLOBAL;
        }
    }

    /* Check if the folded area is too large */
    double outputNegativeArea = 0;
    double outputAbsoluteArea = 0;
    for (auto fptr : sd.optimizationArea) {
        double areaUV = AreaUV(*fptr);
        if (areaUV < 0)
            outputNegativeArea += areaUV;
        outputAbsoluteArea += std::abs(areaUV);
    }

    double inputRatio = std::abs(sd.inputNegativeArea / sd.inputAbsoluteArea);
    double outputRatio = std::abs(outputNegativeArea / outputAbsoluteArea);

    if (outputRatio > inputRatio) {
        LOG_DEBUG << "[DIAG] Rejecting move for charts " << sd.a->id
                  << (sd.a != sd.b ? ("/" + std::to_string(sd.b->id)) : "")
                  << " due to increased folded area ratio: input=" << inputRatio
                  << ", output=" << outputRatio;
        return FAIL_LOCAL_OVERLAP;
    }

    // Functions to detect if the half-edges have already been fixed (in which case detecting the intersection is meaningless,
    // the half-edges were intersecting to begin with)

    auto FixedPair = [&] (const HalfEdgePair& hep) -> bool {
        return /*hep.first.fp->id == hep.second.fp->id
                &&*/ sd.fixedVerticesFromIntersectingEdges.find(hep.first.V0()) != sd.fixedVerticesFromIntersectingEdges.end()
                && sd.fixedVerticesFromIntersectingEdges.find(hep.first.V1()) != sd.fixedVerticesFromIntersectingEdges.end()
                && sd.fixedVerticesFromIntersectingEdges.find(hep.second.V0()) != sd.fixedVerticesFromIntersectingEdges.end()
                && sd.fixedVerticesFromIntersectingEdges.find(hep.second.V1()) != sd.fixedVerticesFromIntersectingEdges.end();
    };

    auto FixedFirst = [&] (const HalfEdgePair& hep) -> bool {
        return /*hep.first.fp->id == hep.second.fp->id
                &&*/ sd.fixedVerticesFromIntersectingEdges.find(hep.first.V0()) != sd.fixedVerticesFromIntersectingEdges.end()
                && sd.fixedVerticesFromIntersectingEdges.find(hep.first.V1()) != sd.fixedVerticesFromIntersectingEdges.end();
    };

    // ensure the optimization border does not self-intersect
    std::vector<HalfEdge> sVec;
    for (auto fptr : sd.optimizationArea)
        for (int i = 0; i < 3; ++i)
            if (face::IsBorder(*fptr, i) || (sd.optimizationArea.find(fptr->FFp(i)) == sd.optimizationArea.end()))
                sVec.push_back(HalfEdge{fptr, i});

    if (sVec.size() > 0) {
        sd.intersectionOpt = Intersection(sVec);
        sd.intersectionOpt.erase(std::remove_if(sd.intersectionOpt.begin(), sd.intersectionOpt.end(), FixedPair), sd.intersectionOpt.end());
        if (sd.intersectionOpt.size() > 0) {
            return FAIL_GLOBAL_OVERLAP_AFTER_OPT;
        }
    }

    // also ensure the optimization border does not intersect the border of the fixed area
    // note that this check is not suficient, we should make sure that the optimization AREA
    // does not intersect with the non-optimized area. This check should be done either with
    // rasterization or triangle intersections
    std::vector<HalfEdge> nopVecBorder;
    for (auto fptr : sd.a->fpVec)
        if (sd.optimizationArea.find(fptr) == sd.optimizationArea.end())
            for (int i = 0; i < 3; ++i)
                if (face::IsBorder(*fptr, i) /* || (sd.optimizationArea.find(fptr->FFp(i)) != sd.optimizationArea.end()) */)
                    nopVecBorder.push_back(HalfEdge{fptr, i});

    if (sd.a != sd.b) {
        for (auto fptr : sd.b->fpVec)
            if (sd.optimizationArea.find(fptr) == sd.optimizationArea.end())
                for (int i = 0; i < 3; ++i)
                    if (face::IsBorder(*fptr, i) /* || (sd.optimizationArea.find(fptr->FFp(i)) != sd.optimizationArea.end()) */)
                        nopVecBorder.push_back(HalfEdge{fptr, i});
    }

    if (sVec.size() > 0 && nopVecBorder.size() > 0) {
        sd.intersectionBoundary = CrossIntersection(sVec, nopVecBorder);
        sd.intersectionBoundary.erase(std::remove_if(sd.intersectionBoundary.begin(), sd.intersectionBoundary.end(), FixedFirst), sd.intersectionBoundary.end());
        if (sd.intersectionBoundary.size() > 0) {
            return FAIL_GLOBAL_OVERLAP_AFTER_BND;
        }
    }

    // also ensure that the optimization border does not overlap any internal edge (inside or outside the optimization area)
    // to speed things up, only check edges that are inside the bbox of the opt area
    vcg::Box2d optBox;
    for (auto fptr : sd.optimizationArea)
        for (int i = 0; i < 3; ++i)
            optBox.Add(fptr->V(i)->T().P());

    std::vector<HalfEdge> internal = ExtractHalfEdges({sd.a, sd.b}, optBox, true); // internal only

    if (sVec.size() > 0 && internal.size() > 0) {
        sd.intersectionInternal = CrossIntersection(sVec, internal);
        sd.intersectionInternal.erase(std::remove_if(sd.intersectionInternal.begin(), sd.intersectionInternal.end(), FixedFirst), sd.intersectionInternal.end());
        if (sd.intersectionInternal.size() > 0) {
            return FAIL_GLOBAL_OVERLAP_AFTER_BND;
        }
    }

    vcg::Box2d box;
    for (auto fptr : sd.b->fpVec)
        for (int i = 0; i < 3; ++i)
            box.Add(fptr->V(i)->T().P());

    // also check if the edges of b overlap the edges of a (only check the edges inside the bbox of b)
    std::vector<HalfEdge> aVec = ExtractHalfEdges({sd.a}, box, false);
    std::vector<HalfEdge> bVec = ExtractHalfEdges({sd.b}, box, false);
    if ((aVec.size() > 0) && (bVec.size() > 0)) {
        if (HasAnyCrossIntersection(aVec, bVec))
            return FAIL_GLOBAL_OVERLAP_UNFIXABLE;
    }

    return PASS;
}

static CheckStatus CheckAfterLocalOptimization(SeamData& sd, AlgoStateHandle state, const AlgoParameters& params)
{
    PERF_TIMER_START;
    LOG_DEBUG << "Running CheckAfterLocalOptimization()";
    CheckStatus status = CheckAfterLocalOptimizationInner(sd, state, params);
    PERF_TIMER_ACCUMULATE(t_check_after);
    return status;
}

static CheckStatus OptimizeChart(SeamData& sd, GraphHandle graph, bool fixIntersectingEdges)
{
    PERF_TIMER_START;

    // Create a support face group that contains only the faces that must be
    // optimized with arap, i.e. the faces that have at least one vertex in vset
    // also initialize the tex coords to the stored tex coords
    // (needed if this is called more than once)
    auto itV = sd.texcoordoptVert.begin();
    auto itW = sd.texcoordoptWedge.begin();
    FaceGroup support(graph->mesh, INVALID_ID);
    for (auto fptr : sd.optimizationArea) {
        support.AddFace(fptr);
        fptr->V(0)->T().P() = *itV++; fptr->WT(0).P() = *itW++;
        fptr->V(1)->T().P() = *itV++; fptr->WT(1).P() = *itW++;
        fptr->V(2)->T().P() = *itV++; fptr->WT(2).P() = *itW++;
    }

    // [UV SCALE GUARD] Measure UV bounding box before ARAP optimization
    vcg::Box2d optBoxBefore;
    for (auto fptr : sd.optimizationArea) {
        for (int i = 0; i < 3; ++i) {
            optBoxBefore.Add(fptr->WT(i).P());
        }
    }
    double scaleBefore = std::max(optBoxBefore.DimX(), optBoxBefore.DimY());
    if (scaleBefore <= 0 || !std::isfinite(scaleBefore)) {
        scaleBefore = 1.0; // avoid div-by-zero
    }

    // before updating the wedge tex coords, compute the arap contribution of the optimization area
    // WARNING: it is critial that at this point the wedge tex coords HAVE NOT YET BEEN UPDATED
    ARAP::ComputeEnergyFromStoredWedgeTC(support.fpVec, graph->mesh, &sd.inputArapNum, &sd.inputArapDenom);

    WedgeTexFromVertexTex(sd.a);
    if (sd.a != sd.b)
        WedgeTexFromVertexTex(sd.b);

    LOG_DEBUG << "Building shell...";

    sd.shell.Clear();
    sd.shell.ClearAttributes();
    bool singleComponent = BuildShellWithTargetsFromUV(sd.shell, support, 1.0);

    if (!singleComponent)
        LOG_DEBUG << "Shell is not single component";

    // Use the existing texture coordinates as starting point for ARAP

    // Copy the wedge texcoords
    for (unsigned i = 0; i < support.FN(); ++i) {
        auto& sf = sd.shell.face[i];
        auto& f = *(support.fpVec[i]);
        for (int j = 0; j < 3; ++j) {
            sf.WT(j) = f.V(j)->T();
            sf.V(j)->T() = sf.WT(j);
        }
    }

    // split non manifold vertices
    while (tri::Clean<Mesh>::SplitNonManifoldVertex(sd.shell, 0.3))
        ;

    CutAlongSeams(sd.shell);

    int nholes = tri::Clean<Mesh>::CountHoles(sd.shell);
    int genus = tri::Clean<Mesh>::MeshGenus(sd.shell);
    if (nholes == 0 || genus != 0) {
        return FAIL_TOPOLOGY;
    }

    if (singleComponent && nholes > 1)
        CloseHoles3D(sd.shell);

    SyncShellWithUV(sd.shell);

    PERF_TIMER_ACCUMULATE(t_optimize_build);

    LOG_DEBUG << "Optimizing...";
    ARAP arap(sd.shell);
    arap.SetMaxIterations(100);

    // select the vertices, using the fact that the faces are mirrored in
    // the support object

    for (unsigned i = 0; i < support.FN(); ++i) {
        for (int j = 0; j < 3; ++j) {
            if (sd.verticesWithinThreshold.find(support.fpVec[i]->V(j)) == sd.verticesWithinThreshold.end()) {
                ensure(sd.shell.face[i].IsHoleFilling() == false);
                sd.shell.face[i].V(j)->SetS();
            }
        }
    }

    if (fixIntersectingEdges) {
        unsigned fixedBefore = sd.fixedVerticesFromIntersectingEdges.size();
        for (auto hep : sd.intersectionOpt) {
            sd.fixedVerticesFromIntersectingEdges.insert(hep.first.fp->V0(hep.first.e));
            sd.fixedVerticesFromIntersectingEdges.insert(hep.first.fp->V1(hep.first.e));
            sd.fixedVerticesFromIntersectingEdges.insert(hep.second.fp->V0(hep.second.e));
            sd.fixedVerticesFromIntersectingEdges.insert(hep.second.fp->V1(hep.second.e));
        }
        for (auto hep : sd.intersectionBoundary) {
            HalfEdge he = hep.first;
            sd.fixedVerticesFromIntersectingEdges.insert(he.fp->V0(he.e));
            sd.fixedVerticesFromIntersectingEdges.insert(he.fp->V1(he.e));
        }
        for (auto hep : sd.intersectionInternal) {
            HalfEdge he = hep.first;
            sd.fixedVerticesFromIntersectingEdges.insert(he.fp->V0(he.e));
            sd.fixedVerticesFromIntersectingEdges.insert(he.fp->V1(he.e));
        }
        if (fixedBefore == sd.fixedVerticesFromIntersectingEdges.size())
            return _END;

        for (unsigned i = 0; i < support.FN(); ++i) {
            for (int j = 0; j < 3; ++j) {
                if (sd.fixedVerticesFromIntersectingEdges.find(support.fpVec[i]->V(j)) != sd.fixedVerticesFromIntersectingEdges.end())
                    sd.shell.face[i].V(j)->SetS();
            }
        }
    }

    // Fix the selected vertices of the shell;
    int nfixed = arap.FixSelectedVertices();
    LOG_DEBUG << "Fixed " << nfixed << " vertices";
    double tol = 0.02;
    while (nfixed < 2) {
        LOG_DEBUG << "Not enough selected vertices found, fixing random edge with tolerance " << tol;
        nfixed += arap.FixRandomEdgeWithinTolerance(tol);
        tol += 0.02;
    }
    ensure(nfixed > 0);

    LOG_DEBUG << "Solving...";
    sd.si = arap.Solve();

    PERF_TIMER_ACCUMULATE_FROM_PREVIOUS(t_optimize_arap);

    // [UV SCALE GUARD] Measure UV bounding box after ARAP optimization (on shell)
    vcg::Box2d optBoxAfter;
    for (auto& sf : sd.shell.face) {
        if (sf.IsHoleFilling()) continue;
        for (int i = 0; i < 3; ++i) {
            optBoxAfter.Add(sf.V(i)->T().P());
        }
    }
    double scaleAfter = std::max(optBoxAfter.DimX(), optBoxAfter.DimY());
    double scaleRatio = (scaleBefore > 0) ? scaleAfter / scaleBefore : 1.0;

    LOG_DEBUG << "[ARAP] Opt area bbox before: " << optBoxBefore.DimX() << "x" << optBoxBefore.DimY()
              << ", after: " << optBoxAfter.DimX() << "x" << optBoxAfter.DimY()
              << ", scaleRatio: " << scaleRatio;

    // Hard guard: reject if ARAP blows up the local UV scale beyond a reasonable factor
    const double MAX_LOCAL_SCALE_RATIO = 50.0;
    if (!std::isfinite(scaleRatio) || scaleRatio > MAX_LOCAL_SCALE_RATIO) {
        LOG_WARN << "[VALIDATION] ARAP produced extreme UV scale explosion (ratio=" << scaleRatio
                 << ") for charts " << sd.a->id
                 << (sd.a != sd.b ? ("/" + std::to_string(sd.b->id)) : "")
                 << ". Rejecting move to prevent packing failure.";
        LogArapExplosionDiagnostics(sd.shell);
        return FAIL_DISTORTION_LOCAL;
    }

    SyncShellWithUV(sd.shell);

    LOG_DEBUG << "Syncing chart...";

    ensure(HasFaceIndexAttribute(sd.shell));
    auto ia = GetFaceIndexAttribute(sd.shell);
    for (auto& sf : sd.shell.face) {
        if (!sf.IsHoleFilling()) {
            auto& f = (graph->mesh).face[ia[sf]];
            for (int k = 0; k < 3; ++k) {
                f.WT(k).P() = sf.V(k)->T().P();
                f.V(k)->T().P() = sf.V(k)->T().P();
            }
        }
    }

    if (!sd.si.numericalError)
        ARAP::ComputeEnergyFromStoredWedgeTC(support.fpVec, graph->mesh, &sd.outputArapNum, &sd.outputArapDenom);

    PERF_TIMER_ACCUMULATE(t_optimize);

    return sd.si.numericalError ? FAIL_NUMERICAL_ERROR : PASS;
}

static bool SeamInterceptsOptimizationArea(ClusteredSeamHandle csh, const SeamData& sd)
{
    const SeamMesh& sm = csh->sm;
    for (auto sh : csh->seams) {
        for (int i : sh->edges) {
            const SeamEdge& edge = sm.edge[i];
            if ((sd.optimizationArea.find(edge.fa) != sd.optimizationArea.end()) || (sd.optimizationArea.find(edge.fb) != sd.optimizationArea.end()))
                return true;
        }
    }
    return false;
}

static void AcceptMove(const SeamData& sd, AlgoStateHandle state, GraphHandle graph, const AlgoParameters& params)
{
    PERF_TIMER_START;

    if (min_energy > sd.si.finalEnergy)
        min_energy = sd.si.finalEnergy;
    if (max_energy < sd.si.finalEnergy)
        max_energy = sd.si.finalEnergy;

    state->changeSet.insert(sd.optimizationArea.begin(), sd.optimizationArea.end());

    std::vector<SeamHandle> shared;
    std::set<ClusteredSeamHandle> sharedClusters; // clusters that can be aggregated after the merge
    std::set<ClusteredSeamHandle> independentClusters; // clusters not directly impacted by the merge

    std::set<ClusteredSeamHandle> selfClusters;

    if (sd.a != sd.b) {
        // ``disjoint'' seams, i.e. seams between B and C with C not in N(a)
        // are inherited by A
        for (auto csh : state->chartSeamMap[sd.b->id]) {
            ChartPair p = GetCharts(csh, graph);
            ChartHandle c = (p.first == sd.b) ? p.second : p.first;
            if (c == sd.a || c == sd.b) {
                selfClusters.insert(csh);
            } else if (sd.a->adj.find(c) == sd.a->adj.end()) {
                independentClusters.insert(csh);
            } else {
                ensure(c->adj.find(sd.a) != c->adj.end());
                ensure(c->adj.find(sd.b) != c->adj.end());
                ensure(sharedClusters.count(csh) == 0);
                sharedClusters.insert(csh);
                for (auto sh : csh->seams)
                    shared.push_back(sh);
            }
        }

        // we also need to recompute the cost of seams between A and C with C not in N(b)
        for (auto csh : state->chartSeamMap[sd.a->id]) {
            ChartPair p = GetCharts(csh, graph);
            ChartHandle c = (p.first == sd.a) ? p.second : p.first;
            if (c == sd.a || c == sd.b) {
                selfClusters.insert(csh);
            } else if (sd.b->adj.find(c) == sd.b->adj.end()) {
                independentClusters.insert(csh);
            } else {
                ensure(c->adj.find(sd.a) != c->adj.end());
                ensure(c->adj.find(sd.b) != c->adj.end());
                ensure(sharedClusters.count(csh) == 0);
                sharedClusters.insert(csh);
                for (auto sh : csh->seams)
                    shared.push_back(sh);
            }
        }

        /*
        for (auto x : std::set<ChartHandle>{sd.a, sd.b}) {
            for (auto csh : state->chartSeamMap[x->id]) {
                ChartPair p = GetCharts(csh, graph);
                ChartHandle c = (p.first == x) ? p.second : p.first;
                if ((sd.a->adj.find(c) != sd.a->adj.end()) && (sd.b->adj.find(c) != sd.b->adj.end())) {
                    ensure(c != sd.a);
                    ensure(c != sd.b);
                    ensure(sharedClusters.count(csh) == 0);
                    sharedClusters.insert(csh);
                    for (auto sh : csh->seams)
                        shared.push_back(sh);
                }
            }
        }
        */

        // update the MeshGraph object
        for (auto fptr : sd.b->fpVec)
            fptr->id = sd.a->Fp()->id;
        sd.a->fpVec.insert(sd.a->fpVec.end(), sd.b->fpVec.begin(), sd.b->fpVec.end());

        sd.a->adj.erase(sd.b);
        for (auto c : sd.b->adj) {
            if (c != sd.a) { // chart a is now (if it wasn't already) adjacent to c
                c->adj.erase(sd.b);
                c->adj.insert(sd.a);
                sd.a->adj.insert(c);
            }
        }
        graph->charts.erase(sd.b->id);

        // update state
        state->chartSeamMap.erase(sd.b->id);
        std::set<RegionID>& failed_b = state->failed[sd.b->id];
        state->failed[sd.a->id].insert(failed_b.begin(), failed_b.end());
        state->failed.erase(sd.b->id);
    } else {
        // if removing a non-disconnecting seam then all the clusters are independent
        independentClusters.insert(state->chartSeamMap[sd.b->id].begin(), state->chartSeamMap[sd.b->id].end());
        independentClusters.erase(sd.csh);
    }

    // invalidate cache
    sd.a->ParameterizationChanged();

    // update current UV border length
    double deltaUVBorderLength = sd.a->BorderUV() - sd.inputUVBorderLength;
    state->currentUVBorderLength += deltaUVBorderLength;

    // update atlas energy
    state->arapNum += (sd.outputArapNum - sd.inputArapNum);
    state->arapDenom += (sd.outputArapDenom - sd.inputArapDenom);

    if (state->failed[sd.a->id].count(sd.b->id) > 0)
        retry_success++;

    // Erase seam
    EraseSeam(sd.csh, state, graph);
    state->penalty.erase(sd.csh);

    for (auto csh : independentClusters) {
        auto it = state->status.find(csh);
        ensure(it != state->status.end());

        CheckStatus clusterStatus = it->second;
        ensure(clusterStatus != PASS);

        CostInfo::MatchingValue mv = state->mvalue[csh];

        EraseSeam(csh, state, graph);

        bool invalidate = (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_BEFORE)
                || (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_AFTER_OPT)
                || (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_AFTER_BND)
                || (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_UNFIXABLE && !SeamInterceptsOptimizationArea(csh, sd))
                || (clusterStatus == CheckStatus::FAIL_TOPOLOGY);

        if (invalidate || (params.ignoreOnReject && mv == CostInfo::REJECTED))
            InvalidateCluster(csh, state, graph, clusterStatus, 1.0);
        else
            InsertNewClusterInQueue(csh, state, graph, params);
    }

    for (auto csh : sharedClusters)
        EraseSeam(csh, state, graph);

    std::vector<ClusteredSeamHandle> cshvec = ClusterSeamsByChartId(shared);
    for (auto csh : cshvec) {
        InsertNewClusterInQueue(csh, state, graph, params);
    }

    if (params.visitComponents) {
        // if potential islands are allowed to ignore the boundary length limit,
        // then check if any chart adjacent to a or b becomes a potential island
        // after the merge and activate the corresponding seams below threshold

        std::set<ClusteredSeamHandle> unfeasibleBoundaryAdj;
        for (ChartHandle c : sd.a->adj)
            for (auto csh : state->chartSeamMap[c->id])
                if (state->mvalue[csh] == CostInfo::MatchingValue::UNFEASIBLE_BOUNDARY)
                    unfeasibleBoundaryAdj.insert(csh);

        for (ClusteredSeamHandle csh : unfeasibleBoundaryAdj) {
            EraseSeam(csh, state, graph);
            InsertNewClusterInQueue(csh, state, graph, params);
        }
    }

    PERF_TIMER_ACCUMULATE(t_accept);
}

static void RejectMove(const SeamData& sd, AlgoStateHandle state, GraphHandle graph, CheckStatus status)
{
    PERF_TIMER_START;

    Mesh& m = graph->mesh;

    // restore texture coordinates and indices
    RestoreChartAttributes(sd.a, m, sd.vertexinda.begin(), sd.texcoorda.begin());
    if (sd.a != sd.b)
        RestoreChartAttributes(sd.b, m, sd.vertexindb.begin(), sd.texcoordb.begin());

    // restore face-face topology
    SeamMesh& seamMesh = sd.csh->sm;
    for (SeamHandle sh : sd.csh->seams) {
        for (int iedge : sh->edges) {
            const SeamEdge& edge = seamMesh.edge[iedge];
            edge.fa->FFp(edge.ea) = edge.fa;
            edge.fa->FFi(edge.ea) = edge.ea;
            edge.fb->FFp(edge.eb) = edge.fb;
            edge.fb->FFi(edge.eb) = edge.eb;
        }
    }

    // restore vertex-face topology
    // iterate over emap, and split the lists according to the original topology
    // recall that we never touched any vertex topology attribute
    for (auto& entry : sd.evec) {
        //ensure(entry.second.size() > 1); not true for self seams at the cone vertex
        const std::vector<Mesh::VertexPointer>& verts = entry.second;
        for (unsigned i = 0; i < entry.second.size() - 1; ++i) {
            sd.vfmap.at(verts[i]).first.back()->VFp(sd.vfmap.at(verts[i]).second.back()) = 0;
            sd.vfmap.at(verts[i]).first.back()->VFi(sd.vfmap.at(verts[i]).second.back()) = 0;
        }
    }

    EraseSeam(sd.csh, state, graph);

    InvalidateCluster(sd.csh, state, graph, status, PENALTY_MULTIPLIER);
    if (sd.a != sd.b)
        state->failed[sd.a->id].insert(sd.b->id);

    PERF_TIMER_ACCUMULATE(t_reject);
}

static void EraseSeam(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph)
{
    ensure(csh->size() > 0);

    std::size_t n = state->cost.erase(csh);
    ensure(n > 0);

    n = state->transform.erase(csh);
    ensure(n > 0);

    n = state->status.erase(csh);
    ensure(n > 0);

    n = state->mvalue.erase(csh);
    ensure(n > 0);

    ChartPair charts = GetCharts(csh, graph);

    // the following check are needed because AcceptMove() may erase seams after
    // fiddling with the chartSeamMap...
    if (state->chartSeamMap.find(charts.first->id) != state->chartSeamMap.end())
        state->chartSeamMap[charts.first->id].erase(csh);

    if (state->chartSeamMap.find(charts.second->id) != state->chartSeamMap.end())
        state->chartSeamMap[charts.second->id].erase(csh);

    // erase seam from endpoint map
    std::set<int> endpoints = GetEndpoints(csh);
    for (auto vi : endpoints) {
        unsigned n = state->emap[vi].erase(csh);
        ensure(n > 0);
    }
}

static void InvalidateCluster(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph, CheckStatus status, double penaltyMultiplier)
{
    ColorizeSeam(csh, statusColor[status]);

    CostInfo ci;
    ci.cost = Infinity();
    ci.mvalue = CostInfo::REJECTED;
    ci.matching = MatchingTransform::Identity();

    state->queue.push(std::make_pair(csh, ci.cost));
    state->cost[csh] = Infinity();
    state->transform[csh] = ci.matching;
    state->status[csh] = status;
    state->mvalue[csh] = ci.mvalue;

    ChartPair p = GetCharts(csh, graph);
    state->chartSeamMap[p.first->id].insert(csh);
    state->chartSeamMap[p.second->id].insert(csh);

    // add penalty if the cluster is later re-evaluated
    double penalty = GetPenalty(csh, state);
    state->penalty[csh] = penalty * penaltyMultiplier;

    // add cluster to endpoint map
    std::set<int> endpoints = GetEndpoints(csh);
    for (auto vi : endpoints)
        state->emap[vi].insert(csh);
}

static void RestoreChartAttributes(ChartHandle c, Mesh& m, std::vector<int>::const_iterator itvi,  std::vector<vcg::Point2d>::const_iterator ittc)
{
    for (auto fptr : c->fpVec) {
        for (int i = 0; i < 3; ++i) {
            fptr->V(i) = &m.vert[*itvi++];
            fptr->V(i)->T().P() = *ittc;
            fptr->WT(i).P() = *ittc++;
        }
    }
}

static CostInfo ReduceSeam(ClusteredSeamHandle csh, AlgoStateHandle state, GraphHandle graph, const AlgoParameters& params)
{
    ClusteredSeamHandle reduced = nullptr;

    double totlen = ComputeSeamLength3D(csh);

    SeamMesh& sm = csh->sm;

    // reduce ``forward''
    ClusteredSeamHandle fwd, bwd;
    {
        fwd = std::make_shared<ClusteredSeam>(sm);
        double lenfwd = 0;
        auto seamHandleIt = csh->seams.begin();
        while (lenfwd < params.reductionFactor * totlen && seamHandleIt != csh->seams.end()) {
            SeamHandle sh  = *seamHandleIt;
            SeamHandle shnew = std::make_shared<Seam>(sm);

            std::map<SeamMesh::VertexPointer, int> visited;
            for (int e : sh->edges) {
                if (lenfwd >= params.reductionFactor * totlen)
                    break;
                shnew->edges.push_back(e);
                visited[sm.edge[e].V(0)]++;
                visited[sm.edge[e].V(1)]++;
                lenfwd += (sm.edge[e].P(0) - sm.edge[e].P(1)).Norm();
            }

            if (shnew->edges.size() == sh->edges.size()) {
                shnew->endpoints = sh->endpoints;
            } else {
                for (auto& entry : visited) {
                    if (entry.second == 1) {
                        shnew->endpoints.push_back(tri::Index(sm, entry.first));
                    }
                }
                if (tri::Index(sm, sm.edge[shnew->edges.front()].V(0)) != (unsigned) shnew->endpoints.front()
                        && tri::Index(sm, sm.edge[shnew->edges.front()].V(1)) != (unsigned) shnew->endpoints.front()) {
                    std::reverse(shnew->endpoints.begin(), shnew->endpoints.end());
                }
            }

            fwd->seams.push_back(shnew);
        }
    }

    // reduce ``backward''
    {
        bwd = std::make_shared<ClusteredSeam>(sm);
        double lenbwd = 0;
        auto seamHandleIt = csh->seams.rbegin();
        while (lenbwd < params.reductionFactor * totlen && seamHandleIt != csh->seams.rend()) {
            SeamHandle sh  = *seamHandleIt;
            SeamHandle shnew = std::make_shared<Seam>(sm);

            std::map<SeamMesh::VertexPointer, int> visited;
            for (auto ei = sh->edges.rbegin(); ei != sh->edges.rend(); ++ei) {
                if (lenbwd >= params.reductionFactor * totlen)
                    break;
                shnew->edges.push_back(*ei);
                visited[sm.edge[*ei].V(0)]++;
                visited[sm.edge[*ei].V(1)]++;
                lenbwd += (sm.edge[*ei].P(0) - sm.edge[*ei].P(1)).Norm();
            }
            std::reverse(shnew->edges.begin(), shnew->edges.end());

            if (shnew->edges.size() == sh->edges.size()) {
                shnew->endpoints = sh->endpoints;
            } else {
                for (auto& entry : visited) {
                    if (entry.second == 1) {
                        shnew->endpoints.push_back(tri::Index(sm, entry.first));
                    }
                }
                if (tri::Index(sm, sm.edge[shnew->edges.front()].V(0)) != (unsigned) shnew->endpoints.front()
                        && tri::Index(sm, sm.edge[shnew->edges.front()].V(1)) != (unsigned) shnew->endpoints.front()) {
                    std::reverse(shnew->endpoints.begin(), shnew->endpoints.end());
                }
            }

            bwd->seams.push_back(shnew);
        }
        std::reverse(bwd->seams.begin(), bwd->seams.end());
    }

    CostInfo cfwd = ComputeCost(fwd, graph, params, GetPenalty(csh, state));
    CostInfo cbwd = ComputeCost(bwd, graph, params, GetPenalty(csh, state));

    if (cfwd.cost < cbwd.cost) {
        csh->seams = fwd->seams;
        return cfwd;
    } else {
        csh->seams = bwd->seams;
        return cbwd;
    }
}

// PARALLEL OPTIMIZATION IMPLEMENTATION

// Global parallel configuration
static ParallelConfig g_parallelConfig;
static std::mutex g_configMutex;

void SetParallelConfig(const ParallelConfig& config)
{
    std::lock_guard<std::mutex> lock(g_configMutex);
    g_parallelConfig = config;
}

ParallelConfig GetParallelConfig()
{
    std::lock_guard<std::mutex> lock(g_configMutex);
    return g_parallelConfig;
}

// Forward declarations for parallel helpers
static TopologyDiff AlignAndMerge_Virtual(ClusteredSeamHandle csh, const ChartHandle a, const ChartHandle b,
                                          const MatchingTransform& mi, const AlgoParameters& params);
static void ComputeOptimizationArea_Virtual(MergeJobResult& result, const TopologyDiff& diff,
                                            const ChartHandle a, const ChartHandle b);
static CheckStatus CheckBoundary_Virtual(const MergeJobResult& result, const TopologyDiff& diff,
                                         const ChartHandle a, const ChartHandle b);
static CheckStatus CheckAfterOptimization_Virtual(MergeJobResult& result, const TopologyDiff& diff,
                                                  double globalArapNum, double globalArapDenom,
                                                  const AlgoParameters& params);
static void ApplyMergeAndUpdateQueue(const MergeJobResult& result, AlgoStateHandle state, GraphHandle graph,
                                     const AlgoParameters& params);
static bool CheckBoundingBoxCollision(const vcg::Box2d& box1, const vcg::Box2d& box2);
static double ComputeArapEnergyFromVirtualPositions(const std::vector<Mesh::FacePointer>& faces, Mesh& mesh,
                                                     const std::unordered_map<Mesh::VertexPointer, vcg::Point2d>& virtualUV,
                                                     double* num, double* denom);

// AlignAndMerge_Virtual: Compute merge WITHOUT modifying global mesh
// Returns a TopologyDiff describing the merge result
static TopologyDiff AlignAndMerge_Virtual(ClusteredSeamHandle csh, const ChartHandle a, const ChartHandle b,
                                          const MatchingTransform& mi, const AlgoParameters& params)
{
    TopologyDiff diff;
    diff.transform = mi;

    // Build a temporary UV position map: start with current positions
    std::unordered_map<Mesh::VertexPointer, vcg::Point2d> virtualUV;

    // Initialize virtual UV positions from current vertex positions
    for (auto fptr : a->fpVec) {
        for (int i = 0; i < 3; ++i) {
            virtualUV[fptr->V(i)] = fptr->V(i)->T().P();
        }
    }
    if (a != b) {
        // Apply transform to chart B vertices virtually
        for (auto fptr : b->fpVec) {
            for (int i = 0; i < 3; ++i) {
                virtualUV[fptr->V(i)] = mi.Apply(fptr->V(i)->T().P());
            }
        }
    }

    // Elect representative vertices for the merge (only 1 vert must be referenced after the merge)
    SeamMesh& seamMesh = csh->sm;
    for (SeamHandle sh : csh->seams) {
        for (int iedge : sh->edges) {
            SeamEdge& edge = seamMesh.edge[iedge];

            Mesh::VertexPointer v0a = edge.fa->V0(edge.ea);
            Mesh::VertexPointer v1a = edge.fa->V1(edge.ea);
            Mesh::VertexPointer v0b = edge.fb->V0(edge.eb);
            Mesh::VertexPointer v1b = edge.fb->V1(edge.eb);

            if (v0a->P() != edge.V(0)->P())
                std::swap(v0a, v1a);
            if (v0b->P() != edge.V(0)->P())
                std::swap(v0b, v1b);

            if (diff.replacements.count(v0a) == 0)
                diff.evec[edge.V(0)].push_back(v0a);
            diff.replacements[v0a] = diff.evec[edge.V(0)].front();

            if (diff.replacements.count(v1a) == 0)
                diff.evec[edge.V(1)].push_back(v1a);
            diff.replacements[v1a] = diff.evec[edge.V(1)].front();

            if (diff.replacements.count(v0b) == 0)
                diff.evec[edge.V(0)].push_back(v0b);
            diff.replacements[v0b] = diff.evec[edge.V(0)].front();

            if (diff.replacements.count(v1b) == 0)
                diff.evec[edge.V(1)].push_back(v1b);
            diff.replacements[v1b] = diff.evec[edge.V(1)].front();
        }
    }

    // For each vertex merged, store its original vfadj fan
    for (auto& entry : diff.replacements) {
        vcg::face::VFStarVF(entry.first, diff.vfmap[entry.first].first, diff.vfmap[entry.first].second);
    }

    // Build mergeGroups: reverse mapping of replacements
    // Maps representative vertex -> all vertices that merge into it
    for (auto& entry : diff.replacements) {
        Mesh::VertexPointer original = entry.first;
        Mesh::VertexPointer representative = entry.second;
        diff.mergeGroups[representative].push_back(original);
    }

    // Compute average positions and displacements (offset map)
    for (auto& entry : diff.evec) {
        vcg::Point2d sumpos = vcg::Point2d::Zero();
        for (auto vp : entry.second) {
            sumpos += virtualUV[vp];
        }
        vcg::Point2d avg = sumpos / (double) entry.second.size();

        double maxOffset = 0;
        for (auto& vp : entry.second)
            maxOffset = std::max(maxOffset, params.offsetFactor * (virtualUV[vp] - avg).Norm());

        diff.offsetMap[entry.second.front()] = maxOffset;
        diff.newUVPositions[entry.second.front()] = avg;
    }

    // Store all virtual UV positions in the diff
    for (auto& entry : virtualUV) {
        if (diff.newUVPositions.find(entry.first) == diff.newUVPositions.end()) {
            diff.newUVPositions[entry.first] = entry.second;
        }
    }

    return diff;
}

// Helper to compute vertices within offset threshold using virtual positions
// CRITICAL: Uses mergeGroups to traverse across the virtual seam merge
static std::unordered_set<Mesh::VertexPointer> ComputeVerticesWithinOffsetThreshold_Virtual(
    const OffsetMap& om, const ChartHandle a, const ChartHandle b,
    const std::unordered_map<Mesh::VertexPointer, Mesh::VertexPointer>& replacements,
    const std::unordered_map<Mesh::VertexPointer, std::vector<Mesh::VertexPointer>>& mergeGroups,
    const std::unordered_map<Mesh::VertexPointer, vcg::Point2d>& virtualUV)
{
    std::unordered_set<Mesh::VertexPointer> vset;

    // Get the representative vertices that define the merge seam
    for (auto& omEntry : om) {
        vset.insert(omEntry.first);
    }

    // BFS from representative vertices using virtual positions
    std::vector<Mesh::VertexPointer> frontier;
    for (auto& entry : om) {
        frontier.push_back(entry.first);
    }

    while (!frontier.empty()) {
        std::vector<Mesh::VertexPointer> next;
        for (auto vp : frontier) {
            double offsetThreshold = 0;
            // Find the nearest seed vertex's offset
            for (auto& entry : om) {
                auto uvIt = virtualUV.find(vp);
                auto seedIt = virtualUV.find(entry.first);
                if (uvIt != virtualUV.end() && seedIt != virtualUV.end()) {
                    double dist = (uvIt->second - seedIt->second).Norm();
                    if (dist <= entry.second) {
                        offsetThreshold = std::max(offsetThreshold, entry.second);
                    }
                }
            }

            if (offsetThreshold > 0) {
                // CRITICAL FIX: Build list of all vertices to query for neighbors
                // This includes vp itself AND all vertices that merge into vp
                std::vector<Mesh::VertexPointer> virtualEquivalents;
                virtualEquivalents.push_back(vp);

                // If vp is a representative, find all vertices that merge into it
                auto groupIt = mergeGroups.find(vp);
                if (groupIt != mergeGroups.end()) {
                    virtualEquivalents.insert(virtualEquivalents.end(),
                                               groupIt->second.begin(),
                                               groupIt->second.end());
                }

                // Query neighbors of ALL virtually-equivalent vertices
                // This allows BFS to cross over to the other chart at the seam
                for (auto v_equiv : virtualEquivalents) {
                    std::vector<Mesh::FacePointer> faces;
                    std::vector<int> indices;
                    vcg::face::VFStarVF(v_equiv, faces, indices);

                    for (size_t i = 0; i < faces.size(); ++i) {
                        for (int j = 0; j < 3; ++j) {
                            Mesh::VertexPointer adj = faces[i]->V(j);
                            // Apply replacement if this vertex would be merged
                            auto repIt = replacements.find(adj);
                            if (repIt != replacements.end()) {
                                adj = repIt->second;
                            }
                            if (vset.find(adj) == vset.end()) {
                                auto uvIt = virtualUV.find(adj);
                                bool withinThreshold = false;
                                if (uvIt != virtualUV.end()) {
                                    for (auto& entry : om) {
                                        auto seedIt = virtualUV.find(entry.first);
                                        if (seedIt != virtualUV.end()) {
                                            if ((uvIt->second - seedIt->second).Norm() <= entry.second) {
                                                withinThreshold = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (withinThreshold) {
                                    vset.insert(adj);
                                    next.push_back(adj);
                                }
                            }
                        }
                    }
                }
            }
        }
        frontier = std::move(next);
    }

    return vset;
}

// ComputeOptimizationArea_Virtual: Compute optimization area from TopologyDiff
static void ComputeOptimizationArea_Virtual(MergeJobResult& result, const TopologyDiff& diff,
                                            const ChartHandle a, const ChartHandle b)
{
    result.verticesWithinThreshold = ComputeVerticesWithinOffsetThreshold_Virtual(
        diff.offsetMap, a, b, diff.replacements, diff.mergeGroups, diff.newUVPositions);

    result.optimizationArea.clear();

    std::vector<Mesh::FacePointer> fpvec;
    fpvec.insert(fpvec.end(), a->fpVec.begin(), a->fpVec.end());
    if (a != b)
        fpvec.insert(fpvec.end(), b->fpVec.begin(), b->fpVec.end());

    for (auto fptr : fpvec) {
        for (int i = 0; i < 3; ++i) {
            Mesh::VertexPointer v = fptr->V(i);
            // Check if the vertex (or its representative) is within threshold
            auto repIt = diff.replacements.find(v);
            if (repIt != diff.replacements.end()) {
                v = repIt->second;
            }
            if (result.verticesWithinThreshold.find(v) != result.verticesWithinThreshold.end()) {
                result.optimizationArea.insert(fptr);
                break;
            }
        }
    }

    // Compute input negative/absolute area using virtual positions
    result.inputNegativeArea = 0;
    result.inputAbsoluteArea = 0;
    for (auto fptr : result.optimizationArea) {
        vcg::Point2d p0, p1, p2;
        auto it0 = diff.newUVPositions.find(fptr->V(0));
        auto it1 = diff.newUVPositions.find(fptr->V(1));
        auto it2 = diff.newUVPositions.find(fptr->V(2));

        p0 = (it0 != diff.newUVPositions.end()) ? it0->second : fptr->V(0)->T().P();
        p1 = (it1 != diff.newUVPositions.end()) ? it1->second : fptr->V(1)->T().P();
        p2 = (it2 != diff.newUVPositions.end()) ? it2->second : fptr->V(2)->T().P();

        double areaUV = 0.5 * ((p1 - p0) ^ (p2 - p0));
        if (areaUV < 0)
            result.inputNegativeArea += areaUV;
        result.inputAbsoluteArea += std::abs(areaUV);
    }
}

// ============================================================================
// ComputeArapEnergyFromVirtualPositions: Thread-safe ARAP energy computation
// Uses virtual UV positions instead of reading from the mesh
// ============================================================================
static double ComputeArapEnergyFromVirtualPositions(const std::vector<Mesh::FacePointer>& faces,
                                                     const std::unordered_map<Mesh::VertexPointer, vcg::Point2d>& virtualUV,
                                                     double* num, double* denom)
{
    double totalNum = 0;
    double totalDenom = 0;

    for (auto fptr : faces) {
        // Get 3D edge vectors
        vcg::Point3d x10_3d = fptr->P(1) - fptr->P(0);
        vcg::Point3d x20_3d = fptr->P(2) - fptr->P(0);

        // Compute local 2D frame for 3D triangle
        vcg::Point2d x10, x20;
        LocalIsometry(x10_3d, x20_3d, x10, x20);

        // Get UV positions (use virtual if available, else from mesh)
        vcg::Point2d u0, u1, u2;
        auto it0 = virtualUV.find(fptr->V(0));
        auto it1 = virtualUV.find(fptr->V(1));
        auto it2 = virtualUV.find(fptr->V(2));

        u0 = (it0 != virtualUV.end()) ? it0->second : fptr->V(0)->T().P();
        u1 = (it1 != virtualUV.end()) ? it1->second : fptr->V(1)->T().P();
        u2 = (it2 != virtualUV.end()) ? it2->second : fptr->V(2)->T().P();

        vcg::Point2d u10 = u1 - u0;
        vcg::Point2d u20 = u2 - u0;

        double area;
        double energy = ARAP::ComputeEnergy(x10, x20, u10, u20, &area);

        totalNum += energy;
        totalDenom += area;
    }

    if (num) *num = totalNum;
    if (denom) *denom = totalDenom;

    return (totalDenom > 0) ? totalNum / totalDenom : 0;
}

// ============================================================================
// CheckBoundary_Virtual: Check if charts collide outside optimization area
// Uses virtual positions from TopologyDiff
// ============================================================================
static CheckStatus CheckBoundary_Virtual(const MergeJobResult& result, const TopologyDiff& diff,
                                         const ChartHandle a, const ChartHandle b)
{
    if (a == b) return PASS;

    // Helper to get virtual UV position
    auto getVirtualUV = [&diff](Mesh::VertexPointer v) -> vcg::Point2d {
        auto it = diff.newUVPositions.find(v);
        if (it != diff.newUVPositions.end()) return it->second;
        // Apply replacement then look up
        auto repIt = diff.replacements.find(v);
        if (repIt != diff.replacements.end()) {
            auto it2 = diff.newUVPositions.find(repIt->second);
            if (it2 != diff.newUVPositions.end()) return it2->second;
        }
        return v->T().P();
    };

    // Extract boundary half-edges from fixed areas of A
    std::vector<std::pair<vcg::Point2d, vcg::Point2d>> aEdges;
    for (auto fptr : a->fpVec) {
        if (result.optimizationArea.find(fptr) == result.optimizationArea.end()) {
            for (int i = 0; i < 3; ++i) {
                if (vcg::face::IsBorder(*fptr, i) ||
                    (result.optimizationArea.find(fptr->FFp(i)) != result.optimizationArea.end())) {
                    vcg::Point2d p0 = getVirtualUV(fptr->V0(i));
                    vcg::Point2d p1 = getVirtualUV(fptr->V1(i));
                    aEdges.push_back({p0, p1});
                }
            }
        }
    }

    // Extract boundary half-edges from fixed areas of B
    std::vector<std::pair<vcg::Point2d, vcg::Point2d>> bEdges;
    for (auto fptr : b->fpVec) {
        if (result.optimizationArea.find(fptr) == result.optimizationArea.end()) {
            for (int i = 0; i < 3; ++i) {
                if (vcg::face::IsBorder(*fptr, i) ||
                    (result.optimizationArea.find(fptr->FFp(i)) != result.optimizationArea.end())) {
                    vcg::Point2d p0 = getVirtualUV(fptr->V0(i));
                    vcg::Point2d p1 = getVirtualUV(fptr->V1(i));
                    bEdges.push_back({p0, p1});
                }
            }
        }
    }

    // Simple segment-segment intersection test
    auto segmentsIntersect = [](const vcg::Point2d& a0, const vcg::Point2d& a1,
                                 const vcg::Point2d& b0, const vcg::Point2d& b1) -> bool {
        vcg::Point2d d1 = a1 - a0;
        vcg::Point2d d2 = b1 - b0;
        double cross = d1.X() * d2.Y() - d1.Y() * d2.X();
        if (std::abs(cross) < 1e-12) return false; // Parallel

        vcg::Point2d diff = b0 - a0;
        double t = (diff.X() * d2.Y() - diff.Y() * d2.X()) / cross;
        double s = (diff.X() * d1.Y() - diff.Y() * d1.X()) / cross;

        const double eps = 1e-9;
        return t > eps && t < 1.0 - eps && s > eps && s < 1.0 - eps;
    };

    // Check for intersections
    for (const auto& aEdge : aEdges) {
        for (const auto& bEdge : bEdges) {
            if (segmentsIntersect(aEdge.first, aEdge.second, bEdge.first, bEdge.second)) {
                return FAIL_GLOBAL_OVERLAP_BEFORE;
            }
        }
    }

    return PASS;
}

// ============================================================================
// OptimizeChart_Virtual: Run ARAP on thread-local shell
// THREAD-SAFE: Does NOT modify the global mesh at all
// Updates the TopologyDiff with new UV positions
// additionalFixedVertices: vertices to fix in addition to those outside the threshold
//                          (used for retry mechanism when fixing intersecting edges)
// ============================================================================
static CheckStatus OptimizeChart_Virtual(MergeJobResult& result, TopologyDiff& diff,
                                         const std::vector<Mesh::FacePointer>& supportFaces, Mesh& mesh,
                                         const AlgoParameters& params,
                                         const std::unordered_set<Mesh::VertexPointer>& additionalFixedVertices = {})
{
    if (supportFaces.empty()) {
        return FAIL_TOPOLOGY;
    }

    // [UV SCALE GUARD] Measure UV bounding box before ARAP optimization using virtual positions
    vcg::Box2d optBoxBefore;
    for (auto fptr : supportFaces) {
        for (int i = 0; i < 3; ++i) {
            auto uvIt = diff.newUVPositions.find(fptr->V(i));
            vcg::Point2d uv = (uvIt != diff.newUVPositions.end()) ? uvIt->second : fptr->V(i)->T().P();
            optBoxBefore.Add(uv);
        }
    }
    double scaleBefore = std::max(optBoxBefore.DimX(), optBoxBefore.DimY());
    if (scaleBefore <= 0 || !std::isfinite(scaleBefore)) {
        scaleBefore = 1.0;
    }

    // Compute input ARAP energy from virtual positions (thread-safe)
    // Only compute on first call (when additionalFixedVertices is empty)
    if (additionalFixedVertices.empty()) {
        ComputeArapEnergyFromVirtualPositions(supportFaces, diff.newUVPositions,
                                              &result.inputArapNum, &result.inputArapDenom);
    }

    // Build shell mesh by copying faces
    // CRITICAL: Apply vertex replacements to connect the seam in the shell
    Mesh shell;

    // Helper to resolve vertex to its representative (merged vertex)
    auto resolveVertex = [&diff](Mesh::VertexPointer v) -> Mesh::VertexPointer {
        auto repIt = diff.replacements.find(v);
        return (repIt != diff.replacements.end()) ? repIt->second : v;
    };

    // Create a mapping from REPRESENTATIVE vertices to shell vertices
    // This ensures merged vertices map to the same shell vertex
    std::unordered_map<Mesh::VertexPointer, int> origToShellVert;
    std::vector<Mesh::VertexPointer> shellToOrigVert;

    // First pass: identify unique REPRESENTATIVE vertices and assign shell indices
    for (auto fptr : supportFaces) {
        for (int i = 0; i < 3; ++i) {
            Mesh::VertexPointer v = fptr->V(i);

            // CRITICAL FIX: Resolve to representative if this vertex is being merged
            Mesh::VertexPointer rep = resolveVertex(v);

            if (origToShellVert.find(rep) == origToShellVert.end()) {
                origToShellVert[rep] = (int)shellToOrigVert.size();
                shellToOrigVert.push_back(rep);
            }
        }
    }

    // Allocate shell vertices and faces
    tri::Allocator<Mesh>::AddVertices(shell, shellToOrigVert.size());
    tri::Allocator<Mesh>::AddFaces(shell, supportFaces.size());

    // Initialize face index attribute for shell
    // THREAD-SAFETY FIX: VCG attribute system uses static global maps
    Mesh::PerFaceAttributeHandle<int> shellFaceIdx;
    Mesh::PerFaceAttributeHandle<CoordStorage> tsa;
    Mesh::PerFaceAttributeHandle<CoordStorage> sa;
    {
        std::lock_guard<std::mutex> lock(g_configMutex);
        shellFaceIdx = tri::Allocator<Mesh>::AddPerFaceAttribute<int>(shell, "OriginalFaceIndex");
        tsa = tri::Allocator<Mesh>::AddPerFaceAttribute<CoordStorage>(shell, "TargetShape");
        sa = tri::Allocator<Mesh>::AddPerFaceAttribute<CoordStorage>(shell, "Shell3DShape");
    }

    // Copy vertex data with virtual UV positions
    for (size_t vi = 0; vi < shellToOrigVert.size(); ++vi) {
        Mesh::VertexPointer origV = shellToOrigVert[vi];
        shell.vert[vi].P() = origV->P();  // 3D position

        // Use virtual UV position (already stored for representative)
        auto uvIt = diff.newUVPositions.find(origV);
        if (uvIt != diff.newUVPositions.end()) {
            shell.vert[vi].T().P() = uvIt->second;
        } else {
            shell.vert[vi].T().P() = origV->T().P();
        }
    }

    // Copy face data - use REPRESENTATIVE vertices for shell indices
    for (size_t fi = 0; fi < supportFaces.size(); ++fi) {
        Mesh::FacePointer origF = supportFaces[fi];
        auto& sf = shell.face[fi];

        for (int i = 0; i < 3; ++i) {
            // CRITICAL FIX: Resolve to representative before looking up shell index
            Mesh::VertexPointer rep = resolveVertex(origF->V(i));
            int shellVi = origToShellVert[rep];
            sf.V(i) = &shell.vert[shellVi];
            sf.WT(i).P() = shell.vert[shellVi].T().P();
        }

        // Store original face index
        shellFaceIdx[sf] = tri::Index(mesh, origF);
    }

    // Update topology and bounding box
    tri::UpdateTopology<Mesh>::FaceFace(shell);
    tri::UpdateTopology<Mesh>::VertexFace(shell);
    tri::UpdateBounding<Mesh>::Box(shell);

    // Check if single component
    bool singleComponent = (tri::Clean<Mesh>::CountConnectedComponents(shell) == 1);

    // Compute target shapes for ARAP
    for (auto& sf : shell.face) {
        // Store current 3D shape
        sa[sf].P[0] = sf.P(0);
        sa[sf].P[1] = sf.P(1);
        sa[sf].P[2] = sf.P(2);

        // Target shape is the current UV configuration (no downscaling needed for optimization)
        for (int i = 0; i < 3; ++i) {
            tsa[sf].P[i] = vcg::Point3d(sf.WT(i).U(), sf.WT(i).V(), 0);
        }
    }

    // Split non-manifold vertices
    while (tri::Clean<Mesh>::SplitNonManifoldVertex(shell, 0.3))
        ;

    CutAlongSeams(shell);

    int nholes = tri::Clean<Mesh>::CountHoles(shell);
    int genus = tri::Clean<Mesh>::MeshGenus(shell);
    if (nholes == 0 || genus != 0) {
        return FAIL_TOPOLOGY;
    }

    if (singleComponent && nholes > 1)
        CloseHoles3D(shell);

    SyncShellWithUV(shell);

    // Setup ARAP solver
    ARAP arap(shell);
    arap.SetMaxIterations(100);

    // Select vertices to fix:
    // 1. Those outside the threshold
    // 2. Additional vertices from intersecting edges (retry mechanism)
    for (auto& sv : shell.vert) {
        sv.ClearS();
    }

    for (size_t vi = 0; vi < shellToOrigVert.size(); ++vi) {
        Mesh::VertexPointer origV = shellToOrigVert[vi];
        // Check if vertex is within threshold (apply replacement first)
        auto repIt = diff.replacements.find(origV);
        Mesh::VertexPointer checkV = (repIt != diff.replacements.end()) ? repIt->second : origV;

        bool shouldFix = false;

        // Fix if outside threshold
        if (result.verticesWithinThreshold.find(checkV) == result.verticesWithinThreshold.end()) {
            shouldFix = true;
        }

        // Also fix if in additionalFixedVertices (retry mechanism)
        if (additionalFixedVertices.find(origV) != additionalFixedVertices.end()) {
            shouldFix = true;
        } else {
            // The intersection detection finds Original vertices.
            // The virtual shell used for optimization contains Representative vertices.
            // Check if any vertex merged into this representative is in additionalFixedVertices.
            auto mgIt = diff.mergeGroups.find(origV);
            if (mgIt != diff.mergeGroups.end()) {
                for (Mesh::VertexPointer vMerged : mgIt->second) {
                    if (additionalFixedVertices.find(vMerged) != additionalFixedVertices.end()) {
                        shouldFix = true;
                        break;
                    }
                }
            }
        }

        if (shouldFix) {
            shell.vert[vi].SetS();
        }
    }

    // Fix selected vertices
    int nfixed = arap.FixSelectedVertices();
    double tol = 0.02;
    while (nfixed < 2) {
        nfixed += arap.FixRandomEdgeWithinTolerance(tol);
        tol += 0.02;
        if (tol > 1.0) break;  // Prevent infinite loop
    }

    if (nfixed < 2) {
        return FAIL_TOPOLOGY;
    }

    // Solve ARAP
    ARAPSolveInfo si = arap.Solve();
    result.finalEnergy = si.finalEnergy;
    result.initialEnergy = si.initialEnergy;
    result.arapIterations = si.iterations;

    if (si.numericalError) {
        return FAIL_NUMERICAL_ERROR;
    }

    // [UV SCALE GUARD] Check for UV scale explosion
    vcg::Box2d optBoxAfter;
    for (auto& sf : shell.face) {
        if (sf.IsHoleFilling()) continue;
        for (int i = 0; i < 3; ++i) {
            optBoxAfter.Add(sf.V(i)->T().P());
        }
    }
    double scaleAfter = std::max(optBoxAfter.DimX(), optBoxAfter.DimY());
    double scaleRatio = (scaleBefore > 0) ? scaleAfter / scaleBefore : 1.0;

    const double MAX_LOCAL_SCALE_RATIO = 15.0;
    if (!std::isfinite(scaleRatio) || scaleRatio > MAX_LOCAL_SCALE_RATIO) {
        LOG_WARN << "[Parallel] ARAP produced extreme UV scale explosion (ratio=" << scaleRatio << ")";
        LogArapExplosionDiagnostics(shell);
        return FAIL_DISTORTION_LOCAL;
    }

    SyncShellWithUV(shell);

    // Extract new UV positions from shell and store in diff
    // Map shell vertices back to original vertices
    for (size_t vi = 0; vi < std::min(shellToOrigVert.size(), (size_t)shell.VN()); ++vi) {
        if (vi < shell.vert.size()) {
            Mesh::VertexPointer origV = shellToOrigVert[vi];
            vcg::Point2d newPos = shell.vert[vi].T().P();
            diff.newUVPositions[origV] = newPos;
        }
    }

    // Also update wedge positions by iterating through faces
    for (size_t fi = 0; fi < supportFaces.size() && fi < shell.face.size(); ++fi) {
        if (!shell.face[fi].IsHoleFilling()) {
            Mesh::FacePointer origF = supportFaces[fi];
            for (int k = 0; k < 3; ++k) {
                vcg::Point2d newPos = shell.face[fi].V(k)->T().P();
                diff.newWedgeUVPositions[{origF, k}] = newPos;
            }
        }
    }

    // Compute output ARAP energy using the new virtual positions
    ComputeArapEnergyFromVirtualPositions(supportFaces, diff.newUVPositions,
                                          &result.outputArapNum, &result.outputArapDenom);

    // Compute result bounding box for collision detection
    diff.resultBoundingBox.SetNull();
    for (auto fptr : supportFaces) {
        for (int i = 0; i < 3; ++i) {
            auto uvIt = diff.newUVPositions.find(fptr->V(i));
            if (uvIt != diff.newUVPositions.end()) {
                diff.resultBoundingBox.Add(uvIt->second);
            }
        }
    }

    return PASS;
}

// Helper to get UV position using virtual positions or mesh
static vcg::Point2d GetVirtualUV(Mesh::VertexPointer v, const TopologyDiff& diff)
{
    auto it = diff.newUVPositions.find(v);
    if (it != diff.newUVPositions.end()) return it->second;
    return v->T().P();
}

// Helper to check if two segments intersect (using virtual positions)
static bool SegmentsIntersectVirtual(Mesh::FacePointer f1, int e1, Mesh::FacePointer f2, int e2,
                                     const TopologyDiff& diff)
{
    vcg::Point2d a0 = GetVirtualUV(f1->V0(e1), diff);
    vcg::Point2d a1 = GetVirtualUV(f1->V1(e1), diff);
    vcg::Point2d b0 = GetVirtualUV(f2->V0(e2), diff);
    vcg::Point2d b1 = GetVirtualUV(f2->V1(e2), diff);

    vcg::Point2d d1 = a1 - a0;
    vcg::Point2d d2 = b1 - b0;
    double cross = d1.X() * d2.Y() - d1.Y() * d2.X();
    if (std::abs(cross) < 1e-12) return false; // Parallel

    vcg::Point2d d = b0 - a0;
    double t = (d.X() * d2.Y() - d.Y() * d2.X()) / cross;
    double s = (d.X() * d1.Y() - d.Y() * d1.X()) / cross;

    const double eps = 1e-9;
    return t > eps && t < 1.0 - eps && s > eps && s < 1.0 - eps;
}

// ============================================================================
// CheckAfterOptimization_Virtual: Validate after ARAP optimization
// Uses virtual positions from TopologyDiff
// NOTE: Takes globalArapNum/Denom as copies to avoid accessing shared state
// Populates result.intersectionOpt/Boundary for retry mechanism
// ============================================================================
static CheckStatus CheckAfterOptimization_Virtual(MergeJobResult& result, const TopologyDiff& diff,
                                                  double globalArapNum, double globalArapDenom,
                                                  const AlgoParameters& params,
                                                  const ChartHandle a, const ChartHandle b)
{
    // Check global ARAP energy (guard against division by zero)
    if (globalArapDenom > 0) {
        double newArapVal = (globalArapNum + (result.outputArapNum - result.inputArapNum)) / globalArapDenom;
        if (newArapVal > params.globalDistortionThreshold) {
            LOG_DEBUG << "[Parallel] Rejecting for global ARAP energy " << newArapVal << " > " << params.globalDistortionThreshold;
            return FAIL_DISTORTION_GLOBAL;
        }
    }

    // Check local distortion (guard against division by zero)
    if (result.outputArapDenom > 0) {
        double localDistortion = result.outputArapNum / result.outputArapDenom;
        if (localDistortion > params.distortionTolerance) {
            LOG_DEBUG << "[Parallel] Rejecting for local distortion " << localDistortion << " > " << params.distortionTolerance;
            return FAIL_DISTORTION_LOCAL;
        }
    }

    // Check UV size limit
    const double MAX_UV_DIM = 32000.0;
    if (!diff.resultBoundingBox.IsNull() &&
        (diff.resultBoundingBox.DimX() > MAX_UV_DIM || diff.resultBoundingBox.DimY() > MAX_UV_DIM)) {
        LOG_DEBUG << "[Parallel] Rejecting for UV size explosion";
        return FAIL_DISTORTION_GLOBAL;
    }

    // Check folded area ratio
    double outputNegativeArea = 0;
    double outputAbsoluteArea = 0;
    for (auto fptr : result.optimizationArea) {
        vcg::Point2d p0 = GetVirtualUV(fptr->V(0), diff);
        vcg::Point2d p1 = GetVirtualUV(fptr->V(1), diff);
        vcg::Point2d p2 = GetVirtualUV(fptr->V(2), diff);

        double areaUV = 0.5 * ((p1 - p0) ^ (p2 - p0));
        if (areaUV < 0)
            outputNegativeArea += areaUV;
        outputAbsoluteArea += std::abs(areaUV);
    }

    // Guard against division by zero in area ratio check
    if (result.inputAbsoluteArea > 1e-12 && outputAbsoluteArea > 1e-12) {
        double inputRatio = std::abs(result.inputNegativeArea / result.inputAbsoluteArea);
        double outputRatio = std::abs(outputNegativeArea / outputAbsoluteArea);

        if (outputRatio > inputRatio) {
            return FAIL_LOCAL_OVERLAP;
        }
    }

    // Clear previous intersection data
    result.intersectionOpt.clear();
    result.intersectionBoundary.clear();
    result.intersectionInternal.clear();

    // Helper to check if edge pair vertices are all already fixed
    auto AllVerticesFixed = [&result](Mesh::FacePointer f1, int e1, Mesh::FacePointer f2, int e2) -> bool {
        return result.fixedVerticesFromIntersectingEdges.count(f1->V0(e1)) > 0
            && result.fixedVerticesFromIntersectingEdges.count(f1->V1(e1)) > 0
            && result.fixedVerticesFromIntersectingEdges.count(f2->V0(e2)) > 0
            && result.fixedVerticesFromIntersectingEdges.count(f2->V1(e2)) > 0;
    };

    auto FirstEdgeFixed = [&result](Mesh::FacePointer f1, int e1) -> bool {
        return result.fixedVerticesFromIntersectingEdges.count(f1->V0(e1)) > 0
            && result.fixedVerticesFromIntersectingEdges.count(f1->V1(e1)) > 0;
    };

    // Extract boundary half-edges from optimization area (using virtual positions)
    std::vector<std::pair<Mesh::FacePointer, int>> optBoundary;
    for (auto fptr : result.optimizationArea) {
        for (int i = 0; i < 3; ++i) {
            if (vcg::face::IsBorder(*fptr, i) ||
                (result.optimizationArea.find(fptr->FFp(i)) == result.optimizationArea.end())) {
                optBoundary.push_back({fptr, i});
            }
        }
    }

    // Check optimization boundary self-intersection
    for (size_t i = 0; i < optBoundary.size(); ++i) {
        for (size_t j = i + 1; j < optBoundary.size(); ++j) {
            auto& he1 = optBoundary[i];
            auto& he2 = optBoundary[j];

            // Skip adjacent edges (they share a vertex)
            if (he1.first->V0(he1.second) == he2.first->V0(he2.second) ||
                he1.first->V0(he1.second) == he2.first->V1(he2.second) ||
                he1.first->V1(he1.second) == he2.first->V0(he2.second) ||
                he1.first->V1(he1.second) == he2.first->V1(he2.second)) {
                continue;
            }

            if (SegmentsIntersectVirtual(he1.first, he1.second, he2.first, he2.second, diff)) {
                if (!AllVerticesFixed(he1.first, he1.second, he2.first, he2.second)) {
                    result.intersectionOpt.push_back({HalfEdge{he1.first, he1.second},
                                                       HalfEdge{he2.first, he2.second}});
                }
            }
        }
    }

    if (!result.intersectionOpt.empty()) {
        return FAIL_GLOBAL_OVERLAP_AFTER_OPT;
    }

    // Extract fixed boundary edges (outside optimization area)
    std::vector<std::pair<Mesh::FacePointer, int>> fixedBoundary;
    for (auto fptr : a->fpVec) {
        if (result.optimizationArea.find(fptr) == result.optimizationArea.end()) {
            for (int i = 0; i < 3; ++i) {
                if (vcg::face::IsBorder(*fptr, i)) {
                    fixedBoundary.push_back({fptr, i});
                }
            }
        }
    }
    if (a != b) {
        for (auto fptr : b->fpVec) {
            if (result.optimizationArea.find(fptr) == result.optimizationArea.end()) {
                for (int i = 0; i < 3; ++i) {
                    if (vcg::face::IsBorder(*fptr, i)) {
                        fixedBoundary.push_back({fptr, i});
                    }
                }
            }
        }
    }

    // Check optimization boundary vs fixed boundary intersection
    for (auto& optHe : optBoundary) {
        for (auto& fixHe : fixedBoundary) {
            // Skip if same face
            if (optHe.first == fixHe.first) continue;

            // Skip adjacent edges
            if (optHe.first->V0(optHe.second) == fixHe.first->V0(fixHe.second) ||
                optHe.first->V0(optHe.second) == fixHe.first->V1(fixHe.second) ||
                optHe.first->V1(optHe.second) == fixHe.first->V0(fixHe.second) ||
                optHe.first->V1(optHe.second) == fixHe.first->V1(fixHe.second)) {
                continue;
            }

            if (SegmentsIntersectVirtual(optHe.first, optHe.second, fixHe.first, fixHe.second, diff)) {
                if (!FirstEdgeFixed(optHe.first, optHe.second)) {
                    result.intersectionBoundary.push_back({HalfEdge{optHe.first, optHe.second},
                                                           HalfEdge{fixHe.first, fixHe.second}});
                }
            }
        }
    }

    if (!result.intersectionBoundary.empty()) {
        return FAIL_GLOBAL_OVERLAP_AFTER_BND;
    }

    return PASS;
}

// ============================================================================
// ApplyMergeAndUpdateQueue: Commit TopologyDiff and update priority queue
// This is called from the main thread during the Commit phase
// Includes neighbor re-scoring logic from original AcceptMove
// ============================================================================
static void ApplyMergeAndUpdateQueue(const MergeJobResult& result, AlgoStateHandle state, GraphHandle graph,
                                     const AlgoParameters& params)
{
    Mesh& m = graph->mesh;

    ChartHandle a = graph->GetChart(result.chartIdA);
    ChartHandle b = (result.chartIdA != result.chartIdB) ? graph->GetChart(result.chartIdB) : a;

    if (!a || (result.chartIdA != result.chartIdB && !b)) {
        LOG_ERR << "[Parallel] Cannot apply merge: chart not found";
        return;
    }

    const TopologyDiff& diff = result.diff;

    // Collect clusters that need queue updates BEFORE modifying graph structure
    std::vector<SeamHandle> shared;
    std::set<ClusteredSeamHandle> sharedClusters;
    std::set<ClusteredSeamHandle> independentClusters;
    std::set<ClusteredSeamHandle> selfClusters;

    if (a != b) {
        // Seams from B to other charts
        for (auto csh : state->chartSeamMap[b->id]) {
            ChartPair p = GetCharts(csh, graph);
            ChartHandle c = (p.first == b) ? p.second : p.first;
            if (c == a || c == b) {
                selfClusters.insert(csh);
            } else if (a->adj.find(c) == a->adj.end()) {
                independentClusters.insert(csh);
            } else {
                sharedClusters.insert(csh);
                for (auto sh : csh->seams)
                    shared.push_back(sh);
            }
        }

        // Seams from A to other charts
        for (auto csh : state->chartSeamMap[a->id]) {
            ChartPair p = GetCharts(csh, graph);
            ChartHandle c = (p.first == a) ? p.second : p.first;
            if (c == a || c == b) {
                selfClusters.insert(csh);
            } else if (b->adj.find(c) == b->adj.end()) {
                independentClusters.insert(csh);
            } else {
                sharedClusters.insert(csh);
                for (auto sh : csh->seams)
                    shared.push_back(sh);
            }
        }
    } else {
        // Self-merge: all clusters are independent
        independentClusters.insert(state->chartSeamMap[b->id].begin(), state->chartSeamMap[b->id].end());
        independentClusters.erase(result.csh);
    }

    // 1. Apply alignment transform to chart B vertices
    if (a != b) {
        std::unordered_set<Mesh::VertexPointer> visited;
        for (auto fptr : b->fpVec) {
            for (int i = 0; i < 3; ++i) {
                if (visited.count(fptr->V(i)) == 0) {
                    visited.insert(fptr->V(i));
                    fptr->V(i)->T().P() = diff.transform.Apply(fptr->V(i)->T().P());
                }
            }
        }
    }

    // 2. Update vertex references to representatives
    for (auto fptr : a->fpVec) {
        for (int i = 0; i < 3; ++i) {
            auto repIt = diff.replacements.find(fptr->V(i));
            if (repIt != diff.replacements.end()) {
                fptr->V(i) = repIt->second;
            }
        }
    }
    if (a != b) {
        for (auto fptr : b->fpVec) {
            for (int i = 0; i < 3; ++i) {
                auto repIt = diff.replacements.find(fptr->V(i));
                if (repIt != diff.replacements.end()) {
                    fptr->V(i) = repIt->second;
                }
            }
        }
    }

    // 3. Update face-face topology along the seam
    SeamMesh& seamMesh = result.csh->sm;
    for (SeamHandle sh : result.csh->seams) {
        for (int iedge : sh->edges) {
            SeamEdge& edge = seamMesh.edge[iedge];
            edge.fa->FFp(edge.ea) = edge.fb;
            edge.fa->FFi(edge.ea) = edge.eb;
            edge.fb->FFp(edge.eb) = edge.fa;
            edge.fb->FFi(edge.eb) = edge.ea;
        }
    }

    // 4. Update vertex-face topology (concatenate VF adjacency lists)
    for (auto& entry : diff.evec) {
        if (entry.second.size() > 1) {
            const std::vector<Mesh::VertexPointer>& verts = entry.second;
            for (unsigned i = 0; i < verts.size() - 1; ++i) {
                auto vfIt = diff.vfmap.find(verts[i]);
                auto vfNextIt = diff.vfmap.find(verts[i+1]);
                if (vfIt != diff.vfmap.end() && vfNextIt != diff.vfmap.end()) {
                    const auto& fan = vfIt->second;
                    const auto& nextFan = vfNextIt->second;
                    if (!fan.first.empty() && !nextFan.first.empty()) {
                        fan.first.back()->VFp(fan.second.back()) = nextFan.first.front();
                        fan.first.back()->VFi(fan.second.back()) = nextFan.second.front();
                    }
                }
            }
        }
    }

    // 5. Apply computed UV positions
    for (auto& entry : diff.newUVPositions) {
        entry.first->T().P() = entry.second;
    }
    for (auto& entry : diff.newWedgeUVPositions) {
        entry.first.first->WT(entry.first.second).P() = entry.second;
    }

    // 6. Update wedge texture coordinates from vertex texture for all affected faces
    for (auto fptr : a->fpVec) {
        for (int i = 0; i < 3; ++i) {
            fptr->WT(i).P() = fptr->V(i)->T().P();
        }
    }
    if (a != b) {
        for (auto fptr : b->fpVec) {
            for (int i = 0; i < 3; ++i) {
                fptr->WT(i).P() = fptr->V(i)->T().P();
            }
        }
    }

    // 7. Update MeshGraph structure
    if (a != b) {
        for (auto fptr : b->fpVec)
            fptr->id = a->Fp()->id;
        a->fpVec.insert(a->fpVec.end(), b->fpVec.begin(), b->fpVec.end());

        a->adj.erase(b);
        for (auto c : b->adj) {
            if (c != a) {
                c->adj.erase(b);
                c->adj.insert(a);
                a->adj.insert(c);
            }
        }
        graph->charts.erase(b->id);

        // Update state chartSeamMap and failed sets
        state->chartSeamMap.erase(b->id);
        std::set<RegionID>& failed_b = state->failed[b->id];
        state->failed[a->id].insert(failed_b.begin(), failed_b.end());
        state->failed.erase(b->id);
    }

    // 8. Invalidate cache
    a->ParameterizationChanged();

    // 9. Update ARAP energy state
    state->arapNum += (result.outputArapNum - result.inputArapNum);
    state->arapDenom += (result.outputArapDenom - result.inputArapDenom);

    // 10. Update UV border length
    double deltaUVBorderLength = a->BorderUV() - result.inputUVBorderLength;
    state->currentUVBorderLength += deltaUVBorderLength;

    // 11. Track changed faces
    state->changeSet.insert(result.optimizationArea.begin(), result.optimizationArea.end());

    // 12. Erase the merged seam
    EraseSeam(result.csh, state, graph);
    state->penalty.erase(result.csh);

    // 13. Re-insert independent clusters into queue
    for (auto csh : independentClusters) {
        auto it = state->status.find(csh);
        if (it == state->status.end()) continue;

        CheckStatus clusterStatus = it->second;
        CostInfo::MatchingValue mv = state->mvalue[csh];

        EraseSeam(csh, state, graph);

        bool invalidate = (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_BEFORE)
                || (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_AFTER_OPT)
                || (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_AFTER_BND)
                || (clusterStatus == CheckStatus::FAIL_GLOBAL_OVERLAP_UNFIXABLE)
                || (clusterStatus == CheckStatus::FAIL_TOPOLOGY);

        if (invalidate || (params.ignoreOnReject && mv == CostInfo::REJECTED))
            InvalidateCluster(csh, state, graph, clusterStatus, 1.0);
        else
            InsertNewClusterInQueue(csh, state, graph, params);
    }

    // 14. Re-cluster and re-insert shared clusters
    for (auto csh : sharedClusters)
        EraseSeam(csh, state, graph);

    std::vector<ClusteredSeamHandle> cshvec = ClusterSeamsByChartId(shared);
    for (auto csh : cshvec) {
        InsertNewClusterInQueue(csh, state, graph, params);
    }

    // 15. Handle unfeasibleBoundaryAdj if visiting components
    if (params.visitComponents) {
        std::set<ClusteredSeamHandle> unfeasibleBoundaryAdj;
        for (ChartHandle c : a->adj) {
            for (auto csh : state->chartSeamMap[c->id]) {
                if (state->mvalue[csh] == CostInfo::MatchingValue::UNFEASIBLE_BOUNDARY)
                    unfeasibleBoundaryAdj.insert(csh);
            }
        }

        for (ClusteredSeamHandle csh : unfeasibleBoundaryAdj) {
            EraseSeam(csh, state, graph);
            InsertNewClusterInQueue(csh, state, graph, params);
        }
    }
}

// ============================================================================
// CheckBoundingBoxCollision: Check if two bounding boxes overlap
// Used for "Moving Walls" bug detection
// ============================================================================
static bool CheckBoundingBoxCollision(const vcg::Box2d& box1, const vcg::Box2d& box2)
{
    if (box1.IsNull() || box2.IsNull()) return false;

    // Check for overlap with a small margin
    const double margin = 1e-6;
    return !(box1.max.X() + margin < box2.min.X() ||
             box2.max.X() + margin < box1.min.X() ||
             box1.max.Y() + margin < box2.min.Y() ||
             box2.max.Y() + margin < box1.min.Y());
}

// ============================================================================
// GreedyOptimization_Parallel: Main parallel optimization loop
// ============================================================================
void GreedyOptimization_Parallel(GraphHandle graph, AlgoStateHandle state, const AlgoParameters& params)
{
    ClearGlobals();

    Timer t;
    Timer tglobal;

    PrintStateInfo(state, graph, params);

    LOG_INFO << "Atlas energy before optimization is " << ARAP::ComputeEnergyFromStoredWedgeTC(graph->mesh, nullptr, nullptr);
    LOG_INFO << "Starting PARALLEL greedy optimization loop with " << state->queue.size() << " operations...";

    // Get parallel configuration
    ParallelConfig config = GetParallelConfig();

#ifdef _OPENMP
    int numThreads = config.numThreads > 0 ? config.numThreads : omp_get_max_threads();
    omp_set_num_threads(numThreads);

    // CRITICAL: Disable nested parallelism to prevent thread oversubscription
    // ARAP::Solve() has internal OpenMP parallel loops - without this fix,
    // 16 threads spawning 16 threads each = 256 threads, killing performance
    omp_set_max_active_levels(1);
#else
    int numThreads = 1;
    LOG_WARN << "OpenMP not available, falling back to single-threaded execution";
#endif

    const int BASE_BATCH_SIZE = numThreads * config.batchMultiplier;
    LOG_INFO << "Using " << numThreads << " threads with base batch size " << BASE_BATCH_SIZE;

    std::vector<WeightedSeam> deferredOps;
    std::vector<WeightedSeam> currentBatch;
    std::unordered_set<RegionID> lockedCharts;

    // Memory optimization: track total faces in batch to avoid OOM on large charts
    // Each thread copies UV data for its charts, so memory scales with (threads * chart_faces)
    const size_t HEAVY_CHART_THRESHOLD = config.heavyTaskThreshold;
    const size_t MAX_BATCH_FACES = HEAVY_CHART_THRESHOLD * numThreads; // Total faces across all batch items

    std::atomic<int> totalAccepted{0};
    std::atomic<int> totalRejected{0};
    std::atomic<int> totalCollisions{0};
    int batchNumber = 0;

    // Use relative logging for batches
    size_t nCharts = graph->Count();
    int estimatedBatches = std::max(1, (int)(nCharts / BASE_BATCH_SIZE));
    int logBatchInterval = std::max(100, estimatedBatches / 20);
    LOG_INFO << "Log batch frequency set to every " << logBatchInterval << " batches.";

    while (state->queue.size() > 0) {
        batchNumber++;

        if (batchNumber <= 100 || (batchNumber % logBatchInterval) == 0) {
            LOG_INFO << "Batch " << batchNumber << " - Queue size: " << state->queue.size()
                     << ", Accepted: " << totalAccepted.load() << ", Rejected: " << totalRejected.load();
        }

        // Purge queue if it gets too large
        if (state->queue.size() > 2 * state->cost.size())
            PurgeQueue(state);

        if (state->queue.size() == 0) {
            LOG_INFO << "Queue is empty, interrupting.";
            break;
        }

        if (params.timelimit > 0 && t.TimeElapsed() > params.timelimit) {
            LOG_INFO << "Timelimit hit, interrupting.";
            break;
        }

        if (params.UVBorderLengthReduction > (state->currentUVBorderLength / state->inputUVBorderLength)) {
            LOG_INFO << "Target UV border reduction reached, interrupting.";
            break;
        }

        Timer batchTimer;

        // --- PHASE 1: SELECTION (Serial) ---
        currentBatch.clear();
        lockedCharts.clear();
        size_t batchFaceCount = 0; // Track total faces for memory optimization

        while (currentBatch.size() < (size_t)BASE_BATCH_SIZE && !state->queue.empty()) {
            WeightedSeam ws = state->queue.top();
            state->queue.pop();

            if (!Valid(ws, state)) {
                continue;
            }

            if (!std::isfinite(ws.second)) {
                // Sanity check - push it back if we have other work
                if (!currentBatch.empty()) {
                    state->queue.push(ws);
                }
                break;
            }

            ChartPair charts = GetCharts(ws.first, graph);
            if (!charts.first || !charts.second) {
                continue;
            }

            RegionID idA = charts.first->id;
            RegionID idB = charts.second->id;

            // Memory optimization: calculate combined face count for this seam
            size_t seamFaceCount = charts.first->FN();
            if (charts.first != charts.second) {
                seamFaceCount += charts.second->FN();
            }

            // Check if adding this seam would exceed memory budget
            // If batch is empty, always allow at least one item (process serially if needed)
            if (!currentBatch.empty() && batchFaceCount + seamFaceCount > MAX_BATCH_FACES) {
                // Would exceed memory limit - defer to next batch
                deferredOps.push_back(ws);
                continue;
            }

            // Check locks with Distance-1 Independence:
            // Two seams can only be processed in parallel if their charts don't share
            // boundary vertices. This happens when charts are adjacent, so we must check:
            // 1. Neither chart is already locked (Distance-0)
            // 2. Neither chart is adjacent to an already-locked chart (Distance-1)
            bool conflict = false;

            // Distance-0 check: charts themselves
            if (lockedCharts.count(idA) || lockedCharts.count(idB)) {
                conflict = true;
            }

            // Distance-1 check: neighbors of chart A
            if (!conflict) {
                for (const auto& neighbor : charts.first->adj) {
                    if (lockedCharts.count(neighbor->id)) {
                        conflict = true;
                        break;
                    }
                }
            }

            // Distance-1 check: neighbors of chart B
            if (!conflict) {
                for (const auto& neighbor : charts.second->adj) {
                    if (lockedCharts.count(neighbor->id)) {
                        conflict = true;
                        break;
                    }
                }
            }

            if (conflict) {
                deferredOps.push_back(ws); // Conflict, try next batch
            } else {
                currentBatch.push_back(ws);
                lockedCharts.insert(idA);
                lockedCharts.insert(idB);
                batchFaceCount += seamFaceCount;

                // If this single seam is very large, stop adding more to reduce memory pressure
                if (seamFaceCount > HEAVY_CHART_THRESHOLD) {
                    break;
                }
            }
        }

        if (currentBatch.empty()) {
            // Re-insert deferred ops and continue
            for (const auto& op : deferredOps) {
                state->queue.push(op);
            }
            deferredOps.clear();
            if (state->queue.empty()) break;
            continue;
        }

        double selectionTime = batchTimer.TimeElapsed();

        // --- PHASE 2: EVALUATION (Parallel) ---
        Timer evalTimer;
        std::vector<MergeJobResult> results(currentBatch.size());

        // Copy state values needed during parallel execution to avoid race conditions
        // These are read-only during the parallel phase
        std::unordered_map<ClusteredSeamHandle, MatchingTransform> batchTransforms;
        for (const auto& ws : currentBatch) {
            auto it = state->transform.find(ws.first);
            if (it != state->transform.end()) {
                batchTransforms[ws.first] = it->second;
            }
        }
        double globalArapNum = state->arapNum;
        double globalArapDenom = state->arapDenom;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < currentBatch.size(); ++i) {
            MergeJobResult& result = results[i];
            result.csh = currentBatch[i].first;

            // Get charts
            ChartPair charts = GetCharts(result.csh, graph);
            if (!charts.first || !charts.second) {
                result.checkStatus = FAIL_TOPOLOGY;
                continue;
            }

            result.chartIdA = charts.first->id;
            result.chartIdB = charts.second->id;

            // Store original coordinates for potential use
            for (auto fptr : charts.first->fpVec) {
                for (int j = 0; j < 3; ++j) {
                    result.texcoorda.push_back(fptr->V(j)->T().P());
                    result.vertexinda.push_back(vcg::tri::Index(graph->mesh, fptr->V(j)));
                }
            }
            if (charts.first != charts.second) {
                for (auto fptr : charts.second->fpVec) {
                    for (int j = 0; j < 3; ++j) {
                        result.texcoordb.push_back(fptr->V(j)->T().P());
                        result.vertexindb.push_back(vcg::tri::Index(graph->mesh, fptr->V(j)));
                    }
                }
            }

            result.inputUVBorderLength = charts.first->BorderUV();
            if (charts.first != charts.second) {
                result.inputUVBorderLength += charts.second->BorderUV();
            }

            // Get transform from our thread-safe copy
            MatchingTransform transform;
            auto transIt = batchTransforms.find(result.csh);
            if (transIt != batchTransforms.end()) {
                transform = transIt->second;
            } else {
                result.checkStatus = FAIL_TOPOLOGY;
                continue;
            }

            // 1. Virtual Align and Merge
            TopologyDiff diff = AlignAndMerge_Virtual(result.csh, charts.first, charts.second,
                                                       transform, params);

            // 2. Compute Optimization Area
            ComputeOptimizationArea_Virtual(result, diff, charts.first, charts.second);

            // 3. Check boundary after alignment
            CheckStatus status = (charts.first != charts.second) ?
                                 CheckBoundary_Virtual(result, diff, charts.first, charts.second) : PASS;

            if (status == PASS) {
                // Build support faces vector from optimization area
                std::vector<Mesh::FacePointer> supportFaces(result.optimizationArea.begin(),
                                                             result.optimizationArea.end());

                // 4. Run ARAP optimization (thread-safe version)
                status = OptimizeChart_Virtual(result, diff, supportFaces, graph->mesh, params);

                if (status == PASS) {
                    // 5. Validation checks (using copied global values)
                    status = CheckAfterOptimization_Virtual(result, diff, globalArapNum, globalArapDenom,
                                                            params, charts.first, charts.second);
                }

                // 6. Retry mechanism for global overlaps
                const int MAX_RETRIES = 10;
                int retryCount = 0;
                while ((status == FAIL_GLOBAL_OVERLAP_AFTER_OPT || status == FAIL_GLOBAL_OVERLAP_AFTER_BND)
                       && retryCount < MAX_RETRIES) {
                    retryCount++;

                    // Collect vertices from intersecting edges
                    size_t fixedBefore = result.fixedVerticesFromIntersectingEdges.size();

                    for (auto& hep : result.intersectionOpt) {
                        result.fixedVerticesFromIntersectingEdges.insert(hep.first.fp->V0(hep.first.e));
                        result.fixedVerticesFromIntersectingEdges.insert(hep.first.fp->V1(hep.first.e));
                        result.fixedVerticesFromIntersectingEdges.insert(hep.second.fp->V0(hep.second.e));
                        result.fixedVerticesFromIntersectingEdges.insert(hep.second.fp->V1(hep.second.e));
                    }
                    for (auto& hep : result.intersectionBoundary) {
                        result.fixedVerticesFromIntersectingEdges.insert(hep.first.fp->V0(hep.first.e));
                        result.fixedVerticesFromIntersectingEdges.insert(hep.first.fp->V1(hep.first.e));
                    }

                    // If no new vertices were added, retrying won't help
                    if (fixedBefore == result.fixedVerticesFromIntersectingEdges.size()) {
                        break;
                    }

                    // Re-run optimization with additional fixed vertices
                    status = OptimizeChart_Virtual(result, diff, supportFaces, graph->mesh, params,
                                                   result.fixedVerticesFromIntersectingEdges);

                    if (status == PASS) {
                        status = CheckAfterOptimization_Virtual(result, diff, globalArapNum, globalArapDenom,
                                                                params, charts.first, charts.second);
                    } else {
                        break; // Different error, stop retrying
                    }
                }
            }

            result.checkStatus = status;
            if (status == PASS) {
                result.diff = std::move(diff);
            }
        }

        double evaluationTime = evalTimer.TimeElapsed();

        // --- PHASE 3: COMMIT (Serial) ---
        Timer commitTimer;
        std::vector<vcg::Box2d> committedBoxes;
        int batchAccepted = 0;
        int batchRejected = 0;
        int batchCollisions = 0;

        // Sort results by original priority order for determinism
        std::vector<size_t> order(results.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        if (config.deterministicOrder) {
            std::sort(order.begin(), order.end(), [&currentBatch](size_t a, size_t b) {
                return currentBatch[a].second < currentBatch[b].second;
            });
        }

        for (size_t idx : order) {
            MergeJobResult& result = results[idx];

            if (result.checkStatus == PASS) {
                // Check for "Moving Walls" collision with already committed results
                bool collision = false;
                if (config.enableSpatialCheck) {
                    for (const auto& box : committedBoxes) {
                        if (CheckBoundingBoxCollision(result.diff.resultBoundingBox, box)) {
                            collision = true;
                            break;
                        }
                    }
                }

                if (collision) {
                    // Collision detected - re-queue this operation
                    batchCollisions++;
                    deferredOps.push_back(currentBatch[idx]);
                } else {
                    // Apply the merge and update queue (includes EraseSeam and penalty update)
                    ApplyMergeAndUpdateQueue(result, state, graph, params);
                    committedBoxes.push_back(result.diff.resultBoundingBox);

                    batchAccepted++;
                    ColorizeSeam(result.csh, vcg::Color4b(255, 69, 0, 255));
                }
            } else {
                // Rejected - invalidate with penalty
                EraseSeam(result.csh, state, graph);
                InvalidateCluster(result.csh, state, graph, result.checkStatus, PENALTY_MULTIPLIER);

                if (result.chartIdA != result.chartIdB) {
                    state->failed[result.chartIdA].insert(result.chartIdB);
                }

                batchRejected++;
            }
        }

        double commitTime = commitTimer.TimeElapsed();

        // Update global counters
        totalAccepted += batchAccepted;
        totalRejected += batchRejected;
        totalCollisions += batchCollisions;

        // Re-insert deferred operations
        for (const auto& op : deferredOps) {
            state->queue.push(op);
        }
        deferredOps.clear();

        // Log batch stats
        if (batchNumber <= 100 || (batchNumber % 100) == 0 || batchTimer.TimeElapsed() > 5.0) {
            LOG_INFO << "  Batch " << batchNumber << ": size=" << currentBatch.size()
                     << ", accepted=" << batchAccepted << ", rejected=" << batchRejected
                     << ", collisions=" << batchCollisions
                     << " (select=" << std::fixed << std::setprecision(2) << selectionTime * 1000 << "ms"
                     << ", eval=" << evaluationTime * 1000 << "ms"
                     << ", commit=" << commitTime * 1000 << "ms)";
        }
    }

    LOG_INFO << "Parallel optimization completed: " << totalAccepted.load() << " accepted, "
             << totalRejected.load() << " rejected, " << totalCollisions.load() << " spatial collisions";

    PrintStateInfo(state, graph, params);
    LogExecutionStats();

    LOG_INFO << "Atlas energy after optimization is " << ARAP::ComputeEnergyFromStoredWedgeTC(graph->mesh, nullptr, nullptr);
}