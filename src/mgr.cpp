#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace madEscape {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = consts::numAgents,
        .maxInstancesPerWorld = 1000,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

#ifdef MADRONA_CUDA_SUPPORT
static inline uint64_t numTensorBytes(const Tensor &t)
{
    uint64_t num_items = 1;
    uint64_t num_dims = t.numDims();
    for (uint64_t i = 0; i < num_dims; i++) {
        num_items *= t.dims()[i];
    }

    return num_items * (uint64_t)t.numBytesPerItem();
}
#endif

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    WorldReset *worldResetBuffer;
    Action *agentActionsBuffer;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                WorldReset *reset_buffer,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void init() = 0;
    virtual void step() = 0;

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &) = 0;
    virtual void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &) = 0;
#endif

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    virtual Tensor rewardHyperParamsTensor() const = 0;

    static inline Impl * make(const Config &cfg);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;
    RewardHyperParams *rewardHyperParams;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   RewardHyperParams *reward_hyper_params,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          cpuExec(std::move(cpu_exec)),
          rewardHyperParams(reward_hyper_params)
    {}

    inline virtual ~CPUImpl() final
    {
        free(rewardHyperParams);
    }

    inline virtual void init()
    {
        cpuExec.runTaskGraph(TaskGraphID::Init);
    }

    inline virtual void step()
    {
        cpuExec.runTaskGraph(TaskGraphID::Step);
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t, void **, Manager &)
    {
        assert(false);
    }

    virtual void gpuStreamStep(cudaStream_t, void **, Manager &)
    {
        assert(false);
    }
#endif

    virtual Tensor rewardHyperParamsTensor() const final
    {
        return Tensor(rewardHyperParams, TensorElementType::Float32,
            {
                cfg.numPBTPolicies,
                sizeof(RewardHyperParams) / sizeof(float),
            }, Optional<int>::none());
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;
    RewardHyperParams *rewardHyperParams;
    RandKey staggerShuffleRND;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   RewardHyperParams *reward_hyper_params,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   RandKey stagger_shuffle_key)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          stepGraph(gpuExec.buildLaunchGraph(TaskGraphID::Step)),
          rewardHyperParams(reward_hyper_params),
          staggerShuffleRND(stagger_shuffle_key)
    {}

    inline virtual ~CUDAImpl() final
    {
        REQ_CUDA(cudaFree(rewardHyperParams));
    }

    inline virtual void init()
    {
        auto init_graph = gpuExec.buildLaunchGraph(TaskGraphID::Init);
        gpuExec.run(init_graph);
    }

    inline virtual void step()
    {
        gpuExec.run(stepGraph);
    }

#ifdef MADRONA_CUDA_SUPPORT
    inline void ** copyOutObservations(cudaStream_t strm,
                                       void **buffers,
                                       Manager &mgr)
    {
        auto copyFromSim = [&strm](void *dst, const Tensor &src) {
            uint64_t num_bytes = numTensorBytes(src);

            REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
                cudaMemcpyDeviceToDevice, strm));
        };

        // Observations
        copyFromSim(*buffers++, mgr.selfObservationTensor());
        copyFromSim(*buffers++, mgr.myGoalObservationTensor());
        copyFromSim(*buffers++, mgr.enemyGoalObservationTensor());
        copyFromSim(*buffers++, mgr.teamObservationTensor());
        copyFromSim(*buffers++, mgr.enemyObservationTensor());
        copyFromSim(*buffers++, mgr.ballTensor());
        copyFromSim(*buffers++, mgr.stepsRemainingTensor());

        return buffers;
    }

    virtual void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &mgr)
    {
        printf("Simulator stream init\n");

        {
            auto init_graph = gpuExec.buildLaunchGraph(TaskGraphID::Init);
            gpuExec.runAsync(init_graph, strm);
            REQ_CUDA(cudaStreamSynchronize(strm));
        }

        if ((cfg.simFlags & SimFlags::StaggerStarts) ==
                SimFlags::StaggerStarts) {
            HeapArray<WorldReset> resets_staging(cfg.numWorlds);
            for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
                resets_staging[i].reset = 1;
            }

            auto resetSyncStep = [&]()
            {
                cudaMemcpyAsync(worldResetBuffer, resets_staging.data(),
                           sizeof(WorldReset) * cfg.numWorlds,
                           cudaMemcpyHostToDevice, strm);
                gpuExec.runAsync(stepGraph, strm);
                REQ_CUDA(cudaStreamSynchronize(strm));
            };

            HeapArray<int32_t> steps_before_reset(cfg.numWorlds);

            CountT cur_world_idx = 0;
            CountT step_idx;
            for (step_idx = 0;
                    step_idx < (CountT)consts::episodeLen;
                    step_idx++) {
                CountT worlds_remaining = 
                    (CountT)cfg.numWorlds - cur_world_idx;
                CountT episode_steps_remaining = consts::episodeLen - step_idx;

                CountT worlds_per_step = madrona::utils::divideRoundUp(
                    worlds_remaining, episode_steps_remaining);

                bool finished = false;
                for (CountT i = 0; i < worlds_per_step; i++) {
                    if (cur_world_idx >= (CountT)cfg.numWorlds) {
                        finished = true;
                        break;
                    }

                    steps_before_reset[cur_world_idx++] = step_idx;
                }

                if (finished || worlds_per_step == 0) {
                    break;
                }
            }

            assert(cur_world_idx == (CountT)cfg.numWorlds);

            for (int32_t i = 0; i < (int32_t)cfg.numWorlds - 1; i++) {
                int32_t j = rand::sampleI32(
                    rand::split_i(staggerShuffleRND, i), i, cfg.numWorlds);

                std::swap(steps_before_reset[i], steps_before_reset[j]);
            }

            
            CountT max_steps = step_idx;
            for (CountT step = 0; step < max_steps; step++) {
                for (CountT world_idx = 0; world_idx < (CountT)cfg.numWorlds;
                     world_idx++) {
                    if (steps_before_reset[world_idx] == step) {
                        resets_staging[world_idx].reset = 1;
                    } else {
                        resets_staging[world_idx].reset = 0;
                    }
                }

                resetSyncStep();
            }
        }

        copyOutObservations(strm, buffers, mgr);

        printf("Simulator stream init finished\n");
    }

    virtual void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &mgr)
    {
        auto copyToSim = [&strm](const Tensor &dst, void *src) {
            uint64_t num_bytes = numTensorBytes(dst);

            REQ_CUDA(cudaMemcpyAsync(dst.devicePtr(), src, num_bytes,
                cudaMemcpyDeviceToDevice, strm));
        };

        copyToSim(mgr.actionTensor(), *buffers++);
        copyToSim(mgr.resetTensor(), *buffers++);

        if (cfg.numPBTPolicies > 0) {
            copyToSim(mgr.policyAssignmentsTensor(), *buffers++);
            copyToSim(mgr.rewardHyperParamsTensor(), *buffers++);
        }

        gpuExec.runAsync(stepGraph, strm);

        buffers = copyOutObservations(strm, buffers, mgr);

        auto copyFromSim = [&strm](void *dst, const Tensor &src) {
            uint64_t num_bytes = numTensorBytes(src);

            REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
                cudaMemcpyDeviceToDevice, strm));
        };

        copyFromSim(*buffers++, mgr.rewardTensor());
        copyFromSim(*buffers++, mgr.doneTensor());
        copyFromSim(*buffers++, mgr.episodeResultTensor());
    }
#endif

    virtual Tensor rewardHyperParamsTensor() const final
    {
        return Tensor(rewardHyperParams, TensorElementType::Float32,
            {
                cfg.numPBTPolicies,
                sizeof(RewardHyperParams) / sizeof(float),
            }, cfg.gpuID);
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static void loadRenderObjects(render::RenderManager &render_mgr,
                              Vector3 *team_colors)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::AgentTeam0] =
        (std::filesystem::path(DATA_DIR) / "car_render.obj").string();
    render_asset_paths[(size_t)SimObject::AgentTeam1] =
        (std::filesystem::path(DATA_DIR) / "car_render.obj").string();

    render_asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { math::Vector4::fromVector3(team_colors[0], 0.f), -1, 0.8f, 0.2f },
        { math::Vector4::fromVector3(team_colors[1], 0.f), -1, 0.8f, 0.2f },
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,},
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
        { render::rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.9f, 0.1f },
    });

    // Override materials
    render_assets->objects[(CountT)SimObject::AgentTeam0].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::AgentTeam1].meshes[0].materialIDX = 1;

    render_assets->objects[(CountT)SimObject::Wall].meshes[0].materialIDX = 3;

    render_assets->objects[(CountT)SimObject::Sphere].meshes[0].materialIDX = 9;
    render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 6;

    render_mgr.loadObjects(render_assets->objects, materials, {
        { (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) /
           "smile.png").string().c_str() },
    });

    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    std::array<std::string, (size_t)SimObject::NumObjects - 1> asset_paths;
    asset_paths[(size_t)SimObject::AgentTeam0] =
        (std::filesystem::path(DATA_DIR) / "car_collision.obj").string();
    asset_paths[(size_t)SimObject::AgentTeam1] =
        (std::filesystem::path(DATA_DIR) / "car_collision.obj").string();
    asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObject::Sphere] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects - 1> asset_cstrs;
    for (size_t i = 0; i < asset_paths.size(); i++) {
        asset_cstrs[i] = asset_paths[i].c_str();
    }

    char import_err_buffer[4096];
    auto imported_src_hulls = imp::ImportedAssets::importFromDisk(
        asset_cstrs, import_err_buffer, true);

    if (!imported_src_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_src_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(
        (CountT)SimObject::NumObjects);

    auto setupHull = [&](SimObject obj_id,
                         float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)obj_id].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[(CountT)obj_id] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    setupHull(SimObject::AgentTeam0, 1.f, {
        .muS = 2.f,
        .muD = 2.f,
    });

    setupHull(SimObject::AgentTeam1, 1.f, {
        .muS = 2.f,
        .muD = 2.f,
    });

    setupHull(SimObject::Wall, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::Sphere, 1.f, {
        .muS = 0.01f,
        .muD = 0.01f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    src_objs[(CountT)SimObject::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 2.f,
            .muD = 2.f,
        },
    };

    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // This is a bit hacky, but in order to make sure the agents
    // remain controllable by the policy, they are only allowed to
    // rotate around the Z axis (infinite inertia in x & y axes)
    rigid_body_assets.metadatas[
        (CountT)SimObject::AgentTeam0].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObject::AgentTeam0].mass.invInertiaTensor.y = 0.f;

    rigid_body_assets.metadatas[
        (CountT)SimObject::AgentTeam1].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObject::AgentTeam1].mass.invInertiaTensor.y = 0.f;

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

Manager::Impl * Manager::Impl::make(
    const Manager::Config &mgr_cfg)
{
    RandKey init_key = rand::initKey(mgr_cfg.randSeed);
    RandKey sim_init_key = rand::split_i(init_key, 0);
    RandKey stagger_shuffle_key = rand::split_i(init_key, 1);

    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.initRandKey = sim_init_key;
    sim_cfg.flags = mgr_cfg.simFlags;

    madrona::math::Vector3 teamColors[] = {
        Vector3{ 1.f, 0.f, 0.f },
        Vector3{ 0.f, 0.f, 1.f },
    };

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        PhysicsLoader phys_loader(ExecMode::CUDA, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        if (mgr_cfg.numPBTPolicies > 0) {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)cu::allocGPU(
                sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
        } else {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)cu::allocGPU(
                sizeof(RewardHyperParams));

            RewardHyperParams default_reward_hyper_params {
                .teamSpirit = 0.f,
            };

            REQ_CUDA(cudaMemcpy(sim_cfg.rewardHyperParams,
                &default_reward_hyper_params, sizeof(RewardHyperParams),
                cudaMemcpyHostToDevice));
        }

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr, teamColors);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            agent_actions_buffer,
            sim_cfg.rewardHyperParams,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            stagger_shuffle_key,
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        (void)stagger_shuffle_key;

        PhysicsLoader phys_loader(ExecMode::CPU, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        if (mgr_cfg.numPBTPolicies > 0) {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)malloc(
                sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
        } else {
            sim_cfg.rewardHyperParams = (RewardHyperParams *)malloc(
                sizeof(RewardHyperParams));

            *(sim_cfg.rewardHyperParams) = {
                .teamSpirit = 0.f,
            };
        }

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr, teamColors);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
            (uint32_t)TaskGraphID::NumTaskGraphs,
        };

        WorldReset *world_reset_buffer = 
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            agent_actions_buffer,
            sim_cfg.rewardHyperParams,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(cpu_exec),
        };

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::make(cfg))
{}

Manager::~Manager() {}

void Manager::init()
{
    impl_->init();

    if ((impl_->cfg.simFlags & SimFlags::StaggerStarts) ==
            SimFlags::StaggerStarts) {
        assert(false);
    }

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

void Manager::step()
{
    impl_->step();

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuStreamInit(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamInit(strm, buffers, *this);

    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}

void Manager::gpuStreamStep(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamStep(strm, buffers, *this);

    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}
#endif

TrainInterface Manager::trainInterface() const
{
    auto pbt_inputs = std::to_array<NamedTensorInterface>({
        { "policy_assignments", policyAssignmentsTensor().interface() },
        { "reward_hyper_params", rewardHyperParamsTensor().interface() },
    });

    return TrainInterface {
        {
            .actions = actionTensor().interface(),
            .resets = resetTensor().interface(),
            .pbt = impl_->cfg.numPBTPolicies > 0 ?
                pbt_inputs : Span<const NamedTensorInterface>(nullptr, 0),
        },
        {
            .observations = {
                { "self", selfObservationTensor().interface() },
                { "my_goal", myGoalObservationTensor().interface() },
                { "enemy_goal", enemyGoalObservationTensor().interface() },
                { "team", teamObservationTensor().interface() },
                { "enemy", enemyObservationTensor().interface() },
                { "ball", ballTensor().interface() },
                { "stepsRemaining", stepsRemainingTensor().interface() },
            },
            .rewards = rewardTensor().interface(),
            .dones = doneTensor().interface(),
            .pbt = {
                { "episode_results", episodeResultTensor().interface() },
            },
        },
    };
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::episodeResultTensor() const
{
    return impl_->exportTensor(ExportID::EpisodeResult,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(EpisodeResult) / sizeof(int32_t),
                               });
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds * consts::numAgents,
            sizeof(Action) / sizeof(int32_t),
        });
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   sizeof(SelfObservation) / sizeof(float),
                               });
}

Tensor Manager::myGoalObservationTensor() const
{
    return impl_->exportTensor(ExportID::MyGoalObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                                   sizeof(MyGoalObservation) / sizeof(float),
                               });
}

Tensor Manager::enemyGoalObservationTensor() const
{
    return impl_->exportTensor(ExportID::EnemyGoalObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                                   sizeof(EnemyGoalObservation) / sizeof(float),
                               });
}

Tensor Manager::ballTensor() const
{
    return impl_->exportTensor(ExportID::BallObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                                   sizeof(BallObservation) / sizeof(float),
                               });
}

Tensor Manager::teamObservationTensor() const
{
    return impl_->exportTensor(ExportID::TeamObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::numCarsPerTeam - 1,
                                   sizeof(OtherObservation) / sizeof(float),
                               });
}

Tensor Manager::enemyObservationTensor() const
{
    return impl_->exportTensor(ExportID::EnemyObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::numCarsPerTeam,
                                   sizeof(OtherObservation) / sizeof(float),
                               });
}

Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                               });
}

Tensor Manager::policyAssignmentsTensor() const
{
    return impl_->exportTensor(ExportID::CarPolicy,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                               });
}

Tensor Manager::loadCheckpointTensor() const
{
    return impl_->exportTensor(ExportID::LoadCheckpoint,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::checkpointTensor() const
{
    return impl_->exportTensor(ExportID::Checkpoint,
                               TensorElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(Checkpoint),
                               });
}

Tensor Manager::rewardHyperParamsTensor() const
{
    return impl_->rewardHyperParamsTensor();
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t rotate)
{
    Action action { 
        .moveAmount = move_amount,
        .rotate = rotate
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * consts::numAgents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
