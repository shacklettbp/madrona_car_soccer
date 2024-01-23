#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "physics.hpp"
#include "level_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madEscape {

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerComponent<Action>();
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<GrabState>();
    registry.registerComponent<Progress>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<PartnerObservations>();
    registry.registerComponent<RoomEntityObservations>();
    registry.registerComponent<DoorObservation>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<OpenState>();
    registry.registerComponent<DoorProperties>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<BallGoalState>();
    registry.registerComponent<DynamicEntityType>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<DoorEntity>();
    registry.registerArchetype<ButtonEntity>();
    registry.registerArchetype<Car>();
    registry.registerArchetype<Ball>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);

    registry.exportColumn<Car, Action>(
        (uint32_t)ExportID::Action);

    registry.exportColumn<Agent, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);

    registry.exportColumn<Agent, PartnerObservations>(
        (uint32_t)ExportID::PartnerObservations);

    registry.exportColumn<Agent, RoomEntityObservations>(
        (uint32_t)ExportID::RoomEntityObservations);

    registry.exportColumn<Agent, DoorObservation>(
        (uint32_t)ExportID::DoorObservation);

    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);

    registry.exportColumn<Agent, StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);

    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);

    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
}

static inline void cleanupWorld(Engine &ctx)
{
    (void)ctx;
}

static inline void initWorld(Engine &ctx)
{
    phys::RigidBodyPhysicsSystem::reset(ctx);

    // Assign a new episode ID
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        ctx.data().curWorldEpisode++, (uint32_t)ctx.worldID().idx));

    // Defined in src/level_gen.hpp / src/level_gen.cpp
    generateWorld(ctx);
}

// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t should_reset = reset.reset;
    if (ctx.data().autoReset) {
        for (CountT i = 0; i < consts::numAgents; i++) {
            Entity agent = ctx.data().cars[i];
            Done done = ctx.get<Done>(agent);
            if (done.v) {
                should_reset = 1;
            }
        }
    }

    if (should_reset != 0) {
        reset.reset = 0;

        cleanupWorld(ctx);
        initWorld(ctx);
    }
}

// Translates discrete actions from the Action component to forces
// used by the physics simulation.
inline void carMovementSystem(Engine &engine,
                              Entity e,
                              Action &action, 
                              Position &pos,
                              Rotation &rot, 
                              Velocity &vel)
{
    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numTurnBuckets);
    float move_angle = float(action.rotate-2) * move_angle_per_bucket *
                       consts::deltaT;
    Quat rot_diff = Quat::angleAxis(move_angle, { 0.0f, 0.0f, 1.0f });

    rot *= rot_diff;

    Vector3 fwd = rot.rotateVec({ 0.f, 1.f, 0.f });

    // Calculate the uninterupted displacement vector, and velocity.
    if (action.moveAmount > 0) {
        vel.linear += consts::carAcceleration * fwd * consts::deltaT;
    }

    // Hack friction
    vel.linear *= 0.95f;
    
    // This is the uninterrupted displacement vector given no collisions.
    Vector3 dx = vel.linear * consts::deltaT;

    

    // First check collision with the ball
    Entity ball_entity = engine.data().ball;
    Position ball_pos = engine.get<Position>(ball_entity);
    Velocity ball_vel = engine.get<Velocity>(ball_entity);


    Vector3 ball_dx = ball_vel.linear * consts::deltaT;
    Sphere ball_sphere = { Vector3::zero(), consts::ballRadius };

    Vector3 car_ball_rel = rot.inv().rotateVec(pos - ball_pos);
    AABB car_aabb = { car_ball_rel - consts::agentDimensions,
                      car_ball_rel + consts::agentDimensions };

    Vector3 rel_dx = rot.inv().rotateVec(dx - ball_dx);

    float out_t;
    int intersect = intersectMovingSphereAABB(
        ball_sphere, rel_dx, 
        car_aabb, out_t);

    if (intersect) {
        // Take the difference of the sphere's center and the car's
        // center at impact.
        Vector3 diff = rot.rotateVec(out_t * rel_dx - car_ball_rel);

        engine.get<Velocity>(ball_entity).linear = diff * 10.0f;
    }

    // For now, we just naively loop through the other agents, and then 
    // the ball to determine where collisions have happened.
    //
    // (TODO) Add some very simple space partitioning system to make this
    // faster when there are multiple cars.
    

    pos += dx;


    auto create_obb = [](Position pos, Rotation rot) -> OBB {
        static Vector3 car_ground_verts[4] = {
            Vector3{ -consts::agentDimensions.x, consts::agentDimensions.y, 0.f },
            Vector3{ consts::agentDimensions.x, consts::agentDimensions.y, 0.f },
            Vector3{ consts::agentDimensions.x, -consts::agentDimensions.y, 0.f },
            Vector3{ -consts::agentDimensions.x, -consts::agentDimensions.y, 0.f }
        };

        OBB obb = {};

        for (int i = 0; i < 4; ++i) {
            auto v = rot.rotateVec(car_ground_verts[i]) + 
                     Vector3{pos.x, pos.y, 0.f};
            obb.verts[i] = { v.x, v.y };
        }

        return obb;
    };


    OBB e_obb = create_obb(pos, rot);

    // Check the other cars for collisions
    for (int i = 0; i < 2; ++i) {
        Entity car = engine.data().cars[i];

        if (car != e && car.id < e.id) {
            // Check for collision
            OBB car_obb = create_obb(engine.get<Position>(car),
                                     engine.get<Rotation>(car));

            float min_overlap;
            Vector2 min_overlap_axis;
            if (intersectMovingOBBs2D(e_obb, car_obb,
                                      min_overlap, min_overlap_axis)) {

                Vector3 diff = 0.5f * min_overlap * 
                    Vector3{min_overlap_axis.x, min_overlap_axis.y, 0.f};

                pos -= diff;
                vel.linear -= diff / consts::deltaT;

                engine.get<Position>(car) += diff;            
                engine.get<Velocity>(car).linear += diff / consts::deltaT;
            }
        }
    }

    // Check the walls for collisions
    for (int i = 0; i < 4; ++i) {
        auto &plane = engine.data().arena.wallPlanes[i];

        float overlap;
        if (intersectMovingOBBWall(e_obb, plane, overlap)) {
            pos -= overlap * Vector3{plane.normal.x, plane.normal.y, 0.0f};
        }
    }
}

inline void ballMovementSystem(Engine &engine,
                               Position &pos,
                               Velocity &vel,
                               BallGoalState &ball_goal_state)
{
    (void)engine;
    (void)ball_goal_state;

    Vector3 dx = vel.linear * consts::deltaT;

    for (int i = 0; i < 4; ++i) {

        WallPlane &plane = engine.data().arena.wallPlanes[i];

        float t;
        Vector3 p;
        if (intersectMovingSphereWall(Sphere{ pos, consts::ballRadius },
                dx, plane, t, p)) {
            dx = t * dx;

            printf("Ball intersection with wall\n");

            vel.linear *= -1.f;

            // vel.linear = reflect(vel.linear, 
                    // Vector3{plane.normal.x, plane.normal.y, 0.f});
        }
    }

    pos += dx;

    vel.linear *= 0.95f;
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void stepTrackerSystem(Engine &,
                              StepsRemaining &steps_remaining,
                              Done &done)
{
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1) {
        done.v = 0;
    } else if (num_remaining == 0) {
        done.v = 1;
    }
}

// Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =

        ctx.get<StepsRemaining>(car_).t = consts::episodeLen;
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

// Build the task graph
void Sim::setupTasks(TaskGraphBuilder &builder, const Config &cfg)
{
    // Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        carMovementSystem,
            Entity,
            Action,
            Position,
            Rotation,
            Velocity
        >>({});

    auto ball_move_sys = builder.addToGraph<ParallelForNode<Engine,
        ballMovementSystem,
            Position,
            Velocity,
            BallGoalState
        >>({move_sys});

    // Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        stepTrackerSystem,
            StepsRemaining,
            Done
        >>({ball_move_sys});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({done_sys});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});
    (void)clear_tmp;

#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_sys});
    (void)recycle_sys;
#endif

    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, {clear_tmp});
    }
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    // Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    constexpr CountT max_total_entities = consts::numAgents +
        consts::numRooms * (consts::maxEntitiesPerRoom + 3) +
        4; // side walls + floor

    phys::RigidBodyPhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities, max_total_entities * max_total_entities / 2,
        consts::numAgents);

    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;

    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    curWorldEpisode = 0;

    // Creates agents, walls, etc.
    createPersistentEntities(ctx);

    // Generate initial world state
    initWorld(ctx);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
