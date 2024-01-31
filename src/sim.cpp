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
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<TeamObservation>();
    registry.registerComponent<EnemyObservation>();
    registry.registerComponent<BallObservation>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<OpenState>();
    registry.registerComponent<DoorProperties>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<BallGoalState>();
    registry.registerComponent<DynamicEntityType>();
    registry.registerComponent<CollisionData>();
    registry.registerComponent<TeamState>();
    registry.registerComponent<CarBallTouchState>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<DoorEntity>();
    registry.registerArchetype<ButtonEntity>();
    registry.registerArchetype<Car>();
    registry.registerArchetype<Ball>();
    registry.registerArchetype<Collision>();

    registry.exportSingleton<WorldReset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Car, BallObservation>((uint32_t)ExportID::BallObservation);
    registry.exportColumn<Car, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Car, SelfObservation>((uint32_t)ExportID::SelfObservation);
    registry.exportColumn<Car, TeamObservation>((uint32_t)ExportID::TeamObservation);
    registry.exportColumn<Car, EnemyObservation>((uint32_t)ExportID::EnemyObservation);
    registry.exportColumn<Car, StepsRemaining>((uint32_t)ExportID::StepsRemaining);
    registry.exportColumn<Car, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Car, Done>((uint32_t)ExportID::Done);
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
        for (CountT team_idx = 0; team_idx < 2; ++team_idx) {
            Team &team = ctx.data().teams[team_idx];
            for (CountT i = 0; i < consts::numCarsPerTeam; i++) {
                Entity agent = team.players[i];
                Done done = ctx.get<Done>(agent);
                if (done.v) {
                    should_reset = 1;
                }
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
                              Velocity &vel,
                              CarBallTouchState &touch_state)
{
    touch_state.touched = 0;

    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numTurnBuckets);
    float move_angle = float(action.rotate-1) * move_angle_per_bucket *
                       consts::deltaT;
    Quat rot_diff = Quat::angleAxis(move_angle, { 0.0f, 0.0f, 1.0f });

    rot *= rot_diff;

    Vector3 fwd = rot.rotateVec({ 0.f, 1.f, 0.f });

    // Calculate the uninterupted displacement vector, and velocity.
    vel.linear += (float)(action.moveAmount-1) * 
        consts::carAcceleration * fwd * consts::deltaT;

    // Hack friction
    // vel.linear *= 0.95f;
    
    // This is the uninterrupted displacement vector given no collisions.
    Vector3 dx = vel.linear * consts::deltaT;
    pos += dx;
    
#if 0
    Entity ball_entity = engine.data().ball;
    Position ball_pos = engine.get<Position>(ball_entity);
    Sphere ball_sphere = { Vector3::zero(), consts::ballRadius };

    Velocity ball_vel = engine.get<Velocity>(ball_entity);
    Vector3 ball_dx = ball_vel.linear * consts::deltaT;

    Vector3 car_ball_rel = rot.inv().rotateVec(pos - ball_pos);
    AABB car_aabb = { car_ball_rel - consts::agentDimensions,
                      car_ball_rel + consts::agentDimensions };

    Vector3 rel_dx = rot.inv().rotateVec(dx - ball_dx);

    float out_t;
    Vector3 sphere_pos_out;
    int intersect = intersectMovingSphereAABB(
        ball_sphere, rel_dx, 
        car_aabb, out_t, sphere_pos_out);

    if (intersect) {
        // Take the difference of the sphere's center and the car's
        // center at impact.
        Vector3 diff = rot.rotateVec(out_t * rel_dx - car_ball_rel);
        Vector3 overlap = rot.rotateVec(sphere_pos_out);

        CollisionData collision = {
            .a = ball_entity,
            .b = e,
            .overlap = overlap,
            .diff = diff,
        };

        // TODO: Create collision entity
        Loc loc = engine.makeTemporary<Collision>();
        engine.get<CollisionData>(loc) = collision;

        touch_state.touched = 1;
    } else {
        // Possible, that the ball is inside the car
        if (ball_sphere.center.x >= car_aabb.pMin.x - ball_sphere.radius &&
            ball_sphere.center.x <= car_aabb.pMax.x + ball_sphere.radius &&
            ball_sphere.center.y >= car_aabb.pMin.y - ball_sphere.radius &&
            ball_sphere.center.y <= car_aabb.pMax.y + ball_sphere.radius) {
            printf("BALL INSIDE CAR!\n");
        }
    }
#else
    { // Determine collision with the ball
        Entity ball_e = engine.data().ball;
        Vector3 ball_pos = engine.get<Position>(ball_e);
        Vector3 ball_dx = engine.get<Velocity>(ball_e).linear * consts::deltaT;

        AABB car_aabb = { -consts::agentDimensions, consts::agentDimensions };

        // Transform the ball's position and velocities into the ball's frame
        Vector3 rel_ball_pos = rot.inv().rotateVec(ball_pos - pos);
        Vector3 rel_ball_dx = rot.inv().rotateVec(ball_dx/* - dx*/);

        Sphere ball_sphere = { rel_ball_pos, consts::ballRadius };

        float out_t;
        Vector3 sphere_pos_out;
        
        if (intersectMovingSphereAABB(ball_sphere, rel_ball_dx,
                                      car_aabb, out_t, sphere_pos_out)) {
            Vector3 diff = rot.rotateVec(sphere_pos_out.normalize());
            Vector3 overlap = rot.rotateVec(sphere_pos_out - rel_ball_pos);

            CollisionData collision = {
                .a = ball_e,
                .b = e,
                .overlap = overlap,
                .diff = diff,
            };

            // TODO: Create collision entity
            Loc loc = engine.makeTemporary<Collision>();
            engine.get<CollisionData>(loc) = collision;

            touch_state.touched = 1;
        }
    }
#endif


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
    for (CountT team_idx = 0; team_idx < 2; ++team_idx) {
        Team &team = engine.data().teams[team_idx];
        for (int i = 0; i < consts::numCarsPerTeam; ++i) {
            Entity car = team.players[i];

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

                    // TODO: Create collision data
                    CollisionData collision = {
                        .a = e,
                        .b = car,
                        .overlap = diff,
                        .diff = diff
                    };

                    Loc loc = engine.makeTemporary<Collision>();
                    engine.get<CollisionData>(loc) = collision;
                }
            }
        }
    }

    // Check the walls for collisions
    for (int i = 0; i < 4; ++i) {
        auto &plane = engine.data().arena.wallPlanes[i];

        float overlap;
        if (intersectMovingOBBWall(e_obb, plane, overlap)) {
            Vector3 overlap_vec = overlap *
                Vector3{plane.normal.x, plane.normal.y, 0.0f};

#if 0
            pos -= overlap_vec;
#endif

            CollisionData collision = {
                .a = e,
                .b = Entity::none(),
                .overlap = overlap_vec,
                .diff = overlap_vec
            };

            Loc loc = engine.makeTemporary<Collision>();
            engine.get<CollisionData>(loc) = collision;
        }
    }

#if 0
    vel.linear *= 0.95f;
#endif
}

inline void checkBallGoalPosts(Engine &engine,
                               Entity e,
                               Position &pos,
                               Velocity &vel,
                               BallGoalState &ball_gs,
                               int goal_idx)
{
    (void)ball_gs;

    float sign = (goal_idx == 0) ? +1.f : -1.f;

    Goal &goal = engine.data().arena.goals[goal_idx];

    for (int i = 0; i < 2; ++i) {
        Vector2 post0_pos = Vector2::fromVector3(
                engine.get<Position>(goal.outerBorders[i]));
        Scale post0_scale_3d = engine.get<Scale>(goal.outerBorders[i]);

        Vector2 post0_scale = { post0_scale_3d.d0, post0_scale_3d.d1 };

        WallSegment s0 = {
            { post0_pos + Vector2{ post0_scale.x/2.f, 0.f },
              post0_pos - Vector2{ post0_scale.x/2.f, 0.f }},
            Vector2{ 0.f, -sign }
        };

        float min_overlap;
        if (intersectSphereWallSeg(Sphere{ pos, consts::ballRadius },
                    s0, min_overlap)) {
            Vector3 normal_3d = Vector3{s0.normal.x, s0.normal.y, 0.f};

#if 0
            pos += normal_3d * min_overlap;
            vel.linear = reflect(vel.linear, normal_3d);
#endif

            CollisionData collision = {
                .a = e,
                .b = Entity::none(),
                .overlap = normal_3d * min_overlap,
                .diff = reflect(vel.linear, normal_3d),
            };

            Loc loc = engine.makeTemporary<Collision>();
            engine.get<CollisionData>(loc) = collision;
        }
    }
}

inline void ballMovementSystem(Engine &engine,
                               Entity e,
                               Position &pos,
                               Velocity &vel,
                               BallGoalState &ball_gs)
{
    (void)engine;
    (void)ball_gs;

    Vector3 dx = vel.linear * consts::deltaT;
    pos += dx;

    // Check collision against the long walls
    for (int i = 0; i < 2; ++i) {
        WallPlane &plane = engine.data().arena.wallPlanes[i];

        float min_overlap;
        if (intersectSphereWall(Sphere{pos, consts::ballRadius}, 
                    plane, min_overlap)) {
            Vector3 normal_3d = Vector3{plane.normal.x, plane.normal.y, 0.f};

#if 0
            pos += normal_3d * min_overlap;
            vel.linear = reflect(vel.linear, normal_3d);
#endif

            CollisionData collision = {
                .a = e,
                .b = Entity::none(),
                .overlap = normal_3d * min_overlap,
                .diff = reflect(vel.linear, normal_3d),
            };

            Loc loc = engine.makeTemporary<Collision>();
            engine.get<CollisionData>(loc) = collision;
        }
    }

    // Check collision against the goal posts
    for (int i = 0; i < 2; ++i) {
        checkBallGoalPosts(engine, e, pos, vel, ball_gs, i);
    }

#if 0
    vel.linear *= 0.95f;
#endif
}

inline void collisionResolveSystem(Engine &engine,
                                   WorldReset)
{
    auto resolve_collision = [&engine](CollisionData &data) {
        if (data.b == Entity::none()) {
            if (engine.get<DynamicEntityType>(data.a) == DynamicEntityType::Car) {
                // A is a car and B is a wall
                engine.get<Position>(data.a) -= data.overlap;
            } else {
                // A is a ball and B is a wall
                engine.get<Position>(data.a) += data.overlap;
                engine.get<Velocity>(data.a).linear = data.diff;
            }
        } else if (engine.get<DynamicEntityType>(data.a) == 
                DynamicEntityType::Ball) {
            // A is a ball, B is a car
            engine.get<Velocity>(data.a).linear += data.diff * 10.f;
            engine.get<Position>(data.a) += data.overlap;
        } else {
            // A is a car, B is a car
            engine.get<Position>(data.a) -= data.diff;
            engine.get<Velocity>(data.a).linear -= data.diff / consts::deltaT;

            engine.get<Position>(data.b) += data.diff;
            engine.get<Velocity>(data.b).linear += data.diff / consts::deltaT;
        }
    };

    engine.iterateQuery(engine.data().collisionQuery, resolve_collision);
}

inline void velocityCorrectSystem(Engine &engine,
                                  DynamicEntityType,
                                  Velocity &vel)
{
    (void)engine;

    // Friction hack
    vel.linear *= 0.9f;
}

static inline float angleObs(float v) { return v / math::pi; }
static inline float distObs(float v) { return v / consts::worldLength; }
static inline float globalPosObs(float v) { return v / consts::worldLength; }

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

static inline PolarObservation xyToPolar(Vector3 v)
{
    Vector2 xy { v.x, v.y };

    float r = xy.length();

    // Note that this is angle off y-forward
    float theta = atan2f(xy.x, xy.y);

    return PolarObservation {
        .r = distObs(r),
        .theta = angleObs(theta),
    };
}

inline void collectCarObservationSystem(Engine &engine,
                                        Entity e,
                                        Position pos,
                                        Rotation rot,
                                        Velocity vel,
                                        SelfObservation &self_obs,
                                        TeamObservation &team_obs,
                                        EnemyObservation &enemy_obs,
                                        BallObservation &ball_obs,
                                        TeamState team_state)
{
    // Handle self observation first
    self_obs.x = globalPosObs(pos.x);
    self_obs.y = globalPosObs(pos.y);
    self_obs.z = 0.f;
    self_obs.theta = angleObs(computeZAngle(rot));
    auto vel_polar = xyToPolar(vel.linear);
    self_obs.vel_r = vel_polar.r;
    self_obs.vel_theta = vel_polar.theta;

    Quat to_view = rot.inv();

    // Handle team observations next
    Team &my_team = engine.data().teams[team_state.teamIdx];
    for (int i = 0, obs_idx = 0; i < consts::numCarsPerTeam; ++i) {
        // Only car about the other players
        if (my_team.players[i] != e) {
            Entity other_player = my_team.players[i];

            Vector3 other_pos = engine.get<Position>(other_player);
            Rotation other_rot = engine.get<Rotation>(other_player);
            Vector3 to_other = other_pos - pos;

            OtherObservation &obs = team_obs.obs[obs_idx];
            obs.polar = xyToPolar(to_view.rotateVec(to_other));
            obs.o_theta = angleObs(computeZAngle(other_rot));

            ++obs_idx;
        }
    }

    // Handle the enemy team
    Team &other_team = engine.data().teams[team_state.teamIdx ^ 1];
    for (int i = 0; i < consts::numCarsPerTeam; ++i) {
        Entity other_player = other_team.players[i];

        Vector3 other_pos = engine.get<Position>(other_player);
        Rotation other_rot = engine.get<Rotation>(other_player);
        Vector3 to_other = other_pos - pos;

        OtherObservation &obs = enemy_obs.obs[i];
        obs.polar = xyToPolar(to_view.rotateVec(to_other));
        obs.o_theta = angleObs(computeZAngle(other_rot));
    }

    Entity ball_entity = engine.data().ball;
    Vector3 ball_pos = engine.get<Position>(ball_entity);

    ball_obs.x = ball_pos.x / consts::worldLength;
    ball_obs.y = ball_pos.y / consts::worldLength;
    ball_obs.z = 0.f;
}

inline void collectBallObservationSystem(Engine &engine,
                                         Entity e,
                                         Position pos,
                                         BallObservation &obs)
{
    (void)engine, (void)e;

    obs.x = pos.x / consts::worldLength;
    obs.y = pos.y / consts::worldLength;
    obs.z = 0.f;
}

inline void rewardSystem(Engine &engine,
                         Entity e,
                         Position pos,
                         Rotation rot,
                         TeamState team_state,
                         CarBallTouchState touch_state,
                         Reward &reward_out)
{
    (void)e;

    Vector3 car_fwd = rot.rotateVec({0.f, 1.f, 0.f});

    Team &my_team = engine.data().teams[team_state.teamIdx];

    // 1) Ball is in front of / close to the car
    float reward = 0.f;

    Entity ball_entity = engine.data().ball;
    Position ball_pos = engine.get<Position>(ball_entity);

    Vector3 diff = ball_pos - pos;
    Vector3 diff_norm = normalize(diff);

    float cos_theta = diff_norm.dot(car_fwd);
    reward += cos_theta * 0.1f / (diff.length2() + 1.f);

    // 2) Ball was hit by a car in your team
    for (int i = 0; i < consts::numCarsPerTeam; ++i) {
        Entity player_entity = my_team.players[i];

        int32_t t = engine.get<CarBallTouchState>(player_entity).touched;
        if (t) {
            reward += 0.1f;
            break;
        }
    }
    
    // 3) Ball was hit by the car
    if (touch_state.touched) {
        reward += 1.f;
    }

    // 4) Ball is close to enemy's goal / past the goal
    // Team 0 is trying to score towards the -y direction
    // Team 1 is trying to score towards the +y direction
    if (team_state.teamIdx == 0) {
        // Less negative is better
        float target_goal_y = (-consts::worldLength/2.f);
        float ball_diff_to_target = target_goal_y - ball_pos.y;
        float ball_diff_to_center = ball_pos.y;

        // The team won!
        if (ball_diff_to_target > 0.f) {
            reward += 10.f;
        } else {
            reward += 0.1f * ((1.f / (abs(ball_diff_to_target + 1.f))) - 
                    1.f / (abs(ball_diff_to_center) + 1.f));
        }
    } else {
        // Less positive is better
        float target_goal_y = consts::worldLength/2.f;
        float ball_diff_to_target = target_goal_y - ball_pos.y;
        float ball_diff_to_center = ball_pos.y;

        if (ball_diff_to_target < 0.f) {
            reward += 10.f;
        } else {
            reward += 0.1f * ((1.f / (abs(ball_diff_to_target + 1.f))) - 
                    1.f / (abs(ball_diff_to_center) + 1.f));
        }
    }

    reward_out.v = reward;
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
            Velocity,
            CarBallTouchState
        >>({});

    auto ball_move_sys = builder.addToGraph<ParallelForNode<Engine,
        ballMovementSystem,
            Entity,
            Position,
            Velocity,
            BallGoalState
        >>({move_sys});

#ifdef MADRONA_GPU_MODE
    auto sort_collisions = queueSortByWorld<Collision>(builder, {ball_move_sys});

    auto collision_resolve = builder.addToGraph<ParallelForNode<Engine,
         collisionResolveSystem,
            WorldReset
        >>({sort_collisions});
#else
    auto collision_resolve = builder.addToGraph<ParallelForNode<Engine,
         collisionResolveSystem,
            WorldReset
        >>({ball_move_sys});
#endif

    auto velocity_correct_system = builder.addToGraph<ParallelForNode<Engine,
         velocityCorrectSystem,
            DynamicEntityType,
            Velocity
        >>({collision_resolve});

    auto car_obs_system = builder.addToGraph<ParallelForNode<Engine,
         collectCarObservationSystem,
            Entity,
            Position,
            Rotation,
            Velocity,
            SelfObservation,
            TeamObservation,
            EnemyObservation,
            BallObservation,
            TeamState
        >>({velocity_correct_system});

    auto ball_obs_system = builder.addToGraph<ParallelForNode<Engine,
         collectBallObservationSystem,
            Entity,
            Position,
            BallObservation
        >>({car_obs_system});

    // Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        stepTrackerSystem,
            StepsRemaining,
            Done
        >>({ball_obs_system});

    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
        rewardSystem,
            Entity,
            Position,
            Rotation,
            TeamState,
            CarBallTouchState,
            Reward
        >>({done_sys});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({reward_sys});

    auto clear_temporaries = builder.addToGraph<ClearTmpNode<Collision>>({reset_sys});
    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({clear_temporaries});
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

    collisionQuery = ctx.query<CollisionData>();
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
