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
    registry.registerComponent<MyGoalObservation>();
    registry.registerComponent<EnemyGoalObservation>();
    registry.registerComponent<TeamObservation>();
    registry.registerComponent<EnemyObservation>();
    registry.registerComponent<BallObservation>();

    registry.registerComponent<Lidar>();
    registry.registerComponent<StepsRemainingObservation>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<BallGoalState>();
    registry.registerComponent<DynamicEntityType>();
    registry.registerComponent<CollisionData>();
    registry.registerComponent<TeamState>();
    registry.registerComponent<CarBallTouchState>();
    registry.registerComponent<CarPolicy>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<MatchInfo>();
    registry.registerSingleton<EpisodeResult>();
    registry.registerSingleton<TeamRewardState>();
    registry.registerSingleton<SimFlags>();

    registry.registerSingleton<LoadCheckpoint>();
    registry.registerSingleton<Checkpoint>();

    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<Car>();
    registry.registerArchetype<Ball>();
    registry.registerArchetype<Collision>();

    registry.exportSingleton<WorldReset>((uint32_t)ExportID::Reset);
    registry.exportSingleton<EpisodeResult>((uint32_t)ExportID::EpisodeResult);

    registry.exportSingleton<LoadCheckpoint>(
        (uint32_t)ExportID::LoadCheckpoint);
    registry.exportSingleton<Checkpoint>(
        (uint32_t)ExportID::Checkpoint);

    registry.exportColumn<Car, BallObservation>(
        (uint32_t)ExportID::BallObservation);
    registry.exportColumn<Car, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Car, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<Car, MyGoalObservation>(
        (uint32_t)ExportID::MyGoalObservation);
    registry.exportColumn<Car, EnemyGoalObservation>(
        (uint32_t)ExportID::EnemyGoalObservation);
    registry.exportColumn<Car, TeamObservation>(
        (uint32_t)ExportID::TeamObservation);
    registry.exportColumn<Car, EnemyObservation>(
        (uint32_t)ExportID::EnemyObservation);
    registry.exportColumn<Car, StepsRemainingObservation>(
        (uint32_t)ExportID::StepsRemaining);
    registry.exportColumn<Car, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Car, Done>((uint32_t)ExportID::Done);

    registry.exportColumn<Car, CarPolicy>(
        (uint32_t)ExportID::CarPolicy);
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

    ctx.singleton<MatchInfo>().stepsRemaining = consts::episodeLen;
    // Defined in src/level_gen.hpp / src/level_gen.cpp
    generateWorld(ctx);
}

inline void matchInfoSystem(Engine &, MatchInfo &match_info)
{
    match_info.stepsRemaining -= 1;
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
        const MatchInfo &match_info = ctx.singleton<MatchInfo>();

        if (match_info.stepsRemaining == 0) {
            should_reset = 1;
        }
    }

    BallGoalState &ball_gs = ctx.get<BallGoalState>(ctx.data().ball);

    if (should_reset != 0) {
        reset.reset = 0;

        cleanupWorld(ctx);
        initWorld(ctx);
    } else if (ball_gs.state == BallGoalState::State::InGoal) {
        placeEntities(ctx);
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
    touch_state.touched = false;

    constexpr float move_angle_per_bucket =
        3.f * math::pi / float(consts::numTurnBuckets);
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

            touch_state.touched = true;
        }
    }


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

    float goal_post_rad = (consts::worldWidth / 3.f + consts::wallWidth*2.f)/2.f;

    if (goal_idx == 0 && pos.y < -consts::worldLength*0.5f) {
        if (pos.x > (-consts::worldWidth/3.f + goal_post_rad) &&
            pos.x < (+consts::worldWidth/3.f - goal_post_rad)) {
            ball_gs.state = BallGoalState::State::InGoal;
            ball_gs.data = 0;
        }
    } else if (goal_idx == 1 && pos.y > consts::worldLength*0.5f) {
        if (pos.x > (-consts::worldWidth/3.f + goal_post_rad) &&
            pos.x < (+consts::worldWidth/3.f - goal_post_rad)) {
            ball_gs.state = BallGoalState::State::InGoal;
            ball_gs.data = 1;
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
                                  DynamicEntityType type,
                                  Velocity &vel)
{
    (void)engine;

    // Friction hack
    if (type == DynamicEntityType::Ball) {
        vel.linear *= 0.99f;
    } else {
        vel.linear *= 0.9f;
    }
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

static inline PolarObservation xyzToPolar(Vector3 v)
{
    float r = v.length();

    if (r < 1e-5f) {
        return PolarObservation {
            .r = 0.f,
            .theta = 0.f,
            .phi = 0.f,
        };
    }

    v /= r;

    // Note that this is angle off y-forward
    float theta = -atan2f(v.x, v.y);

    // This clamp is necessary on GPU due to sqrt / division
    // approximation
    float phi = asinf(std::clamp(v.z, -1.f, 1.f));

    return PolarObservation {
        .r = distObs(r),
        .theta = angleObs(theta),
        .phi = angleObs(phi),
    };
}

inline void collectCarObservationSystem(
    Engine &ctx,
    Entity e,
    Position pos,
    Rotation rot,
    Velocity vel,
    SelfObservation &self_obs,
    MyGoalObservation &my_goal_obs,
    EnemyGoalObservation &enemy_goal_obs,
    TeamObservation &team_obs,
    EnemyObservation &enemy_obs,
    BallObservation &ball_obs,
    StepsRemainingObservation &steps_remaining_ob,
    TeamState team_state)
{
    const Team &my_team = ctx.data().teams[team_state.teamIdx];
    const Team &enemy_team = ctx.data().teams[team_state.teamIdx ^ 1];

    const Goal &my_goal = ctx.data().arena.goals[my_team.goalIdx];
    const Goal &enemy_goal = ctx.data().arena.goals[enemy_team.goalIdx];

    {
        Vector3 goal_relative_pos = pos - my_goal.centerPosition;
        float z_rot = computeZAngle(rot);

        Vector3 linear_vel = vel.linear;

        // Reflect observations so they're symmetrical
        if (my_team.goalIdx == 0) {
            goal_relative_pos.x *= -1;
            goal_relative_pos.y *= -1;

            if (z_rot > 0) {
                z_rot -= math::pi;
            } else {
                z_rot += math::pi;
            }

            linear_vel.x *= -1;
            linear_vel.y *= -1;
        }

        self_obs.x = globalPosObs(goal_relative_pos.x);
        self_obs.y = globalPosObs(goal_relative_pos.y);
        self_obs.z = globalPosObs(goal_relative_pos.z);
        self_obs.theta = angleObs(z_rot);
        self_obs.vel = xyzToPolar(linear_vel);
    }

    Quat to_view = rot.inv();

    {
        Vector3 to_my_goal = my_goal.centerPosition - pos;
        my_goal_obs.pos = xyzToPolar(to_view.rotateVec(to_my_goal));

        Vector3 to_enemy_goal = enemy_goal.centerPosition - pos;
        enemy_goal_obs.pos = xyzToPolar(to_view.rotateVec(to_enemy_goal));
    }

    // Handle team observations next
    for (int i = 0, obs_idx = 0; i < consts::numCarsPerTeam; ++i) {
        // Only car about the other players
        if (my_team.players[i] != e) {
            Entity other_player = my_team.players[i];

            Vector3 other_pos = ctx.get<Position>(other_player);
            Rotation other_rot = ctx.get<Rotation>(other_player);
            Vector3 other_vel = ctx.get<Velocity>(other_player).linear;

            Vector3 to_other = other_pos - pos;

            OtherObservation &obs = team_obs.obs[obs_idx];
            obs.polar = xyzToPolar(to_view.rotateVec(to_other));
            obs.o_theta = angleObs(computeZAngle(
                (to_view * other_rot).normalize()));
            obs.vel = xyzToPolar(to_view.rotateVec(other_vel));

            ++obs_idx;
        }
    }

    // Handle the enemy team
    for (int i = 0; i < consts::numCarsPerTeam; ++i) {
        Entity enemy_player = enemy_team.players[i];

        Vector3 enemy_pos = ctx.get<Position>(enemy_player);
        Rotation enemy_rot = ctx.get<Rotation>(enemy_player);
        Vector3 enemy_vel = ctx.get<Velocity>(enemy_player).linear;

        Vector3 to_enemy = enemy_pos - pos;

        OtherObservation &obs = enemy_obs.obs[i];
        obs.polar = xyzToPolar(to_view.rotateVec(to_enemy));
        obs.o_theta = angleObs(computeZAngle(
            (to_view * enemy_rot).normalize()));
        obs.vel = xyzToPolar(to_view.rotateVec(enemy_vel));
    }

    {
        Entity ball_entity = ctx.data().ball;
        Vector3 ball_pos = ctx.get<Position>(ball_entity);
        Vector3 ball_vel = ctx.get<Velocity>(ball_entity).linear;

        Vector3 to_ball = ball_pos - pos;

        ball_obs.pos = xyzToPolar(to_view.rotateVec(to_ball));
        ball_obs.vel = xyzToPolar(ball_vel);
    }

    steps_remaining_ob.t = ctx.singleton<MatchInfo>().stepsRemaining;
}

inline void updateResultsSystem(Engine &ctx,
                                EpisodeResult &episode_result)
{
    const MatchInfo &match_info = ctx.singleton<MatchInfo>();

    if (match_info.stepsRemaining == consts::episodeLen - 1) {
        episode_result.numTeamAGoals = 0;
        episode_result.numTeamBGoals = 0;
    }

    Entity ball_entity = ctx.data().ball;
    BallGoalState &ball_gs = ctx.get<BallGoalState>(ball_entity);

    if (ball_gs.state == BallGoalState::State::InGoal) {
        if (ball_gs.data == ctx.data().teams[0].goalIdx) {
            episode_result.numTeamAGoals += 1;
        } else {
            episode_result.numTeamBGoals += 1;
        }
    }

    if (match_info.stepsRemaining == 0) {
        if (episode_result.numTeamAGoals > episode_result.numTeamBGoals) {
            episode_result.winResult = 0;
        } else if (episode_result.numTeamBGoals > episode_result.numTeamAGoals) {
            episode_result.winResult = 1;
        } else {
            episode_result.winResult = 2;
        }
    }
}

inline void individualRewardSystem(
    Engine &ctx,
    Entity e,
    CarPolicy car_policy,
    Position pos,
    Rotation rot,
    TeamState team_state,
    CarBallTouchState touch_state,
    Reward &reward_out)
{
    float reward = 0.f;

    Entity ball_entity = ctx.data().ball;
    BallGoalState &ball_gs = ctx.get<BallGoalState>(ball_entity);

    const RewardHyperParams &reward_hyper_params = ctx.data().rewardHyperParams[
        car_policy.policyIdx];

    if (touch_state.touched == 1) {
        reward += 1.f * reward_hyper_params.hitRewardScale;
    }

    // 2) Goal scored
    if (ball_gs.state == BallGoalState::State::InGoal) {
        if (ball_gs.data == ctx.data().teams[team_state.teamIdx].goalIdx) {
            reward += 5.f;
        } else {
            reward -= 5.f;
        }
    }

    const MatchInfo &match_info = ctx.singleton<MatchInfo>();
    if (match_info.stepsRemaining == 0) {
        int32_t win_result = ctx.singleton<EpisodeResult>().winResult;

        if (win_result == 2) {
            reward -= 5.f;
        } else if (win_result == team_state.teamIdx) {
            reward += 10.f;
        } else {
            reward -= 10.f;
        }
    }

    reward_out.v = reward;

#if 0
#if 0
    Vector3 car_fwd = rot.rotateVec({0.f, 1.f, 0.f});

    // 1) Ball is in front of / close to the car
    Vector3 diff = ball_pos - pos;
    Vector3 diff_norm = normalize(diff);

    float cos_theta = diff_norm.dot(car_fwd);
    reward += cos_theta * 0.1f / (diff.length2() + 1.f);

#endif
    // 2) Ball was hit by a car in your team
    for (int i = 0; i < consts::numCarsPerTeam; ++i) {
        Entity player_entity = my_team.players[i];

        int32_t t = engine.get<CarBallTouchState>(player_entity).touched;
        if (t) {
            reward += 0.1f;
            break;
        }
    }
    
#if 0
    // 3) Ball was hit by the car
    if (touch_state.touched) {
        reward += 0.1f;
    }
#endif

    // 4) Goal scored
    if (ball_gs.state == BallGoalState::State::InGoal) {
        if (ball_gs.data == team_state.teamIdx) {
            reward += 5.f;
        } else {
            reward -= 5.f;
        }
    } else {
#if 0
        // 5) Try to keep ball towards the opponent goal
        // Team 0 is trying to score towards the -y direction
        // Team 1 is trying to score towards the +y direction
        
        constexpr float half_pitch_len = consts::worldLength / 2.f;

        float to_side_y;
        if (team_state.teamIdx == 0) {
            to_side_y = ball_pos.y + half_pitch_len;
        } else {
            to_side_y = half_pitch_len - ball_pos.y;
        }
        
        // Not on the desired side
        //float side_reward_sign = 1.f;
        //if (to_side_y >= half_pitch_len) {
        //    to_side_y -= half_pitch_len;
        //    side_reward_sign = -1.f;
        //}

        float side_reward = (0.05f / half_pitch_len) * 
            (half_pitch_len - to_side_y);

        reward += side_reward;
#endif
    }

    reward_out.v = reward;
#endif
}

inline void teamRewardSystem(Engine &ctx,
                             TeamRewardState &team_reward_state)
{
    for (CountT team_idx = 0; team_idx < consts::numTeams; team_idx++) {
        const Team &team = ctx.data().teams[team_idx];

        float reward_sum = 0.f;
        for (CountT car_idx = 0; car_idx < consts::numCarsPerTeam; car_idx++) {
            Entity car = team.players[car_idx];

            float car_reward = ctx.get<Reward>(car).v;

            reward_sum += car_reward;
        }

        float avg_reward = reward_sum / float(consts::numCarsPerTeam);
        team_reward_state.teamRewards[team_idx] = avg_reward;
    }
}

inline void finalRewardSystem(Engine &ctx,
                              TeamState team,
                              const CarPolicy &car_policy,
                              Reward &reward)
{
    float my_reward = reward.v;

    TeamRewardState &team_rewards = ctx.singleton<TeamRewardState>();
    float team_reward = team_rewards.teamRewards[team.teamIdx];
    float other_team_reward = team_rewards.teamRewards[team.teamIdx ^ 1];

    const RewardHyperParams &reward_hyper_params = ctx.data().rewardHyperParams[
        car_policy.policyIdx];

    float team_spirit = reward_hyper_params.teamSpirit;

    reward.v = my_reward * (1.f - team_spirit) + team_reward * team_spirit -
        other_team_reward;
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void writeDonesSystem(Engine &ctx,
                             Done &done)
{
    int32_t steps_remaining = ctx.singleton<MatchInfo>().stepsRemaining;
    if (steps_remaining == 0 || ctx.singleton<WorldReset>().reset == 1) {
        done.v = 1;
    } else if (steps_remaining == consts::episodeLen - 1) {
        done.v = 0;
    }
}

inline void loadCheckpointSystem(Engine &ctx, const Checkpoint &ckpt)
{
    LoadCheckpoint should_load = ctx.singleton<LoadCheckpoint>();
    if (!should_load.load) {
        return;
    }

    should_load.load = 0;

    for (CountT i = 0; i < 2; i++) {
        Team &team = ctx.data().teams[i];
        const Checkpoint::TeamData &team_ckpt = ckpt.teams[i];

        for (CountT j = 0; j < 3; j++) {
            const Checkpoint::CarData &car_ckpt = team_ckpt.cars[j];

            Entity car = team.players[j];
            ctx.get<Position>(car) = car_ckpt.position;
            ctx.get<Rotation>(car) = car_ckpt.rotation;
            ctx.get<Velocity>(car) = car_ckpt.velocity;
        }

        team.goalIdx = team_ckpt.goalIdx;
    }

    {
        const Checkpoint::BallData &ball_ckpt = ckpt.ball;

        Entity ball = ctx.data().ball;
        ctx.get<Position>(ball) = ball_ckpt.position;
        ctx.get<Rotation>(ball) = ball_ckpt.rotation;
        ctx.get<Velocity>(ball) = ball_ckpt.velocity;
    }

    {
        MatchInfo &match_info = ctx.singleton<MatchInfo>();
        EpisodeResult &episode_result = ctx.singleton<EpisodeResult>();

        match_info.stepsRemaining = ckpt.stepsRemaining;
        episode_result.numTeamAGoals = ckpt.numTeamAGoals;
        episode_result.numTeamBGoals = ckpt.numTeamBGoals;
    }
}

inline void checkpointSystem(Engine &ctx, Checkpoint &ckpt)
{
    for (CountT i = 0; i < 2; i++) {
        const Team &team = ctx.data().teams[i];
        Checkpoint::TeamData &team_ckpt = ckpt.teams[i];

        for (CountT j = 0; j < 3; j++) {
            Entity car = team.players[j];

            Checkpoint::CarData &car_ckpt = team_ckpt.cars[j];
            car_ckpt.position = ctx.get<Position>(car);
            car_ckpt.rotation = ctx.get<Rotation>(car);
            car_ckpt.velocity = ctx.get<Velocity>(car);
        }

        team_ckpt.goalIdx = team.goalIdx;
    }

    {
        Entity ball = ctx.data().ball;

        Checkpoint::BallData &ball_ckpt = ckpt.ball;
        ball_ckpt.position = ctx.get<Position>(ball);
        ball_ckpt.rotation = ctx.get<Rotation>(ball);
        ball_ckpt.velocity = ctx.get<Velocity>(ball);
    }

    {
        const MatchInfo &match_info = ctx.singleton<MatchInfo>();
        const EpisodeResult &episode_result = ctx.singleton<EpisodeResult>();

        ckpt.stepsRemaining = match_info.stepsRemaining;
        ckpt.numTeamAGoals = episode_result.numTeamAGoals;
        ckpt.numTeamBGoals = episode_result.numTeamBGoals;
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

static TaskGraphNodeID gameplayAndRewardsTasks(TaskGraphBuilder &builder,
                                               const Sim::Config &cfg,
                                               Span<const TaskGraphNodeID> deps)
{
    (void)cfg;

    auto match_info_sys = builder.addToGraph<ParallelForNode<Engine,
        matchInfoSystem,
            MatchInfo
        >>(deps);

    // Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        carMovementSystem,
            Entity,
            Action,
            Position,
            Rotation,
            Velocity,
            CarBallTouchState
        >>({match_info_sys});

    auto ball_move_sys = builder.addToGraph<ParallelForNode<Engine,
        ballMovementSystem,
            Entity,
            Position,
            Velocity,
            BallGoalState
        >>({move_sys});

    auto post_collisions = ball_move_sys;

#ifdef MADRONA_GPU_MODE
    auto sort_collisions = queueSortByWorld<Collision>(builder, {ball_move_sys});
    post_collisions = sort_collisions;
#endif

    auto collision_resolve = builder.addToGraph<ParallelForNode<Engine,
         collisionResolveSystem,
            WorldReset
        >>({post_collisions});

    auto clear_colisions = builder.addToGraph<ClearTmpNode<Collision>>(
        {collision_resolve});

    auto velocity_correct_system = builder.addToGraph<ParallelForNode<Engine,
         velocityCorrectSystem,
            DynamicEntityType,
            Velocity
        >>({clear_colisions});

    auto update_results_sys = builder.addToGraph<ParallelForNode<Engine,
        updateResultsSystem,
            EpisodeResult
        >>({velocity_correct_system});

    auto individual_reward_sys = builder.addToGraph<ParallelForNode<Engine,
        individualRewardSystem,
            Entity,
            CarPolicy,
            Position,
            Rotation,
            TeamState,
            CarBallTouchState,
            Reward
        >>({update_results_sys});

    auto team_reward_sys = builder.addToGraph<ParallelForNode<Engine,
        teamRewardSystem,
            TeamRewardState
        >>({individual_reward_sys});

    auto final_reward_sys = builder.addToGraph<ParallelForNode<Engine,
        finalRewardSystem,
            TeamState,
            CarPolicy,
            Reward
        >>({team_reward_sys});

    // Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        writeDonesSystem,
            Done
        >>({final_reward_sys});

    return done_sys;
}

static TaskGraphNodeID resetTasks(TaskGraphBuilder &builder,
                                  const Sim::Config &cfg,
                                  Span<const TaskGraphNodeID> deps)
{
    (void)cfg;

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>(deps);

    auto load_ckpt_sys = builder.addToGraph<ParallelForNode<Engine,
        loadCheckpointSystem,
            Checkpoint
        >>({reset_sys});

    auto ckpt_sys = builder.addToGraph<ParallelForNode<Engine,
        checkpointSystem,
            Checkpoint
        >>({load_ckpt_sys});

    return ckpt_sys;
}

static void obsTasks(TaskGraphBuilder &builder,
                     const Sim::Config &cfg,
                     Span<const TaskGraphNodeID> deps)
{
    builder.addToGraph<ParallelForNode<Engine,
         collectCarObservationSystem,
            Entity,
            Position,
            Rotation,
            Velocity,
            SelfObservation,
            MyGoalObservation,
            EnemyGoalObservation,
            TeamObservation,
            EnemyObservation,
            BallObservation,
            StepsRemainingObservation,
            TeamState
        >>(deps);

    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, deps);
    }
}

static void setupInitTasks(TaskGraphBuilder &builder,
                           const Sim::Config &cfg)
{
#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto gpu_init = builder.addToGraph<RecycleEntitiesNode>({});

    gpu_init = queueSortByWorld<Car>(builder, {gpu_init});
    gpu_init = queueSortByWorld<Ball>(builder, {gpu_init});
    gpu_init = queueSortByWorld<PhysicsEntity>(builder, {gpu_init});
#endif

    auto reset = resetTasks(builder, cfg, {
#ifdef MADRONA_GPU_MODE
        gpu_init
#endif
    });

    obsTasks(builder, cfg, {reset});
}

static void setupStepTasks(TaskGraphBuilder &builder,
                           const Sim::Config &cfg)
{
    auto gameplay = gameplayAndRewardsTasks(builder, cfg, {});
    auto reset = resetTasks(builder, cfg, {gameplay});
    obsTasks(builder, cfg, {reset});
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Sim::Config &cfg)
{
    setupInitTasks(taskgraph_mgr.init(TaskGraphID::Init), cfg);
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    constexpr CountT max_total_entities =
        5 + consts::numTeams * consts::numCarsPerTeam;

    ctx.singleton<SimFlags>() = cfg.flags;

    ctx.singleton<LoadCheckpoint>().load = 0;

    phys::RigidBodyPhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities);

    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;
    rewardHyperParams = cfg.rewardHyperParams;

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

    ctx.singleton<WorldReset>().reset = 1;
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
