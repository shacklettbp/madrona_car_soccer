#include "level_gen.hpp"
#include "consts.hpp"

namespace madEscape {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace consts {

inline constexpr float doorWidth = consts::worldWidth / 3.f;

}

enum class RoomType : uint32_t {
    SingleButton,
    DoubleButton,
    CubeBlocking,
    CubeButtons,
    NumTypes,
};

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) =
        RigidBodyPhysicsSystem::registerEntity(ctx, e, obj_id);
}

// idx = 0 => Front
// idx = 1 => Back
static Goal makeGoal(Engine &ctx,
                     uint32_t idx)
{
    float sign = (idx == 0) ? +1.0f : -1.0f;

    Goal goal;

    goal.outerBorders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        goal.outerBorders[0],
        Vector3 {
            2.f * consts::worldWidth / 6.f,
            sign * (consts::wallWidth / 2.f + consts::worldLength / 2.f),
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth/3.f + consts::wallWidth * 2,
            consts::wallWidth,
            consts::wallHeight
        });

    goal.outerBorders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        goal.outerBorders[1],
        Vector3 {
            -2.f * consts::worldWidth / 6.f,
            sign * (consts::wallWidth / 2.f + consts::worldLength / 2.f),
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth/3.f + consts::wallWidth * 2,
            consts::wallWidth,
            consts::wallHeight
        });

    return goal;
}

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
    // Create the floor entity, just a simple static plane.
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Plane,
        EntityType::None, // Floor plane type should never be queried
        ResponseType::Static);

    // Create the outer wall entities
    // Right
    ctx.data().arena.longBorders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().arena.longBorders[0],
        Vector3 {
            consts::worldWidth / 2.f + consts::wallWidth / 2.f,
            0,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            consts::wallHeight
        });
    ctx.data().arena.wallPlanes[0] = {
        Vector2{ consts::worldWidth/2.f, 0.0f },
        Vector2{ -1.0f, 0.0f }
    };

    // Left
    ctx.data().arena.longBorders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().arena.longBorders[1],
        Vector3 {
            -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
            0,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            consts::wallHeight
        });
    ctx.data().arena.wallPlanes[1] = {
        Vector2{ -consts::worldWidth/2.f, 0.0f },
        Vector2{ 1.0f, 0.0f }
    };

    // Create the wall planes for the goals
    ctx.data().arena.wallPlanes[2] = {
        Vector2{ 0.0f, consts::worldLength/2.f },
        Vector2{ 0.0f, -1.0f }
    };
    ctx.data().arena.wallPlanes[3] = {
        Vector2{ 0.0f, -consts::worldLength/2.f },
        Vector2{ 0.0f, 1.0f }
    };

    ctx.data().arena.goals[0] = makeGoal(ctx, 0);
    ctx.data().arena.goals[1] = makeGoal(ctx, 1);

    constexpr CountT total_num_cars =
        consts::numCarsPerTeam * consts::numTeams;

    for (CountT i = 0; i < total_num_cars; i++) {
        SimObject team_obj = (i / consts::numCarsPerTeam == 0) ?
            SimObject::AgentTeam0 : SimObject::AgentTeam1;

        Entity car = ctx.data().cars[i] = ctx.makeRenderableEntity<Car>();

        if (ctx.data().enableRender) {
            render::RenderingSystem::attachEntityToView(ctx,
                    car,
                    100.f, 0.001f,
                    1.5f * math::up);
        }

        ctx.get<Scale>(car) = Diag3x3::fromVec(consts::agentDimensions);
        ctx.get<ObjectID>(car) = ObjectID { (int32_t)team_obj };
        ctx.get<EntityType>(car) = EntityType::Agent;
        ctx.get<DynamicEntityType>(car) = DynamicEntityType::Car;

        // This needs to be initialized for the viewer, which won't pass
        // in a legitimate policyIdx.
        ctx.get<CarPolicy>(car).policyIdx = 0;
    }

    Entity ball = ctx.data().ball =
        ctx.makeRenderableEntity<Ball>();

    float ball_rad = consts::ballRadius;
    ctx.get<Scale>(ball) = Diag3x3 { ball_rad, ball_rad, ball_rad };
    ctx.get<ObjectID>(ball) = ObjectID { (int32_t)SimObject::Sphere };
    ctx.get<EntityType>(ball) = EntityType::Ball;
    ctx.get<DynamicEntityType>(ball) = DynamicEntityType::Ball;
    ctx.get<BallGoalState>(ball).state = BallGoalState::State::NotInGoal;
}

void placeEntities(Engine &ctx)
{
    for (CountT team_idx = 0; team_idx < 2; ++team_idx) {
        Team &team = ctx.data().teams[team_idx];

        for (CountT car_idx = 0; car_idx < consts::numCarsPerTeam; ++car_idx) {
            Entity car_entity = team.players[car_idx];

            // Place the agents near the starting wall
            Vector3 pos { 0.f, 0.f, consts::agentDimensions.z };
            Quat rot{};

            if (team_idx % 2 == 0) {
                pos.x = 0.0f;
                pos.y = consts::worldLength / 2.5f;

                rot = Quat::angleAxis(
                    math::pi,
                    math::up);
            } else {
                pos.x = 0.0f;
                pos.y = -consts::worldLength / 2.5f;

                rot = Quat::angleAxis(
                    0.0f,
                    math::up);
            }

            // Set the position's x component
            pos.x = ((float)car_idx+1.f) * (consts::worldWidth / ((float)consts::numCarsPerTeam+1.f)) -
                consts::worldWidth/2.f;

            ctx.get<Position>(car_entity) = pos;
            ctx.get<Rotation>(car_entity) = rot;

            ctx.get<Velocity>(car_entity) = {
                Vector3::zero(),
                Vector3::zero(),
            };
            ctx.get<Action>(car_entity) = Action {
                .moveAmount = 1,
                .rotate = 1,
            };
            ctx.get<CarBallTouchState>(car_entity).touched = 0;
        }
    }

    Entity ball_entity = ctx.data().ball;

    ctx.get<Position>(ball_entity) = Vector3{ 0.f, 0.f, consts::ballRadius };
    ctx.get<Rotation>(ball_entity) = 
        Quat::angleAxis(0.0f, Vector3{0.f, 0.f, 1.f});
    ctx.get<Velocity>(ball_entity) = {
        Vector3::zero(),
        Vector3::zero()
    };
    ctx.get<BallGoalState>(ball_entity) = BallGoalState{
        BallGoalState::State::None,
        0
    };
}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

    Arena &arena = ctx.data().arena;

    for (CountT i = 0; i < 2; i++) {
        registerRigidBodyEntity(ctx, arena.longBorders[i], SimObject::Wall);

        registerRigidBodyEntity(ctx, arena.goals[i].outerBorders[0], SimObject::Wall);
        registerRigidBodyEntity(ctx, arena.goals[i].outerBorders[1], SimObject::Wall);
    }

    CountT team_offsets[consts::numTeams];
    team_offsets[0] = 0;
    team_offsets[1] = consts::numCarsPerTeam;

    if ((ctx.singleton<SimFlags>() & SimFlags::RandomFlipTeams) ==
            SimFlags::RandomFlipTeams) {
        if (ctx.data().rng.sampleUniform() < 0.5) {
            team_offsets[0] = consts::numCarsPerTeam;
            team_offsets[1] = 0;
        }
    }

    for (CountT team_idx = 0; team_idx < 2; ++team_idx) {
        Team &team = ctx.data().teams[team_idx];

        for (CountT car_idx = 0; car_idx < consts::numCarsPerTeam; ++car_idx) {
            Entity car_entity =
                ctx.data().cars[team_offsets[team_idx] + car_idx];

            team.players[car_idx] = car_entity;
            ctx.get<TeamState>(car_entity).teamIdx = (int32_t)team_idx;
        }
    }

    placeEntities(ctx);
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
}

}
