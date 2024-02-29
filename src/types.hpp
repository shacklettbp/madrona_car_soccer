#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"
#include "physics.hpp"
#include "sim_flags.hpp"

namespace madEscape {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;
using madrona::math::Vector3;
using madrona::math::Quat;

// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

// Discrete action component. Ranges are defined by consts::numMoveBuckets (5),
// repeated here for clarity
struct Action {
    int32_t moveAmount; // [-1, 1]
    int32_t rotate; // [-1, 1]
};

// Per-agent reward
// Exported as an [N * A, 1] float tensor to training code
struct Reward {
    float v;
};

// Per-agent component that indicates that the agent's episode is finished
// This is exported per-agent for simplicity in the training code
struct Done {
    // Currently bool components are not supported due to
    // padding issues, so Done is an int32_t
    int32_t v;
};

static_assert(sizeof(WorldReset) == sizeof(int32_t));
static_assert(sizeof(Reward) == sizeof(float));
static_assert(sizeof(Done) == sizeof(int32_t));

struct EpisodeResult {
    int32_t winResult;
    int32_t numTeamAGoals;
    int32_t numTeamBGoals;
};

struct MatchInfo {
    int32_t stepsRemaining;
};

struct TeamRewardState {
    float teamRewards[consts::numTeams];
};

struct PolarObservation {
    float r, theta, phi;
};

struct RewardHyperParams {
    float teamSpirit = 1.f;
    float hitRewardScale = 0.1f;
};

struct CarPolicy {
    int32_t policyIdx;
};

// Observation state for the current agent.
// Positions are rescaled to the bounds of the play area to assist training.
struct SelfObservation {
    // For now, only x and y are used, but when we introduce the ramps and walls,
    // we're going to use z.
    float x, y, z;
    
    // The direction in which the car is facing.
    float theta;

    PolarObservation vel;

};

struct MyGoalObservation {
    PolarObservation pos;
};

struct EnemyGoalObservation {
    PolarObservation pos;
};

static_assert(sizeof(SelfObservation) == sizeof(float) * 7);

// Global position of the ball
struct BallObservation {
    PolarObservation pos;
    PolarObservation vel;
};

struct OtherObservation {
    // Used to get the relative position/direction from current agent.
    PolarObservation polar;
    // The other's facing direction.
    float o_theta;

    // Velocity direction and magnitude
    PolarObservation vel;
};

struct TeamObservation {
    OtherObservation obs[consts::numCarsPerTeam-1];
};

struct EnemyObservation {
    OtherObservation obs[consts::numCarsPerTeam];
};

struct LidarSample {
    float depth;
    float encodedType;
};

// Linear depth values and entity type in a circle around the agent
struct Lidar {
    LidarSample samples[consts::numLidarSamples];
};

// Number of steps remaining in the episode. Allows non-recurrent policies
// to track the progression of time.
struct StepsRemainingObservation {
    uint32_t t;
};

// Tracks progress the agent has made through the challenge, used to add
// reward when more progress has been made
struct Progress {
    float maxY;
};

// Per-agent component storing Entity IDs of the other agents. Used to
// build the egocentric observations of their state.
struct OtherAgents {
    madrona::Entity e[consts::numAgents - 1];
};

// Tracks if an agent is currently grabbing another entity
struct GrabState {
    Entity constraintEntity;
};

// Not an actual momentum
struct Momentum {
    float rho;
};

// This enum is used to track the type of each entity for the purposes of
// classifying the objects hit by each lidar sample.
enum class EntityType : uint32_t {
    None,
    Button,
    Cube,
    Wall,
    Agent,
    Door,
    Ball,
    NumTypes,
};

enum class DynamicEntityType : uint32_t {
    None,
    Car,
    Ball,
    NumTypes,
};

// Encapsulates goal cage
struct Goal {
    // Left, back, right walls
    Entity outerBorders[2];
    
    Vector3 centerPosition;
};

struct Arena {
    Goal goals[2];

    // Across the length of the arena
    Entity longBorders[2];

    WallPlane wallPlanes[4];
};

// For car-car collision, a is the car to which we subtract `overlap`,
// and b is the car to which we add `overlap`.
struct CollisionData {
    Entity a;
    Entity b;

    // These vectors will be used differently depending on what type 
    // of entity a and b are.
    madrona::math::Vector3 overlap;
    madrona::math::Vector3 diff;
};

struct TeamState {
    int32_t teamIdx;
};

struct CarBallTouchState {
    bool touched;
};

/* ECS Archetypes for the game */

struct BallGoalState {
    enum class State {
        None,
        InGoal,
        NotInGoal,
        NumStates
    };

    State state;

    // If InGoal, specify which goal it's in.
    int data;
};

enum class GoalState {
    None,
    InGoal,
    NotInGoal,
    NumStates,
};

struct LoadCheckpoint {
    int32_t load;
};

struct Checkpoint {
    struct CarData {
        Vector3 position;
        Quat rotation;
        Velocity velocity;
    };

    struct TeamData {
        CarData cars[consts::numCarsPerTeam];
        uint32_t goalIdx;
    };

    struct BallData {
        Vector3 position;
        Quat rotation;
        Velocity velocity;
    };

    TeamData teams[2];
    BallData ball;
    int32_t stepsRemaining;
    int32_t numTeamAGoals;
    int32_t numTeamBGoals;
};

struct Ball : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    EntityType,
    BallGoalState,
    DynamicEntityType,
    madrona::render::Renderable
> {};

struct Car : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    Action,
    EntityType,
    Done,
    DynamicEntityType,
    TeamState,
    CarBallTouchState,

    CarPolicy,

    // Observations
    SelfObservation,
    MyGoalObservation,
    EnemyGoalObservation,
    TeamObservation,
    EnemyObservation,
    BallObservation,
    StepsRemainingObservation,
    Reward,

    madrona::render::RenderCamera,
    madrona::render::Renderable
> {};

// Generic archetype for entities that need physics but don't have custom
// logic associated with them.
struct PhysicsEntity : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    EntityType,
    madrona::render::Renderable
> {};

struct Collision : public madrona::Archetype<
    CollisionData
> {};

struct Team {
    Entity players[consts::numCarsPerTeam];
    int32_t goalIdx;
};

}
