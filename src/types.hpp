#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"
#include "physics.hpp"

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

struct PolarObservation {
    float r, theta, phi;
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
struct StepsRemaining {
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

// A per-door component that tracks whether or not the door should be open.
struct OpenState {
    bool isOpen;
};

// Linked buttons that control the door opening and whether or not the door
// should remain open after the buttons are pressed once.
struct DoorProperties {
    Entity buttons[consts::maxEntitiesPerRoom];
    int32_t numButtons;
    bool isPersistent;
};

// Similar to OpenState, true during frames where a button is pressed
struct ButtonState {
    bool isPressed;
};

// Encapsulates goal cage
struct Goal {
    // Left, back, right walls
    Entity outerBorders[2];
    
    madrona::math::Vector3 minBound;
    madrona::math::Vector3 maxBound;
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
    int32_t touched;
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

struct Ball : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    EntityType,
    BallGoalState,
    DynamicEntityType,
    BallObservation,
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

    // Observations
    SelfObservation,
    TeamObservation,
    EnemyObservation,
    BallObservation,
    StepsRemaining,
    Reward,

    madrona::render::RenderCamera,
    madrona::render::Renderable
> {};

// There are 2 Agents in the environment trying to get to the destination
struct Agent : public madrona::Archetype<
    // Basic components required for physics. Note that the current physics
    // implementation requires archetypes to have these components first
    // in this exact order.
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

    // Internal logic state.
    GrabState,
    Progress,
    OtherAgents,
    EntityType,

    // Input
    Action,

    // Observations
    SelfObservation,
    // PartnerObservations,
    // RoomEntityObservations,
    // DoorObservation,
    Lidar,
    StepsRemaining,

    // Reward, episode termination
    Reward,
    Done,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    madrona::render::RenderCamera,
    // All entities with the Renderable component will be drawn by the
    // viewer and batch renderer
    madrona::render::Renderable
> {};

// Archetype for the doors blocking the end of each challenge room
struct DoorEntity : public madrona::Archetype<
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
    OpenState,
    DoorProperties,
    EntityType,
    madrona::render::Renderable
> {};

// Archetype for the button objects that open the doors
// Buttons don't have collision but are rendered
struct ButtonEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    ButtonState,
    EntityType,
    madrona::render::Renderable,
    Done
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
};

}
