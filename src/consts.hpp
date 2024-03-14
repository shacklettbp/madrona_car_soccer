#pragma once

#include "madrona/math.hpp"
#include <madrona/types.hpp>

namespace madEscape {
namespace consts {

inline constexpr madrona::CountT numTeams = 2;
inline constexpr madrona::CountT numCarsPerTeam = 3;

// Each random world is composed of a fixed number of rooms that the agents
// must solve in order to maximize their reward.
inline constexpr madrona::CountT numRooms = 3;

// Generated levels assume 2 agents
inline constexpr madrona::CountT numAgents = numTeams * numCarsPerTeam;

// Maximum number of interactive objects per challenge room. This is needed
// in order to setup the fixed-size learning tensors appropriately.
inline constexpr madrona::CountT maxEntitiesPerRoom = 6;

// Various world / entity size parameters
inline constexpr float worldLength = 60.f;
inline constexpr float worldWidth = 40.f;
inline constexpr float wallWidth = 1.f;
inline constexpr float wallHeight = 1.f;

// Steps per episode
inline constexpr int32_t episodeLen = 1200;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 5;
inline constexpr madrona::CountT numTurnBuckets = 5;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;

// Time (seconds) per step
inline constexpr float deltaT = 0.05f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4;

inline constexpr float carAcceleration = 80.f;

inline constexpr float ballRadius = 0.7f;

inline constexpr float agentHeight = 0.7f;

}
}
