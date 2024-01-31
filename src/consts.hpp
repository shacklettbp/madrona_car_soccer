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
inline constexpr float buttonWidth = 1.3f;
inline constexpr float agentRadius = 1.f;
inline constexpr float roomLength = worldLength / numRooms;

// Each unit of distance forward (+ y axis) rewards the agents by this amount
inline constexpr float rewardPerDist = 0.05f;
// Each step that the agents don't make additional progress they get a small
// penalty reward
inline constexpr float slackReward = -0.005f;

// Steps per episode
inline constexpr int32_t episodeLen = 2000;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 4;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 5;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;

// Time (seconds) per step
inline constexpr float deltaT = 0.05f;

// Speed at which doors raise and lower
inline constexpr float doorSpeed = 30.f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

inline constexpr float carAcceleration = 80.f;

inline constexpr float ballRadius = 0.7f;

// Half width, length, height
inline constexpr madrona::math::Vector3 agentDimensions
    { 0.8f, 1.7f, 0.7f };

}

}
