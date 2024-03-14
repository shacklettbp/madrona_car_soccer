#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>

#include <imgui.h>

using namespace madrona;
using namespace madrona::viz;
using namespace madEscape;

static void badRecording()
{
    FATAL("Invalid recording");
}

static HeapArray<Checkpoint> readReplayLog(const char *path)
{
    std::ifstream replay_log_file(path, std::ios::binary);
    if (!replay_log_file.is_open()) {
        badRecording();
    }

    replay_log_file.seekg(0, std::ios::end);
    size_t num_bytes = replay_log_file.tellg();
    replay_log_file.seekg(0, std::ios::beg);

    size_t num_steps = num_bytes / sizeof(Checkpoint);
    if (num_steps * sizeof(Checkpoint) != num_bytes) {
        badRecording();
    }

    HeapArray<Checkpoint> log_data(num_steps);

    replay_log_file.read((char *)log_data.data(), num_bytes);

    return log_data;
}

int main(int argc, char *argv[])
{
    constexpr int64_t num_views = consts::numTeams * consts::numCarsPerTeam;

    uint32_t num_worlds = 1;
    ExecMode exec_mode = ExecMode::CPU;

    auto usageErr = [argv]() {
        fprintf(stderr, "%s [NUM_WORLDS] [--backend cpu|cuda] [--record path] [--replay path] [--load-ckpt path] [--print-obs]\n", argv[0]);
        exit(EXIT_FAILURE);
    };

    bool num_worlds_set = false;

    char *record_log_path = nullptr;
    char *replay_log_path = nullptr;
    char *load_ckpt_path = nullptr;
    bool start_frozen = false;
    bool print_obs = false;

    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];

        if (arg[0] == '-' && arg[1] == '-') {
            arg += 2;

            if (!strcmp("backend", arg)) {
                i += 1;

                if (i == argc) {
                    usageErr();
                }

                char *value = argv[i];
                if (!strcmp("cpu", value)) {
                    exec_mode = ExecMode::CPU;
                } else if (!strcmp("cuda", value)) {
                    exec_mode = ExecMode::CUDA;
                } else {
                    usageErr();
                }
            } else if (!strcmp("record", arg)) {
                if (record_log_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                record_log_path = argv[i];
            } else if (!strcmp("replay", arg)) {
                if (replay_log_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                replay_log_path = argv[i];
            } else if (!strcmp("load-ckpt", arg)) {
                if (load_ckpt_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                load_ckpt_path = argv[i];
            } else if (!strcmp("freeze", arg)) {
                start_frozen = true;
            } else if (!strcmp("print-obs", arg)) {
                print_obs = true;
            } else {
                usageErr();
            }
        } else {
            if (num_worlds_set) {
                usageErr();
            }

            num_worlds_set = true;

            num_worlds = (uint32_t)atoi(arg);
        }
    }

    (void)record_log_path;

    auto replay_log = Optional<HeapArray<Checkpoint>>::none();
    CountT cur_replay_step = 0;
    CountT num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / num_worlds;
        if (num_replay_steps * num_worlds != replay_log->size()) {
            badRecording();
        }
    }

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Escape Room", 2730, 1536);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .autoReset = replay_log.has_value(),
        .simFlags = SimFlags::Default,
        .numPBTPolicies = 0,
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });
    mgr.init();

    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, 0, 30 };

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = start_frozen ? 0_u32 : 20_u32,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });

    auto self_tensor = mgr.selfObservationTensor();
    auto my_goal_tensor = mgr.myGoalObservationTensor();
    auto enemy_goal_tensor = mgr.enemyGoalObservationTensor();
    auto team_tensor = mgr.teamObservationTensor();
    auto enemy_tensor = mgr.enemyObservationTensor();
    auto ball_tensor = mgr.ballTensor();
    auto steps_remaining_tensor = mgr.stepsRemainingTensor();
    auto reward_tensor = mgr.rewardTensor();
    auto episode_result_tensor = mgr.episodeResultTensor();

    // Printers
    auto self_printer = self_tensor.makePrinter();
    auto team_printer = team_tensor.makePrinter();
    auto enemy_printer = enemy_tensor.makePrinter();
    auto ball_printer = ball_tensor.makePrinter();
    auto steps_remaining_printer = steps_remaining_tensor.makePrinter();
    auto reward_printer = reward_tensor.makePrinter();
    auto episode_result_printer = episode_result_tensor.makePrinter();

    auto ckpt_tensor = mgr.checkpointTensor();
    auto load_ckpt_tensor = mgr.loadCheckpointTensor();

    HeapArray<LoadCheckpoint> load_all_checkpoints(num_worlds);
    for (CountT i = 0; i < (CountT)num_worlds; i++) {
        load_all_checkpoints[i].load = 1;
    }

#ifdef MADRONA_CUDA_SUPPORT
    cudaStream_t copy_strm;
    REQ_CUDA(cudaStreamCreate(&copy_strm));
#endif

    // Replay step
    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps) {
            return true;
        }

        printf("Step: %ld\n", (long)cur_replay_step);

        const Checkpoint *cur_step_ckpts = replay_log->data() +
            cur_replay_step * (CountT)num_worlds;

        if (exec_mode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
            cudaMemcpyAsync(ckpt_tensor.devicePtr(), cur_step_ckpts,
                sizeof(Checkpoint) * num_worlds,
                cudaMemcpyHostToDevice, copy_strm);

            cudaMemcpyAsync(load_ckpt_tensor.devicePtr(),
                            load_all_checkpoints.data(),
                            sizeof(LoadCheckpoint) * num_worlds,
                            cudaMemcpyHostToDevice, copy_strm);

            REQ_CUDA(cudaStreamSynchronize(copy_strm));
#endif
        } else {
            memcpy(ckpt_tensor.devicePtr(), cur_step_ckpts,
                   sizeof(Checkpoint) * num_worlds);

            memcpy(load_ckpt_tensor.devicePtr(), load_all_checkpoints.data(),
                   sizeof(LoadCheckpoint) * num_worlds);
        }

        cur_replay_step++;

        return false;
    };

    auto printObs = [&]() {
        if (!print_obs) {
            return;
        }

        printf("Self\n");
        self_printer.print();

        printf("Team\n");
        team_printer.print();

        printf("Enemy\n");
        enemy_printer.print();

        printf("Ball\n");
        ball_printer.print();

        printf("Steps Remaining\n");
        steps_remaining_printer.print();

        printf("Reward\n");
        reward_printer.print();

        printf("Episode Result\n");
        episode_result_printer.print();

        printf("\n");
    };


#ifdef MADRONA_CUDA_SUPPORT
    SelfObservation *self_obs_readback = (SelfObservation *)cu::allocReadback(
        sizeof(SelfObservation) * num_views);

    MyGoalObservation *my_goal_obs_readback = 
        (MyGoalObservation *)cu::allocReadback(
            sizeof(MyGoalObservation) * num_views);

    EnemyGoalObservation *enemy_goal_obs_readback = 
        (EnemyGoalObservation *)cu::allocReadback(
            sizeof(EnemyGoalObservation) * num_views);

    Reward *reward_readback = (Reward *)cu::allocReadback(
        sizeof(Reward) * num_views);

    BallObservation *ball_readback = (BallObservation *)cu::allocReadback(
        sizeof(BallObservation));
#endif

    // Main loop for the viewer viewer
    viewer.loop(
    [&mgr](CountT world_idx, const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;
        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx);
        }
    },
    [&mgr](CountT world_idx, CountT agent_idx,
           const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        const int32_t move_mid = consts::numMoveAmountBuckets / 2;
        const int32_t turn_mid = consts::numTurnBuckets / 2;

        int32_t y = 0;
        int32_t r = 0;

        int32_t move_delta = 1;
        int32_t turn_delta = 1;

        if (input.keyPressed(Key::Shift)) {
            move_delta = move_mid;
            turn_delta = turn_mid;
        }

        if (input.keyPressed(Key::W)) {
            y += move_delta;
        }
        if (input.keyPressed(Key::S)) {
            y -= move_delta;
        }

        if (input.keyPressed(Key::Q)) {
            r += turn_delta;
        }
        if (input.keyPressed(Key::E)) {
            r -= turn_delta;
        }

        mgr.setAction(world_idx, agent_idx, move_mid + y, turn_mid + r);
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        mgr.step();

        printObs();
    }, [&]() {
        CountT cur_world_id = viewer.getCurrentWorldID();
        CountT agent_world_offset = cur_world_id * num_views;

        SelfObservation *self_obs_ptr =
            (SelfObservation *)self_tensor.devicePtr();

        MyGoalObservation *my_goal_obs_ptr =
            (MyGoalObservation *)my_goal_tensor.devicePtr();

        EnemyGoalObservation *enemy_goal_obs_ptr =
            (EnemyGoalObservation *)enemy_goal_tensor.devicePtr();

        Reward *reward_ptr = (Reward *)reward_tensor.devicePtr();

        BallObservation *ball_obs_ptr =
            (BallObservation *)ball_tensor.devicePtr();

        self_obs_ptr += agent_world_offset;
        my_goal_obs_ptr += agent_world_offset;
        enemy_goal_obs_ptr += agent_world_offset;
        reward_ptr += agent_world_offset;

        ball_obs_ptr += cur_world_id;

        if (exec_mode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
            cudaMemcpyAsync(self_obs_readback, self_obs_ptr,
                            sizeof(SelfObservation) * num_views,
                            cudaMemcpyDeviceToHost, copy_strm);

            cudaMemcpyAsync(my_goal_obs_readback, my_goal_obs_ptr,
                            sizeof(MyGoalObservation) * num_views,
                            cudaMemcpyDeviceToHost, copy_strm);

            cudaMemcpyAsync(enemy_goal_obs_readback, enemy_goal_obs_ptr,
                            sizeof(EnemyGoalObservation) * num_views,
                            cudaMemcpyDeviceToHost, copy_strm);

            cudaMemcpyAsync(reward_readback, reward_ptr,
                            sizeof(Reward) * num_views,
                            cudaMemcpyDeviceToHost, copy_strm);

            cudaMemcpyAsync(ball_readback, ball_obs_ptr,
                            sizeof(BallObservation),
                            cudaMemcpyDeviceToHost, copy_strm);

            REQ_CUDA(cudaStreamSynchronize(copy_strm));

            self_obs_ptr = self_obs_readback;
            my_goal_obs_ptr = my_goal_obs_readback;
            enemy_goal_obs_ptr = enemy_goal_obs_readback;
            reward_ptr = reward_readback;
            ball_obs_ptr = ball_readback;
#endif
        }

        for (int64_t i = 0; i < num_views; i++) {
            auto player_str = std::string("Player ") + std::to_string(i);
            ImGui::Begin(player_str.c_str());

            const SelfObservation &cur_self = self_obs_ptr[i];
            const MyGoalObservation &cur_my_goal = my_goal_obs_ptr[i];
            const EnemyGoalObservation &cur_enemy_goal = enemy_goal_obs_ptr[i];
            const Reward &reward = reward_ptr[i];
            const BallObservation &ball = ball_obs_ptr[i];

            ImGui::Text("Position:      (%.1f, %.1f, %.1f)",
                cur_self.x, cur_self.y, cur_self.z);
            ImGui::Text("Rotation:      %.2f",
                cur_self.theta);
            ImGui::Text("Velocity:      (%.1f, %.1f, %.1f)",
                cur_self.vel.r, cur_self.vel.theta, cur_self.vel.phi);
            ImGui::Text("To Ball:       (%.1f, %.1f, %.1f)",
                ball.pos.r, ball.pos.theta, ball.pos.phi);

            ImGui::Text("To My Goal:    (%.1f, %.1f, %.1f)",
                cur_my_goal.pos.r, cur_my_goal.pos.theta, cur_my_goal.pos.phi);

            ImGui::Text("To Enemy Goal: (%.1f, %.1f, %.1f)",
                cur_enemy_goal.pos.r, cur_enemy_goal.pos.theta, cur_enemy_goal.pos.phi);

            ImGui::Text("Reward:    %.3f",
                reward.v);

            ImGui::End();
        }
    });
}
