#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>

using namespace madrona;
using namespace madrona::viz;

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    constexpr int64_t num_views = 6;

    // Read command line arguments
    uint32_t num_worlds = 1;
    if (argc >= 2) {
        num_worlds = (uint32_t)atoi(argv[1]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 3) {
        if (!strcmp("--cpu", argv[2])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[2])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    // Setup replay log
    const char *replay_log_path = nullptr;
    if (argc >= 4) {
        replay_log_path = argv[3];
    }

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 2);
    }

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Escape Room", 2730/2, 1536/2);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .autoReset = replay_log.has_value(),
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });

    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, 0, 30 };

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 20,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });

    // Replay step
    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            for (uint32_t j = 0; j < num_views; j++) {
                uint32_t base_idx = 0;
                base_idx = 2 * (cur_replay_step * num_views * num_worlds +
                    i * num_views + j);

                int32_t move_amount = (*replay_log)[base_idx];
                int32_t turn = (*replay_log)[base_idx + 1];

                printf("%d, %d: %d %d\n",
                       i, j, move_amount, turn);
                mgr.setAction(i, j, move_amount, turn);
            }
        }

        cur_replay_step++;

        return false;
    };

    // Printers
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto team_printer = mgr.teamObservationTensor().makePrinter();
    auto enemy_printer = mgr.enemyObservationTensor().makePrinter();
    auto ball_printer = mgr.ballTensor().makePrinter();
    auto steps_remaining_printer = mgr.stepsRemainingTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();

    auto printObs = [&]() {
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

        printf("\n");
    };



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

        int32_t y = 0;
        int32_t r = 0;

        if (input.keyPressed(Key::W)) {
            y += 1;
        }
        if (input.keyPressed(Key::S)) {
            y -= 1;
        }

        if (input.keyPressed(Key::Q)) {
            r += 1;
        }
        if (input.keyPressed(Key::E)) {
            r -= 1;
        }

        int32_t move_amount = y+1;

        mgr.setAction(world_idx, agent_idx, move_amount, r+1);
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        mgr.step();

        //printObs();
    }, []() {});
}
