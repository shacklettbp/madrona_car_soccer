#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

using namespace madrona::py;

namespace madEscape {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_car_soccer, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    nb::enum_<SimFlags>(m, "SimFlags", nb::is_arithmetic())
        .value("Default", SimFlags::Default)
        .value("StaggerStarts", SimFlags::StaggerStarts)
        .value("RandomFlipTeams", SimFlags::RandomFlipTeams)
    ;

    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            bool auto_reset,
                            uint32_t sim_flags,
                            uint32_t num_pbt_policies,
                            bool enable_batch_renderer) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .autoReset = auto_reset,
                .simFlags = SimFlags(sim_flags),
                .numPBTPolicies = num_pbt_policies,
                .enableBatchRenderer = enable_batch_renderer,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("auto_reset"),
           nb::arg("sim_flags"),
           nb::arg("num_pbt_policies"),
           nb::arg("enable_batch_renderer") = false)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("team_observation_tensor", &Manager::teamObservationTensor)
        .def("enemy_observation_tensor",
             &Manager::enemyObservationTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("load_ckpt_tensor", &Manager::loadCheckpointTensor)
        .def("ckpt_tensor", &Manager::checkpointTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("jax", JAXInterface::buildEntry<
                &Manager::trainInterface,
                &Manager::init,
                &Manager::step
#ifdef MADRONA_CUDA_SUPPORT
                ,
                &Manager::gpuStreamInit,
                &Manager::gpuStreamStep
#endif
             >())
    ;
}

}
