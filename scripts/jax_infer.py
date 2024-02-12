import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
import numpy as np

import argparse
from functools import partial

import madrona_rocket_league
from madrona_rocket_league import SimFlags

import madrona_learn

from jax_policy import make_policy

madrona_learn.init(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, default=200)

arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--crossplay', action='store_true')
arg_parser.add_argument('--single-policy', type=int, default=None)
arg_parser.add_argument('--action-dump-path', type=str)

arg_parser.add_argument('--print-obs', action='store_true')
arg_parser.add_argument('--print-action-probs', action='store_true')
arg_parser.add_argument('--print-rewards', action='store_true')

arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

sim = madrona_rocket_league.SimManager(
    exec_mode = madrona_rocket_league.madrona.ExecMode.CUDA if args.gpu_sim else madrona_rocket_league.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    num_pbt_policies = 0,
    auto_reset = True,
    rand_seed = 5,
    sim_flags = SimFlags.Default,
)

team_size = 3
num_teams = 2

num_agents_per_world = team_size * num_teams

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_init, sim_step = sim.jax(jax_gpu)

if args.action_dump_path:
    action_log = open(args.action_dump_path, 'wb')
else:
    action_log = None

step_idx = 0

def host_cb(obs, actions, action_probs, values, dones, rewards):
    global step_idx

    if args.print_obs:
        print(obs)

    print(f"\nStep {step_idx}")

    if args.print_action_probs:
        for i in range(actions.shape[0]):
            if i % num_agents_per_world == 0:
                print(f"World {i // num_agents_per_world}")

            print(f" Agent {i % num_agents_per_world}:")
            print("  Action:", actions[..., i, :])

            print(f"  Move Amount Probs: {float(action_probs[0][i][0]):.2e} {float(action_probs[0][i][1]):.2e} {float(action_probs[0][i][2]):.2e}")
            print(f"  Turn Probs:        {float(action_probs[1][i][0]):.2e} {float(action_probs[1][i][1]):.2e} {float(action_probs[1][i][2]):.2e}")

    if args.print_rewards:
        print("Rewards:", rewards)

    if action_log:
        actions.tofile(action_log)

    step_idx += 1

    return ()

def iter_cb(step_data):
    cb = partial(jax.experimental.io_callback, host_cb, ())

    cb(step_data['obs'],
       step_data['actions'],
       step_data['action_probs'],
       step_data['values'],
       step_data['dones'],
       step_data['rewards'])

dev = jax.devices()[0]

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32

policy = make_policy(dtype)

single_policy_eval = None
multi_policy_eval = None

if args.single_policy != None:
    assert not args.crossplay
    single_policy_eval = args.single_policy
elif args.crossplay:
    multi_policy_eval = madrona_learn.MultiPolicyEvalConfig(
        num_teams = num_teams,
        team_size = team_size,
    )

cfg = madrona_learn.EvalConfig(
    ckpt_path = args.ckpt_path,
    num_worlds = args.num_worlds,
    num_agents_per_world = team_size * num_teams,
    num_eval_steps = args.num_steps,
    policy_dtype = dtype,
    single_policy_eval = single_policy_eval,
    multi_policy_eval = multi_policy_eval,
)

madrona_learn.eval_ckpt(
    dev, cfg, sim_init, sim_step, policy, iter_cb)

del sim
