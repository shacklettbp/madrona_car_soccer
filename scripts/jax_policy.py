import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import FrozenDict

import argparse
from functools import partial
import math

import madrona_learn
from madrona_learn import (
    Policy, ActorCritic, BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
    ObservationsEMANormalizer, ObservationsCaster,
)

from madrona_learn.models import (
    LayerNorm,
    MLP,
    EntitySelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
)
from madrona_learn.rnn import LSTM

def assert_valid_input(tensor):
    checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

def assert_valid_input(tensor):
    return None
    #checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    #checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

class PolicyRNN(nn.Module):
    rnn: nn.Module
    norm: nn.Module

    @staticmethod
    def create(num_hidden_channels, num_layers, dtype, rnn_cls = LSTM):
        return PolicyRNN(
            rnn = rnn_cls(
                num_hidden_channels = num_hidden_channels,
                num_layers = num_layers,
                dtype = dtype,
            ),
            norm = LayerNorm(dtype=dtype),
        )

    @nn.nowrap
    def init_recurrent_state(self, N):
        return self.rnn.init_recurrent_state(N)

    @nn.nowrap
    def clear_recurrent_state(self, rnn_states, should_clear):
        return self.rnn.clear_recurrent_state(rnn_states, should_clear)

    def setup(self):
        pass

    def __call__(
        self,
        cur_hiddens,
        x,
        train,
    ):
        out, new_hiddens = self.rnn(cur_hiddens, x, train)
        return self.norm(out), new_hiddens

    def sequence(
        self,
        start_hiddens,
        seq_ends,
        seq_x,
        train,
    ):
        return self.norm(
            self.rnn.sequence(start_hiddens, seq_ends, seq_x, train))

class PrefixCommon(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        jax.tree_map(lambda x: assert_valid_input(x), obs)

        obs, self_ob = obs.pop('self')
        obs, steps_remaining_ob = obs.pop('stepsRemaining')

        self_ob = jnp.concatenate([
            self_ob,
            steps_remaining_ob,
        ], axis=-1)
        
        obs, my_goal_ob = obs.pop('my_goal')
        obs, enemy_goal_ob = obs.pop('enemy_goal')
        obs, team_ob = obs.pop('team')
        obs, enemy_ob = obs.pop('enemy')
        obs, ball_ob = obs.pop('ball')

        assert len(obs) == 0

        return FrozenDict({
            'self': self_ob, 
            'my_goal': my_goal_ob, 
            'enemy_goal': enemy_goal_ob, 
            'ball_ob': ball_ob,
            'team': team_ob,
            'enemy': enemy_ob,
        })


class SimpleNet(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        num_batch_dims = len(obs['self'].shape) - 1
        obs = jax.tree_map(
            lambda o: o.reshape(*o.shape[0:num_batch_dims], -1), obs)

        flattened, _ = jax.tree_util.tree_flatten(obs)
        flattened = jnp.concatenate(flattened, axis=-1)

        return MLP(
                num_channels = 256,
                num_layers = 3,
                dtype = self.dtype,
            )(flattened, train)

class ActorNet(nn.Module):
    dtype: jnp.dtype
    use_simple: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        if self.use_simple:
            return SimpleNet(dtype=self.dtype)(obs, train)
        else:
            return EntitySelfAttentionNet(
                    num_embed_channels = 128,
                    num_out_channels = 256,
                    num_heads = 4,
                    dtype = self.dtype,
                )(obs, train=train)


class CriticNet(nn.Module):
    dtype: jnp.dtype
    use_simple: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        if self.use_simple:
            return SimpleNet(dtype=self.dtype)(obs, train)
        else:
            return EntitySelfAttentionNet(
                    num_embed_channels = 128,
                    num_out_channels = 256,
                    num_heads = 4,
                    dtype = self.dtype,
                )(obs, train=train)

def make_policy(dtype):
    actor_encoder = RecurrentBackboneEncoder(
        net = ActorNet(dtype, use_simple=False),
        rnn = PolicyRNN.create(
            num_hidden_channels = 256,
            num_layers = 1,
            dtype = dtype,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = CriticNet(dtype, use_simple=False),
        rnn = PolicyRNN.create(
            num_hidden_channels = 256,
            num_layers = 1,
            dtype = dtype,
        ),
    )

    backbone = BackboneSeparate(
        prefix = PrefixCommon(
            dtype = dtype,
        ),
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    actor_critic = ActorCritic(
        backbone = backbone,
        actor = DenseLayerDiscreteActor(
            actions_num_buckets = [3, 3],
            dtype = dtype,
        ),
        critic = DenseLayerCritic(dtype=dtype),
    )

    obs_preprocess = ObservationsEMANormalizer.create(
        decay = 0.99999,
        dtype = dtype,
        prep_fns = {},
        skip_normalization = {},
    )

    def get_episode_scores(episode_result):
        winner_id = episode_result[..., 0]
        is_a_winner = winner_id == 0
        is_b_winner = winner_id == 1

        a_score = jnp.where(is_a_winner, 1, jnp.where(is_b_winner, 0, 0.5))
        b_score = 1 - a_score

        return jnp.stack((a_score, b_score))

    return Policy(
        actor_critic = actor_critic,
        obs_preprocess = obs_preprocess,
        get_episode_scores = get_episode_scores,
    )
