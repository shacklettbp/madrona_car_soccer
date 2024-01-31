import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial
import math

import madrona_learn
from madrona_learn import (
    ActorCritic, BackboneShared, BackboneSeparate,
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

class PolicyLSTM(nn.Module):
    num_hidden_channels: int
    num_layers: int
    dtype: jnp.dtype

    def setup(self):
        self.lstm = LSTM(
            num_hidden_channels = self.num_hidden_channels,
            num_layers = self.num_layers,
            dtype = self.dtype,
        )

        self.layernorm = LayerNorm(dtype=self.dtype)

    def __call__(
        self,
        cur_hiddens,
        x,
        train,
    ):
        return self.layernorm(self.lstm(cur_hiddens, x, train))

    def sequence(
        self,
        start_hiddens,
        seq_ends,
        seq_x,
        train,
    ):
        return self.layernorm(
            self.lstm.sequence(start_hiddens, seq_ends, seq_x, train))

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
        obs, ball_ob = obs.pop('ball')
        obs, steps_remaining_ob = obs.pop('stepsRemaining')

        self_ob = jnp.concatenate([
            self_ob,
            ball_ob,
            steps_remaining_ob,
        ], axis=-1)
        
        obs, team_ob = obs.pop('team')
        obs, enemy_ob = obs.pop('enemy')
        
        assert len(obs) == 0

        return FrozenDict({
            'self': self_ob, 
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
        rnn = PolicyLSTM(
            num_hidden_channels = 256,
            num_layers = 1,
            dtype = dtype,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = CriticNet(dtype, use_simple=False),
        rnn = PolicyLSTM(
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

    policy = ActorCritic(
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

    return policy, obs_preprocess
