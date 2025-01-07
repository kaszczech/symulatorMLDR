import jax
import jax.numpy as jnp
from mldr_sim.constants import *


def channel_state_selector(actions):
    """
    function return a state of the channel due to how many STA transmit in the same time
    :param actions: vector of stations that transmit at a given time
    :return:
    int: -1 if more than one STA transmit, 1 if exactly one STA transmit, 0 if noone transmit at the moment.
    """

    ones_count = jnp.sum(actions)
    return jnp.where(ones_count > 1, -1,
                     jnp.where(ones_count == 1, 1, 0))


def buffer_clearing(buffer_states, actions):
    """
    Updates the buffer_states vector based on the action vector.
    Sets the value to 0 in buffer_states where there is a 1 in both vectors.
    :param buffer_states: vector with binary value of STAs' buffer occupation.
        0 - STA's buffer is empty
        1 - STA's buffer is full
    :param actions: binary vector describes STAs' actions
        0 - channel sensing
        1 - transmission
    :return:
        jnp.ndarray: updated buffer_states.
    """
    return jnp.where((buffer_states == 1) & (actions == 1), 0, buffer_states)


def add_new_frames(buffer_states, new_frames):
    """
    Updates the buffer_states by adding new frames from generator
    :param buffer_states: vector with binary value of STAs' buffer occupation.
    :param new_frames: vector with binary value which STA generates new frames.
    :return:
        jnp.ndarray: updated buffer_states.
    """
    return jnp.bitwise_or(buffer_states, new_frames)


def no_transmission(args):
    buffer_state, r, channel_state = args
    reward = 0.0
    return reward, buffer_state, r


def transmission(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(channel_state == 0, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(r < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(args):
    buffer_state, r, channel_state = args
    buffer_state = 0
    r = 0
    reward = COLISSION_PENALTY
    return reward, buffer_state, r


def retransmission(args):
    buffer_state, r, channel_state = args
    reward = COLISSION_PENALTY
    buffer_state = buffer_state
    r = r + 1
    return reward, buffer_state, r


def transmission_without_collision(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(buffer_state == 1, successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(args):
    buffer_state, r, channel_state = args
    reward = 0.0
    return reward, buffer_state, r


def successful_transmission(args):
    buffer_state, r, channel_state = args
    reward = (TX_REWARD / (r + 1) ** 2)
    buffer_state = 0
    r = 0
    return reward, buffer_state, r
