import jax
import jax.numpy as jnp

MAX_RETRANSMISSION = 8
TX_REWARD = 1.0
COLISSION_PENALTY = -1.0


@jax.jit
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


@jax.jit
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


@jax.jit
def add_new_frames(buffer_states, new_frames):
    """
    Updates the buffer_states by adding new frames from generator
    :param buffer_states: vector with binary value of STAs' buffer occupation.
    :param new_frames: vector with binary value which STA generates new frames.
    :return:
        jnp.ndarray: updated buffer_states.
    """
    return jnp.bitwise_or(buffer_states, new_frames)


@jax.jit
def simulate(buffer_states, new_frames, actions):
    channel_state = channel_state_selector(actions)

    buffer_states = jnp.where((channel_state == 1), buffer_clearing(buffer_states, actions),
                              buffer_states)  # buffer handling
    buffer_states = add_new_frames(buffer_states, new_frames)

    return buffer_states, channel_state


@jax.jit
def no_transmission(args):
    buffer_state, r, channel_state = args
    reward = 0.0
    return reward, buffer_state, r


@jax.jit
def transmission(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(channel_state == 0, transmission_without_collision, transmission_with_collision, args)


@jax.jit
def transmission_with_collision(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(r < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


@jax.jit
def max_retransmission_collision(args):
    buffer_state, r, channel_state = args
    buffer_state = 0
    r = 0
    reward = COLISSION_PENALTY
    return reward, buffer_state, r


@jax.jit
def retransmission(args):
    buffer_state, r, channel_state = args
    reward = COLISSION_PENALTY
    buffer_state = buffer_state
    r = r + 1
    return reward, buffer_state, r


@jax.jit
def transmission_without_collision(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(buffer_state == 1, successful_transmission, empty_buffer_transmission, args)


@jax.jit
def empty_buffer_transmission(args):
    buffer_state, r, channel_state = args
    reward = 0.0
    return reward, buffer_state, r


@jax.jit
def successful_transmission(args):
    buffer_state, r, channel_state = args
    reward = (TX_REWARD / (r + 1) ** 2)
    buffer_state = 0
    r = 0
    return reward, buffer_state, r


@jax.jit
def process_rl_output(buffer_states, actions, channel_state, obs_i_t_minus, i):
    """

    :param buffer_states:
    :param actions:
    :param channel_state:
    :param obs_i_t_minus:
    :param i:
    :return:
    """
    obs_i_t, r = obs_i_t_minus

    channel_state = jnp.where(channel_state == -1, 1, channel_state)
    action = actions[i]
    buffer_state = buffer_states[i]
    args = (buffer_state, r, channel_state)
    R_i, buffer_state, r = jax.lax.cond(action == 1, transmission, no_transmission, args)
    buffer_states = buffer_states.at[i].set(buffer_state)

    # update history
    obs_t = jnp.array([buffer_states[i], channel_state])
    obs_i_t = jnp.roll(obs_i_t, -1, axis=0)
    obs_i_t = obs_i_t.at[-1].set(obs_t)
    obs_i_t = (obs_i_t, r)

    return buffer_states, obs_i_t, R_i


# # # Przykład użycia
# buffer_states = jnp.array([1, 0, 1, 0])
# new_frames = jnp.array([0, 1, 0, 1])
# actions = jnp.array([1, 0, 0, 0])
# obs_i_t_minus_1 = (jnp.zeros((5, 2)), 5)  # Historia obserwacji dla węzła i
#
# print("First bufer states:", buffer_states)
# buffer_states, channel_state = simulate(buffer_states, new_frames, actions)
#
# print("Buffer states:", buffer_states)


buffer_states = jnp.array([0, 0, 1, 0])
new_frames = jnp.array([0, 1, 0, 1])
actions = jnp.array([1, 0, 0, 1])
obs_i_t_minus_1 = (jnp.zeros((5, 2)), 2)  # Historia obserwacji dla węzła i

channel_state = 1

buffer_states, obs_i_t, R_i = process_rl_output(buffer_states, actions, channel_state, obs_i_t_minus_1, 0)

print("Updated buffer states:", buffer_states)
print("Channel state:", channel_state)
print("Observation:", obs_i_t)
print("Reward:", R_i)
