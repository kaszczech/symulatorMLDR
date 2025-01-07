from mldr_sim.utils import *


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

    r = obs_i_t_minus[-1][2]
    channel_state = jnp.where(channel_state == -1, 1, channel_state)
    action = actions[i]
    buffer_state = buffer_states[i]
    args = (buffer_state, r, channel_state)
    R_i, buffer_state, r = jax.lax.cond(action == 1, transmission, no_transmission, args)
    buffer_states = buffer_states.at[i].set(buffer_state)

    # update history
    obs_t = jnp.array([buffer_states[i], channel_state, r])
    obs_i_t = jnp.roll(obs_i_t_minus, -1, axis=0)
    obs_i_t = obs_i_t.at[-1].set(obs_t)

    return buffer_states, obs_i_t, R_i
