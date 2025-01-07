from mldr_sim.utils import *


@jax.jit
def simulate(buffer_states, new_frames, actions):
    channel_state = channel_state_selector(actions)

    buffer_states = jnp.where((channel_state == 1), buffer_clearing(buffer_states, actions),
                              buffer_states)  # buffer handling
    buffer_states = add_new_frames(buffer_states, new_frames)

    return buffer_states, channel_state
