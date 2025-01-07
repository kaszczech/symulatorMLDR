import unittest
from mldr_sim.process_rl_output import *


class TestProcessRlOutput(unittest.TestCase):

    def test_process_rl_output(self):
        buffer_states = jnp.array([0, 0, 1, 0])
        actions = jnp.array([1, 0, 0, 1])
        channel_state = 1
        obs_i_t_minus = jnp.array([
            [0, 0, 6],
            [0, 0, 7],
            [0, 0, 8]
        ])
        i = 0

        expected_buffer_states = jnp.array([0, 0, 1, 0])
        expected_obs_i_t = jnp.array([
            [0, 0, 7],
            [0, 0, 8],
            [0, 1, 0]
        ])
        expected_R_i = -1.0

        result_buffer_states, result_obs_i_t, result_R_i = process_rl_output(
            buffer_states, actions, channel_state, obs_i_t_minus, i
        )

        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertTrue(jnp.array_equal(result_obs_i_t, expected_obs_i_t))
        self.assertEqual(result_R_i, expected_R_i)


if __name__ == '__main__':
    unittest.main()
