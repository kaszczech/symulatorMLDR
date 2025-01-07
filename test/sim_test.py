import unittest
from mldr_sim.sim import *


class TestSimulate(unittest.TestCase):

    def test_simulate(self):
        # Test full simulation
        buffer_states = jnp.array([1, 1, 1, 0])
        new_frames = jnp.array([0, 1, 1, 1])
        actions = jnp.array([1, 0, 1, 0])

        expected_buffer_states = jnp.array([0, 1, 1, 1])
        expected_channel_state = 1

        result_buffer_states, result_channel_state = simulate(buffer_states, new_frames, actions)
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)

        # Test simulation with no actions
        buffer_states = jnp.array([1, 0, 1, 0])
        new_frames = jnp.array([0, 1, 0, 1])
        actions = jnp.array([0, 0, 0, 0])

        expected_buffer_states = jnp.array([1, 1, 1, 1])
        expected_channel_state = 0

        result_buffer_states, result_channel_state = simulate(buffer_states, new_frames, actions)
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)


if __name__ == "__main__":
    unittest.main()
