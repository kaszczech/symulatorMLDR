import unittest
from mldr_sim.process_rl_output import *


class TestUtils(unittest.TestCase):
    def test_channel_state_selector(self):
        # Test no actions
        actions = jnp.array([0, 0, 0, 0])
        self.assertEqual(channel_state_selector(actions), 0)

        # Test single action
        actions = jnp.array([1, 0, 0, 0])
        self.assertEqual(channel_state_selector(actions), 1)

        # Test multiple actions
        actions = jnp.array([1, 1, 0, 0])
        self.assertEqual(channel_state_selector(actions), -1)

    def test_buffer_clearing(self):
        # Test clearing buffers when actions match
        buffer_states = jnp.array([1, 0, 1, 0])
        actions = jnp.array([1, 0, 0, 0])
        expected = jnp.array([0, 0, 1, 0])
        result = buffer_clearing(buffer_states, actions)
        self.assertTrue(jnp.array_equal(result, expected))

        # Test no clearing when no matching actions
        buffer_states = jnp.array([1, 0, 1, 0])
        actions = jnp.array([0, 0, 0, 0])
        expected = jnp.array([1, 0, 1, 0])
        result = buffer_clearing(buffer_states, actions)
        self.assertTrue(jnp.array_equal(result, expected))

    def test_add_new_frames(self):
        # Test adding new frames
        buffer_states = jnp.array([1, 0, 1, 0])
        new_frames = jnp.array([0, 1, 0, 0])
        expected = jnp.array([1, 1, 1, 0])
        result = add_new_frames(buffer_states, new_frames)
        self.assertTrue(jnp.array_equal(result, expected))

        # Test no new frames added
        buffer_states = jnp.array([1, 0, 1, 0])
        new_frames = jnp.array([0, 0, 0, 0])
        expected = jnp.array([1, 0, 1, 0])
        result = add_new_frames(buffer_states, new_frames)
        self.assertTrue(jnp.array_equal(result, expected))

    def test_no_transmission(self):
        args = (1, 5, 0)
        reward, buffer_state, r = no_transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(buffer_state, 1)
        self.assertEqual(r, 5)

    def test_transmission_without_collision_empty_buffer(self):
        args = (0, 3, 0)
        reward, buffer_state, r = transmission_without_collision(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 3)

    def test_transmission_without_collision_successful(self):
        args = (1, 2, 0)
        reward, buffer_state, r = transmission_without_collision(args)
        self.assertAlmostEqual(reward, TX_REWARD / (2 + 1) ** 2)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 0)

    def test_transmission_with_collision_retransmission(self):
        args = (1, 2, 1)
        reward, buffer_state, r = transmission_with_collision(args)
        self.assertEqual(reward, COLISSION_PENALTY)
        self.assertEqual(buffer_state, 1)
        self.assertEqual(r, 3)

    def test_transmission_with_collision_max_retransmission(self):
        args = (1, MAX_RETRANSMISSION, 1)
        reward, buffer_state, r = transmission_with_collision(args)
        self.assertEqual(reward, COLISSION_PENALTY)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 0)

    def test_transmission_collision_path(self):
        args = (1, 7, 1)
        reward, buffer_state, r = transmission(args)
        self.assertEqual(reward, COLISSION_PENALTY)
        self.assertEqual(buffer_state, 1)
        self.assertEqual(r, 8)

    def test_transmission_successful_path(self):
        args = (1, 1, 0)
        reward, buffer_state, r = transmission(args)
        self.assertAlmostEqual(reward, TX_REWARD / (1 + 1) ** 2)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 0)

    def test_transmission_empty_buffer(self):
        args = (0, 1, 0)
        reward, buffer_state, r = transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 1)


if __name__ == "__main__":
    unittest.main()
