import unittest

from snake_dqn.direction import Direction


class DirectionTests(unittest.TestCase):
    def test_turns_left_and_right(self) -> None:
        self.assertEqual(Direction.UP.left(), Direction.LEFT)
        self.assertEqual(Direction.UP.right(), Direction.RIGHT)
        self.assertEqual(Direction.RIGHT.left(), Direction.UP)
        self.assertEqual(Direction.RIGHT.right(), Direction.DOWN)
        self.assertEqual(Direction.DOWN.left(), Direction.RIGHT)
        self.assertEqual(Direction.DOWN.right(), Direction.LEFT)
        self.assertEqual(Direction.LEFT.left(), Direction.DOWN)
        self.assertEqual(Direction.LEFT.right(), Direction.UP)
