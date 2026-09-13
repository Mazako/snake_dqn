from random import Random
import unittest

from random_set import RandomSet


class RandomSetTests(unittest.TestCase):
    def test_add_and_discard(self) -> None:
        values = RandomSet([1, 2, 3])

        self.assertFalse(values.add(2))
        self.assertTrue(values.discard(2))
        self.assertFalse(values.discard(2))
        self.assertEqual(set(values), {1, 3})

    def test_discard_non_last_value_keeps_indices_valid(self) -> None:
        values = RandomSet([1, 2, 3])

        values.discard(1)
        values.discard(3)

        self.assertEqual(list(values), [2])

    def test_pop_random_removes_selected_value(self) -> None:
        values = RandomSet([1, 2, 3])

        value = values.pop_random(Random(42))

        self.assertNotIn(value, values)
        self.assertEqual(len(values), 2)

    def test_choice_from_empty_set_fails(self) -> None:
        with self.assertRaises(IndexError):
            RandomSet().choice(Random())
