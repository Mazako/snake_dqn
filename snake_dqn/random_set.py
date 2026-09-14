from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from random import Random


class RandomSet[T: Hashable]:
    def __init__(self, values: Iterable[T] = ()) -> None:
        self._values: list[T] = []
        self._indices: dict[T, int] = {}

        for value in values:
            self.add(value)

    def __contains__(self, value: object) -> bool:
        return value in self._indices

    def __iter__(self) -> Iterator[T]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, value: T) -> bool:
        if value in self._indices:
            return False

        self._indices[value] = len(self._values)
        self._values.append(value)
        return True

    def discard(self, value: T) -> bool:
        index = self._indices.pop(value, None)
        if index is None:
            return False

        last_value = self._values.pop()
        if index < len(self._values):
            self._values[index] = last_value
            self._indices[last_value] = index

        return True

    def choice(self, rng: Random) -> T:
        if not self._values:
            raise IndexError("cannot choose from an empty RandomSet")

        return self._values[rng.randrange(len(self._values))]

    def pop_random(self, rng: Random) -> T:
        value = self.choice(rng)
        self.discard(value)
        return value
