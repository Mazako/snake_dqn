from enum import Enum, auto


class EndReason(Enum):
    COLLISION = auto()
    STARVATION = auto()
    WIN = auto()
