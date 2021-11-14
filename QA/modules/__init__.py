import enum

from .instruction import Instruction
from .gat import GATLayer
from .nsm import NSMLayer
from .entity_encoder import EntityInit


class GraphEncType(enum.Enum):
    GAT = 0
    NSM = 1
    MIX = 2
