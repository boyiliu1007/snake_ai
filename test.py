import torch as pt
import numpy as np
import random
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

class Player:
    def __init__(self, training) -> None:
        self.model = 