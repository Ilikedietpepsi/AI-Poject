# Student agent: Add your own agent here
from copy import deepcopy

import numpy as np

from agents.agent import Agent
from node import Node
from store import register_agent
from state import State
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.max = 0
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }


    def find_walls(self, chess_board, pos):
        walls = 0
        pos_r, pos_c = pos
        for i in range(4):
            if chess_board[pos_r, pos_c, i]:
                walls += 1
        return walls





    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        state = State(chess_board, my_pos, adv_pos, max_step)
        node = Node(state)
        return node.best_action()




