# Student agent: Add your own agent here
from copy import deepcopy

import numpy as np

from agents.agent import Agent
from store import register_agent
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
        self.max = max_step
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        my_r, my_c = my_pos
        adv_r, adv_c = adv_pos
        my_walls = self.find_walls(chess_board, my_pos)
        adv_walls = self.find_walls(chess_board, adv_pos)

        state_queue = [(my_pos, 0)]
        visited = {tuple(my_pos)}

        while state_queue:
            cur = state_queue.pop(0)
            cur_pos, cur_step = cur
            print(cur)
            r,c = tuple(cur_pos)
            if cur_step == max_step:
                break
            if self.find_walls(chess_board, cur_pos) == 0:
                my_pos = (r,c)
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue
                m_r, m_c = move
                next_pos = (r+m_r, c+m_c)
                if next_pos == adv_pos or tuple(next_pos) in visited:
                    continue

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
        return my_pos, dir

        # for i in range(4):
        #     if not chess_board[adv_r, adv_c, i]:
        #         m_r, m_c = moves[i]
        #         if adv_r + m_r - my_r + adv_c + m_c - my_c <= max_step:
        #             my_pos = (adv_r + m_r, adv_c + m_c)
        #             dir = (i+2) % 4
        #             break
