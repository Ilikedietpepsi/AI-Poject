# Agent 0: beats the random player, no AI algorithm.
from copy import deepcopy

from agents.agent import Agent
from store import register_agent
import numpy as np


@register_agent("agent_0")
class Agent0(Agent):

    def __init__(self):
        super(Agent0, self).__init__()
        self.name = "Player0"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        self.opposites = {
            0: 2,
            1: 3,
            2: 0,
            3: 1,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        adv_cell = chess_board[adv_pos[0], adv_pos[1]]
        free_walls_indices = [i for i in range(len(adv_cell)) if not adv_cell[i]]

        # Check if you can reach the neighbouring cell of the opponent and potentially enclosing them
        for ind in free_walls_indices:
            dest_x, dest_y, direction = get_target_cell(adv_pos, ind)
            if check_valid_step(chess_board, max_step, my_pos, (dest_x, dest_y), direction,adv_pos):

                # If the enclosing does not give me less squares
                temp_board = deepcopy(chess_board)
                temp_board[dest_x, dest_y, direction] = True
                move = moves[direction]
                temp_board[dest_x + move[0], dest_y + move[1], self.opposites[direction]] = True
                end, my_score, adv_score = check_endgame(temp_board, (dest_x, dest_y), adv_pos)
                if not end or my_score >= adv_score:
                    return (dest_x, dest_y), direction

        # Else take the max step in some direction
        ori_pos = deepcopy(my_pos)

        h = 0
        while True:
            my_pos = ori_pos
            for _ in range(max_step):
                r, c = my_pos
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

                k = 0
                while chess_board[r, c, dir] or my_pos == adv_pos:
                    k += 1
                    if k > 300:
                        break
                    dir = np.random.randint(0, 4)
                    m_r, m_c = moves[dir]
                    my_pos = (r + m_r, c + m_c)

                if k > 300:
                    my_pos = ori_pos
                    break

            # Put Barrier
            dir = np.random.randint(0, 4)
            r, c = my_pos
            while chess_board[r, c, dir]:
                dir = np.random.randint(0, 4)

            temp_board = deepcopy(chess_board)
            temp_board[my_pos[0], my_pos[1], dir] = True
            # Set the opposite barrier to True
            move = moves[dir]
            temp_board[my_pos[0] + move[0], my_pos[1] + move[1], self.opposites[dir]] = True
            end, my_score, adv_score = check_endgame(temp_board, my_pos, adv_pos)

            if not end or my_score >= adv_score or h > 50:
                return my_pos, dir

            h = h+1


def check_valid_step(chess_board, max_step, start_pos, end_pos, barrier_dir, adv_pos):

    # Endpoint already has barrier or is boarder
    r, c = end_pos
    if chess_board[r, c, barrier_dir]:
        return False
    if np.array_equal(start_pos, end_pos):
        return True

    # BFS
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}
    is_reached = False
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            if chess_board[r, c, dir]:
                continue

            next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached


def get_target_cell(position, available_wall):
    r, c = position
    if available_wall == 0:  # Up
        return r - 1, c, 2
    elif available_wall == 1:  # Right
        return r, c + 1, 3
    elif available_wall == 2:  # Down
        return r + 1, c, 0
    elif available_wall == 3:  # Left
        return r, c - 1, 1
    else:
        return -1


def check_endgame(chessboard, p1, p2):

    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """

    # Union-Find
    father = dict()
    for r in range(len(chessboard)):
        for c in range(len(chessboard)):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2
        father[pos1] = pos2

    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    for r in range(len(chessboard)):
        for c in range(len(chessboard)):
            for dir, move in enumerate(
                    moves[1:3]
            ):  # Only check down and right
                if chessboard[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(len(chessboard)):
        for c in range(len(chessboard)):
            find((r, c))

    p0_r = find(tuple(p1))
    p1_r = find(tuple(p2))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)

    if p0_r == p1_r:
        return False, p0_score, p1_score

    return True, p0_score, p1_score
