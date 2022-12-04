# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
from math import dist


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.autoplay = True
        self.moves_dict = {}
        self.walls = set()

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

        if len(self.moves_dict) == 0:
            # Calculate all legal moves on first move
            self.moves_dict = calculate_all_legal_steps(chess_board, max_step)
        else:
            # Add the adversary's position to the list of walls, except the first move
            for i in range(len(chess_board[adv_pos[0], adv_pos[1]])):
                if chess_board[adv_pos[0], adv_pos[1]][i]:
                    self.walls.add((adv_pos, i))
                    self.walls.add(((adv_pos[0] + self.moves[i][0], adv_pos[1] + self.moves[i][1]), self.opposites[i]))

        root = self.MonteCarloTreeSearchNode(state=chess_board, p1_pos=(my_pos, None), p2_pos=(adv_pos, None),
                                             max_step=max_step, turn=1, legal_pos_dict=self.moves_dict,
                                             a_walls=self.walls)

        selected_node = root.best_action(num_children_to_explore=4, num_game_simulations=1, exploration_parameter=0.6)

        selected_pos = selected_node.get_selected_pos()
        target_pos, direction = selected_pos
        op_move = self.moves[direction]
        self.walls.add(selected_pos)
        self.walls.add(((target_pos[0] + op_move[0], target_pos[1] + op_move[1]), self.opposites[direction]))

        return selected_pos

    class MonteCarloTreeSearchNode:
        def __init__(self, state, p1_pos, p2_pos, max_step, turn, legal_pos_dict, a_walls, parent=None,
                     parent_action=None):
            self.state = state
            self.turn = turn
            self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            self.p1_pos = p1_pos
            self.p2_pos = p2_pos
            self.max_step = max_step
            self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
            self.parent = parent
            self.parent_action = parent_action
            self.children = []
            self._number_of_visits = 0
            self._results = {1: 0, -1: 0, 0: 0}
            self.pos_dict = legal_pos_dict
            self.new_walls = a_walls
            self.index = len(self.new_walls)
            self.eval_function_values = {}
            self._untried_actions = self.untried_actions()
            return

        def untried_actions(self):
            """
            Returns the list of untried actions from a given state.
            """
            self._untried_actions = self.get_legal_positions()
            return self._untried_actions

        def q(self):
            """
            Returns the difference of wins - losses
            """
            wins = self._results[1]
            loses = self._results[-1]
            return wins - loses

        def n(self):
            """
            Returns the number of times each node is visited.
            """
            return self._number_of_visits

        def expand(self):
            """
            From the present state, next state is generated depending on the action which is carried out.
            In this step all the possible child nodes corresponding to generated states are appended to the children
            array and the child_node is returned
            """

            action = self._untried_actions.pop(0)

            while not check_valid_step(self.state, self.p1_pos[0], action[0], action[1], self.max_step, self.p2_pos):
                try:
                    action = self._untried_actions.pop(0)
                except IndexError:
                    return self.best_child(c_param=0.6)

            # New state for the child
            temp_node = deepcopy(self)
            temp_node.move(action)
            child_node = self.__class__(temp_node.state, p1_pos=temp_node.p1_pos, p2_pos=temp_node.p2_pos, parent=self,
                                        max_step=self.max_step, turn=temp_node.turn, parent_action=action,
                                        legal_pos_dict=temp_node.pos_dict, a_walls=temp_node.new_walls)

            self.children.append(child_node)
            return child_node

        def is_terminal_node(self):
            """
            Check if the current node is terminal or not
            """
            return self.is_game_over()[0]

        def rollout(self):
            """
            From the current state, entire game is simulated till there is an outcome for the game.
            This outcome of the game is returned.
            """

            end, p1_score, p2_score = self.is_game_over()
            break_out = False

            while not end:
                possible_moves = self.get_legal_positions()  # Based on the player whose turn it is

                if len(possible_moves) == 0:
                    # In this case the player has lost, has no more moves to make
                    if self.turn:
                        p1_score = -1
                    else:
                        p2_score = -1
                    break

                action = possible_moves[0]  # Try the first move, since it has the highest evaluation func value.

                if self.turn:
                    p = self.p1_pos[0]
                    p1 = self.p2_pos[0]
                else:
                    p = self.p2_pos[0]
                    p1 = self.p1_pos[0]

                w = 0
                while not check_valid_step(self.state, p, action[0], action[1], self.max_step, p1):
                    w = w + 1
                    try:
                        action = possible_moves[w]
                    except IndexError:
                        break_out = True
                        if self.turn:
                            p1_score = -1
                        else:
                            p2_score = -1
                        break

                if break_out:
                    break

                self.move(action)  # Turn switch
                end, p1_score, p2_score = self.is_game_over()

            if p1_score > p2_score:
                return 1
            elif p1_score < p2_score:
                return -1
            else:
                return 0

        def back_propagate(self, result):
            """
            In this step all the statistics for the nodes are updated.
            """
            self._number_of_visits += 1.
            self._results[result] += 1.
            if self.parent:
                self.parent.back_propagate(result)

        def is_fully_expanded(self):
            """
            All the actions are popped out of _untried_actions one by one. When it becomes empty,
            that is when the size is zero, it is fully expanded
            """
            return len(self._untried_actions) == 0

        def best_child(self, c_param):
            """
            Once fully expanded, this function selects the best child out of the children array.
            """

            choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in
                               self.children]
            return self.children[np.argmax(choices_weights)]

        def evaluation_function(self, a_position, adv_position):
            """
            Describes the value of a legal position as its closeness to the adversary position,
            """
            te = deepcopy(self.state)
            target_pose, direction = a_position
            te[target_pose[0], target_pose[1], direction] = True
            # Set the opposite barrier to True
            op_move = self.moves[direction]
            te[target_pose[0] + op_move[0], target_pose[1] + op_move[1], self.opposites[direction]] = True
            _, a_score, adv_score = self.is_game_over(te, a_position, adv_position)
            diff = adv_score - a_score

            if diff != 0:
                return diff
            else:
                return dist(a_position[0], adv_position[0])

        def _tree_policy(self, exploration_parameter):
            """
            Selects node to run rollout.
            """
            current_node = self
            while not current_node.is_terminal_node():
                if not current_node.is_fully_expanded():
                    return current_node.expand()
                else:
                    current_node = current_node.best_child(c_param=exploration_parameter)
            return current_node

        def best_action(self, num_children_to_explore, num_game_simulations, exploration_parameter):
            """
            This is the best action function which returns the node corresponding to best possible move.
            """

            while num_children_to_explore:
                v = self._tree_policy(exploration_parameter)
                if self.eval_function_values[v.parent_action] < 0:
                    return v
                for j in range(num_game_simulations):
                    reward = v.rollout()
                    v.back_propagate(reward)
                    v.new_walls = set(list(v.new_walls)[:v.index])
                num_children_to_explore = num_children_to_explore - 1

            return self.best_child(c_param=exploration_parameter)

        def get_legal_positions(self):

            pos = self.p1_pos if self.turn else self.p2_pos
            adv_pos = self.p1_pos if not self.turn else self.p2_pos

            def f(x):
                val = self.evaluation_function(x, adv_pos)
                self.eval_function_values[x] = val
                return val

            positions = list(
                self.pos_dict[pos[0]] - self.new_walls - {(adv_pos[0], direction) for direction in range(4)})
            positions.sort(key=lambda x: f(x))

            return positions

        def is_game_over(self, state=None, p1=None, p2=None):
            """
            It is the game over condition. Returns true/ false, plus the scores of the players.
            """

            # Union-Find
            father = dict()
            for r in range(len(self.state)):
                for c in range(len(self.state)):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(len(self.state)):
                for c in range(len(self.state)):
                    for d, move in enumerate(
                            self.moves[1:3]
                    ):  # Only check down and right
                        if state is not None:
                            if state[r, c, d + 1]:
                                continue
                        else:
                            if self.state[r, c, d + 1]:
                                continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(len(self.state)):
                for c in range(len(self.state)):
                    find((r, c))
            p0_r = find(tuple(p1[0])) if p1 is not None else find(tuple(self.p1_pos[0]))
            p1_r = find(tuple(p2[0])) if p2 is not None else find(tuple(self.p2_pos[0]))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, p0_score, p1_score
            return True, p0_score, p1_score

        def move(self, new_position):
            """
            Changes the state of the game with a new value for player1_position, player_2position, and the board state.
            Returns the new state after making a move.
            """
            target_pose, direction = new_position

            # Set barrier
            self.state[target_pose[0], target_pose[1], direction] = True
            # Set the opposite barrier to True
            op_move = self.moves[direction]
            self.state[target_pose[0] + op_move[0], target_pose[1] + op_move[1], self.opposites[direction]] = True

            self.new_walls.add(new_position)
            self.new_walls.add(((target_pose[0] + op_move[0], target_pose[1] + op_move[1]), self.opposites[direction]))

            if self.turn:
                self.p1_pos = new_position
            else:
                self.p2_pos = new_position

            # Change turn
            self.turn = 1 - self.turn

            return self.state

        def get_selected_pos(self):
            """
            Returns the best position to move to.
            """
            return self.parent_action


def check_valid_step(board, start_pos, end_pos, barrier_dir, max_steps, adv=None):
    """
    Check if the step the agent takes is valid (reachable and within max steps).

    Parameters
    ----------
    adv
    max_steps
    board
    start_pos : tuple
        The start position of the agent.
    end_pos : tuple
        The end position of the agent.
    barrier_dir : int
        The direction of the barrier.
    """
    # Endpoint already has barrier or is boarder
    r, c = end_pos
    if board[r, c, barrier_dir]:
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
        if cur_step == max_steps:
            break
        for d, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            if board[r, c, d]:
                continue

            next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
            if tuple(next_pos) in visited or (adv and np.array_equal(next_pos, adv[0])):
                continue
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached


def get_neighbours(board, pos, initial_pos, max_steps):
    legal_actions = set()
    for move in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        next_pose = (pos[0] + move[0], pos[1] + move[1])
        if 0 <= next_pose[0] < len(board) and 0 <= next_pose[1] < len(board):
            for direction in range(4):
                if not board[next_pose[0], next_pose[1], direction]:
                    if check_valid_step(board, initial_pos, next_pose, direction, max_steps):
                        legal_actions.add((next_pose, direction))
    return legal_actions


def calculate_all_legal_steps(board, num_steps):
    """
     Constructs a dictionary of all possible actions from all cells.
     """

    legal_positions_dict = {}

    for r in range(len(board)):
        for c in range(len(board)):
            max_steps = num_steps
            legal_positions = set()
            pos = ((r, c), None)
            pos_to_try = set()
            temp = set()
            pos_to_try.add(pos)
            tried_positions = set()

            while max_steps:
                temp.clear()
                for p in pos_to_try:
                    tried_positions.add(p)
                    res = get_neighbours(board, p[0], pos[0], num_steps)
                    legal_positions.update(res)
                    temp.update(res)
                pos_to_try = temp - tried_positions
                max_steps = max_steps - 1
            legal_positions_dict[(r, c)] = legal_positions

    return legal_positions_dict
