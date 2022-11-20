import numpy as np
from collections import defaultdict

class Node():
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.number_of_visits = 0;
        self.results = defaultdict(int)
        self.results[1] = 0
        self.results[-1] = 0
        self.untried_actions = state.legal_actions()
        return



    def score(self):
        wins = self.results[1]
        loses = self.results[-1]
        return wins-loses

    def visits(self):
        return self.number_of_visits;

    def expand(self):
        action = self.untried_actions.pop(0)
        next_state = self.state.move(action)
        child_node = Node(next_state, self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def roll_out(self):
        cur_state = self.state

        while not cur_state.is_game_over():
            possible_moves = cur_state.legal_actions()
            action = self.rollout_policy(possible_moves)
            cur_state = cur_state.move(action)

        return cur_state.game_result()

    def backtrack(self, result):
        self.number_of_visits += 1
        self.results[result] += 1
        if self.parent:
            self.parent.backtrack(result)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.score() / c.visits()) + c_param * np.sqrt((2 * np.log(self.visits()) / c.visits())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        return possible_moves[np.random.randint(len(possible_moves))]

    def tree_policy(self):

        cur_node = self
        while not cur_node.is_terminal_node():

            if not cur_node.is_fully_expanded():
                return cur_node.expand()
            else:
                cur_node = cur_node.best_child()
        return cur_node

    def best_action(self):
        simulation_no = 100

        for i in range(simulation_no):
            v = self.tree_policy()
            result = v.rollout()
            v.backtrack(result)

        return self.best_child()



