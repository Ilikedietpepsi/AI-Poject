# Player 0: beats the random player, no AI algorithm.
from agents.agent import Agent
from store import register_agent


@register_agent("player_0")
class Player0(Agent):

    def __init__(self):
        super(Player0, self).__init__()
        self.name = "Player0"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
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
        # dummy return
        return my_pos, self.dir_map["u"]
