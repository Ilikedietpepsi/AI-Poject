from copy import deepcopy


class State():
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

    def legal_actions(self):
        legal_action = []
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        state_queue = [(self.my_pos, 0)]
        visited = {tuple(self.my_pos)}

        while state_queue:
            cur = state_queue.pop(0)
            cur_pos, cur_step = cur
            r, c = tuple(cur_pos)
            if cur_step > self.max_step:
                break

            for dir, move in enumerate(moves):
                if self.chess_board[r, c, dir]:
                    continue
                m_r, m_c = move
                next_pos = (r + m_r, c + m_c)
                if next_pos == self.adv_pos or tuple(next_pos) in visited:
                    continue

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
                for i in range(4):
                    if self.chess_board[r + m_r, c + m_c, i]:
                        continue
                    legal_action.append(r + m_r, c + m_c, i)

        return legal_action

    def move(self, action):
        cur_row, cur_col, wall = action
        new_chess_board = deepcopy(self.chess_board)
        new_chess_board[cur_row, cur_col, wall] = True
        return State(new_chess_board, (cur_row,cur_col), self.adv_pos)

    def is_game_over(self):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        state_queue = [(self.my_pos, 0)]
        visited = {tuple(self.my_pos)}

        while state_queue:
            cur = state_queue.pop(0)
            cur_pos, cur_step = cur
            r, c = tuple(cur_pos)

            for dir, move in enumerate(moves):
                if self.chess_board[r, c, dir]:
                    continue
                m_r, m_c = move
                next_pos = (r + m_r, c + m_c)
                if self.adv_pos == next_pos:
                    return False
                if tuple(next_pos) in visited:
                    continue

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return True

    def game_result(self):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        state_queue1 = [(self.my_pos, 0)]
        visited1 = {tuple(self.my_pos)}

        state_queue2 = [(self.adv_pos, 0)]
        visited2 = {tuple(self.adv_pos)}

        while state_queue1:
            cur = state_queue1.pop(0)
            cur_pos, cur_step = cur
            r, c = tuple(cur_pos)

            for dir, move in enumerate(moves):
                if self.chess_board[r, c, dir]:
                    continue
                m_r, m_c = move
                next_pos = (r + m_r, c + m_c)
                if tuple(next_pos) in visited1:
                    continue

                visited1.add(tuple(next_pos))
                state_queue1.append((next_pos, cur_step + 1))

        while state_queue2:
            cur = state_queue2.pop(0)
            cur_pos, cur_step = cur
            r, c = tuple(cur_pos)

            for dir, move in enumerate(moves):
                if self.chess_board[r, c, dir]:
                    continue
                m_r, m_c = move
                next_pos = (r + m_r, c + m_c)
                if tuple(next_pos) in visited2:
                    continue

                visited2.add(tuple(next_pos))
                state_queue1.append((next_pos, cur_step + 1))
        if (len(visited1) > len(visited2)):
            return 1
        else:
            return -1
