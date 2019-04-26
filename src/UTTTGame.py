#!/usr/bin/env python3

import sys
import numpy as np
import pathlib
import NeuralNet
import MCTS
import copy as cp

class utttgame:
    NUMBER_OF_SAVED_GAME_STATES = 4
    def __init__(self):
        #turn player 1 (x) == 1, player 2 (o) == 2
        self.turn = 1
        #board is split in 3x3 array of 3x3 arrays (outer array mirrors inner array)
        #makes for easier indexation and rule checking
        #first two indices are for outer cross, last two for inner cross
        #first entry is row, second is column
        #top left is (0,0) bottom right is (3,3) for inner and outer cross
        #empty field is 0, x is 1, o is 2
        self.board = np.zeros((3,3,3,3), dtype=int)
        self.outer_field_state = np.ones((3,3), dtype=int) * (-1)
        #last move to check rule compliance of next move
        self.last_move = []
        #state gives winner (1 or 2), or draw (0) or still active (-1)
        self.state = -1
        self.last_N_board_states = np.zeros((self.NUMBER_OF_SAVED_GAME_STATES, 3,3,3,3),
                                            dtype=int)

    def get_normalized_win_value(self):
        if state == 0:
            return 0.5
        if state == 1:
            return 1
        if state == 2:
            return 0
        return -1
        
    def copy_game(self):
        copy = utttgame()
        copy.turn = self.turn
        copy.board = np.copy(self.board)
        copy.outer_field_state = np.copy(self.outer_field_state)
        copy.last_move = cp.deepcopy(self.last_move)
        copy.sate = self.state
        copy.last_N_board_states = np.copy(self.last_N_board_states)
        return copy
        
    def move(self, coords):
        if self.state != -1:
            print("Game is already over. State =", self.state)
            return
        if self.check_legality(coords):
            self.board[coords[0], coords[1], coords[2], coords[3]] = self.turn
            self.turn = self.turn % 2 + 1
            self.last_move = coords
            self.update_state()
            for i in range(1, self.NUMBER_OF_SAVED_GAME_STATES):
                self.last_N_board_states[i-1] = np.copy(self.last_N_board_states[i])
            self.last_N_board_states[-1] = np.copy(self.board)
            #self.print_board()
            #print(self.last_move)

    def print_board(self, board = None):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if board is not None:
                            print(board[i,k,j,l], end="")
                        else:
                            print(self.board[i,k,j,l], end="")
                    print(" ", end="")
                print("")
            print("")

    def check_legality(self, coords):
        #first move is always legal
        if len(self.last_move) == 0:
            return True
        #check if outer field isnt already decided
        if self.outer_field_state[coords[0], coords[1]] != -1:
            return False
        #if last_move sent player to decided outer field ignore last_move check
        is_target_field_decided = False
        if self.outer_field_state[self.last_move[2], self.last_move[3]] != -1:
            is_target_field_decided = True
        if not is_target_field_decided:
            if self.last_move[2] != coords[0] or self.last_move[3] != coords[1]:
                return False
        #check if inner field is already placed
        if self.board_get(coords) != 0:
            return False
        return True

    def get_legal_move_indices(self):
        lmi = []
        for i in range(81):
            action = np.zeros(81)
            action[i] = 1
            if self.check_legality(self.cnn_action_to_coords(action)):
                lmi.append(i)
        return np.array(lmi)

    def update_state(self):
        #Updates self.state as well as self.outer_field_state
        #check outer field of last move if decision was made
        small_board = self.board[self.last_move[0], self.last_move[1]]
        self.outer_field_state[self.last_move[0],
                               self.last_move[1]] = self.check_single_board(small_board,
                                                                            self.last_move[2],
                                                                            self.last_move[3])
        #check if enough outer fields are decided for a win or if all outer fields are decided
        if np.sum(self.outer_field_state == 1) >= 5:
            self.state = 1
            return
        elif np.sum(self.outer_field_state == 2) >= 5:
            self.state = 2
            return
        elif np.sum(self.outer_field_state == -1) == 0:
            self.state = 0
            return

    #def check_single_board(self, coords, copied board?):
    def check_single_board(self, board, x, y):

        #check if previous move caused a win on vertical line 
        if board[0][y] == board[1][y] == board [2][y]:
            return self.turn % 2 + 1

        #check if previous move caused a win on horizontal line 
        if board[x][0] == board[x][1] == board [x][2]:
            return self.turn % 2 + 1

        #check if previous move was on the main diagonal and caused a win
        if x == y and board[0][0] == board[1][1] == board [2][2]:
            return self.turn % 2 + 1

        #check if previous move was on the secondary diagonal and caused a win
        if x + y == 2 and board[0][2] == board[1][1] == board [2][0]:
            return self.turn % 2 + 1

        if np.sum(board == 0) == 0:
            return 0
        
        return -1     

    def board_get(self, coords):
        return self.board[coords[0], coords[1], coords[2], coords[3]]

    def flatten_board(self, board):
        #takes 3x3x3x3 board in and returns 9x9 board
        flat = np.zeros((9,9), dtype=int)
        for i in range(3):
            for j in range(3):
                flat[3*i+j] = board[i,:,j,:].reshape(9)
        return flat
    
    def split_flat_board(self, board):
        flat = self.flatten_board(board)
        board_p1 = (flat == 1) * 1
        board_p2 = (flat == 2) * 1
        split = np.zeros((9,9,2), dtype=int)
        split[:,:,0] = board_p1
        split[:,:,1] = board_p2
        return split

    def get_convnet_input(self):
        cnn_input = np.zeros((9,9,2*self.NUMBER_OF_SAVED_GAME_STATES + 1))
        for i in range(self.NUMBER_OF_SAVED_GAME_STATES):
            cnn_input[:,:,(2*i,2*i+1)] = self.split_flat_board(self.last_N_board_states[i])
        #renorm turn variable to 0 (x) and 1 (o)
        cnn_input[:,:,-1] = np.ones((9,9)) * (self.turn - 1)
        return cnn_input

    def rl_move(self, cnn_action):
        #perform action a_t and return ((s_t-N+1, .., s_t-3, s_t-2, s_t-1, s_t),
        #                               a_t,
        #                               r_t,
        #                               (s_t-N+2, .., s_t-2, s_t-1, s_t, s_t+1))
        current_state = self.get_convnet_input()
        self.move(self.cnn_action_to_coords(cnn_action))
        action = cnn_action
        #REWARD HAS TO BE MANUALLY SET AT THE END OF THE GAME (DISCOUNT, check which player won)
        #saving the state helps with figuring out who won to adjust the reward
        reward = self.state
        next_state = self.get_convnet_input()
        return (current_state, action, reward, next_state)

    def flatten_action(self, action_coords):
        action = np.zeros(81)
        index = (9*(3*action_coords[0]
                    + action_coords[2])
                 + 3*action_coords[1]
                 + action_coords[3])
        action[index] = 1
        return action
    
    def cnn_action_to_coords(self, cnn_action):
        index = np.where(cnn_action == 1.0)[0]
        row, column = divmod(index, 9)
        outer_row, inner_row = divmod(row, 3)
        outer_column, inner_column = divmod(column, 3)
        return [int(outer_row),
                int(outer_column),
                int(inner_row),
                int(inner_column)]

def main():
    nn = NeuralNet.neuralnetwork()
    game = utttgame()
    while(game.state == -1):
        action, _ = MCTS.run_mcts(game, nn, print_root=True)
        game.move(game.cnn_action_to_coords(action))
        print(game.board)
    sys.exit(2)
    #count = 0
    stats = np.zeros(3, dtype=int)
    while(True):
        game.move([np.random.randint(3),
                    np.random.randint(3),
                    np.random.randint(3),
                    np.random.randint(3)])
        #print(nn.evaluate(game.get_convnet_input().reshape(-1,9,9,9)))
        #count += 1
        if game.state != -1:
            stats[game.state] += 1
            for i in range(game.NUMBER_OF_SAVED_GAME_STATES):
                print("last",i,"board:", game.last_N_board_states[i].shape)
                game.print_board(game.last_N_board_states[i])
                print(game.flatten_board(game.last_N_board_states[i]))
                print(game.split_flat_board(game.last_N_board_states[i]))
            print(game.get_convnet_input(), "test")    
            game.__init__()
            print(stats)
            sys.exit(2)

if __name__ == "__main__":
    sys.exit(int(main() or 0))


