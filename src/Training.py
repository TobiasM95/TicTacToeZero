#!/usr/bin/env python3

import sys
import os
import numpy as np
import pathlib
import pickle

import keras

import UTTTGame as g
import NeuralNet as nn
import MCTS as mc

SCRIPT_DIR = pathlib.Path(sys.path[0])
PROJECT_DIR = SCRIPT_DIR.parent
NETWORK_DIR = PROJECT_DIR / "trained_networks"
GAME_DIR = PROJECT_DIR / "game_data"

TRAIN_CYCLES = 20
TRAIN_EPOCHS = 10
GAMES_PER_UPDATE = 20
GAME_BUFFER_SIZE = 20

def main():
    if len(sys.argv) <= 1:
        raise ValueError("Specify network name!")
    #init network or load exisiting one
    if (NETWORK_DIR / sys.argv[1]).is_file():
        print("Load pre-existing network...")
        net = nn.neuralnetwork(NETWORK_DIR / sys.argv[1], init=False)
    else:
        print("Initialize new network...")
        net = nn.neuralnetwork(NETWORK_DIR / sys.argv[1])
        if not NETWORK_DIR.is_dir():
            NETWORK_DIR.mkdir(parents=True)
        net.save_net("_init")
    #load all exisiting game state transitions into memory (up to a maximum of 400 games)
    net_game_dir = GAME_DIR / sys.argv[1]
    if not net_game_dir.is_dir():
        net_game_dir.mkdir(parents=True)
    game_buffer = []
    num_of_all_games = 0
    for child in reversed(sorted(net_game_dir.iterdir())):
        c = pathlib.Path(child)
        if c.is_file() and c.suffix == ".game":
            if num_of_all_games <= 0:
                num_of_all_games = int(str(c)[:-5].split("_")[-1])
            games = pickle.load(open(str(c), "rb"))
            for game in games:
                game_buffer.append(game)
            if len(game_buffer) >= GAME_BUFFER_SIZE:
                game_buffer = game_buffer[-GAME_BUFFER_SIZE:]
                break

    game_emu = g.utttgame()
    
    for _ in range(TRAIN_CYCLES):
        #generate 10 new games and replace earliest 10 games with new games (or add if not enough)
        #train for a while (dataset has 400*81=32k)
        for k in range(GAMES_PER_UPDATE):
            game_buffer.append(game_emu.play_game(net))
        num_of_all_games += GAMES_PER_UPDATE
        save_path = str(net_game_dir
                             / ("games_"
                                + str(num_of_all_games))) + ".game"
        pickle.dump(game_buffer[-GAMES_PER_UPDATE:], open(save_path,"wb"))
        if len(game_buffer) >= GAME_BUFFER_SIZE:
            game_buffer = game_buffer[-GAME_BUFFER_SIZE:]
        #prepare buffer for training session (needs numpy array of shape n_tx9x9x9)
        input_states = []
        target_policies = []
        target_values = []
        for game in game_buffer:
            for move in game:
                input_states.append(move[0])
                target_policies.append(move[1])
                target_values.append(move[2])
        input_states = np.vstack(input_states)
        target_policies = np.vstack(target_policies)
        target_values = np.vstack(target_values)
        #train network
        net.nn.fit(x = input_states,
                   y = [target_policies, target_values],
                   batch_size = int(input_states.shape[0]/100.0),
                   epochs = TRAIN_EPOCHS,
                   verbose = 1)
        net.save_net()

if __name__ == "__main__":
    sys.exit(int(main() or 0))
