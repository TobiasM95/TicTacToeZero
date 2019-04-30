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

TRAIN_CYCLES = 2
GAMES_PER_UPDATE = 10
GAME_BUFFER_SIZE = 400

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
        net.save_net("_init")
    #load all exisiting game state transitions into memory (up to a maximum of 400 games)
    net_game_dir = GAME_DIR / sys.argv[1]
    if not net_game_dir.is_dir():
        net_game_dir.mkdir(parents=True)
    game_buffer = []
    num_of_all_games = -1
    for c in reversed(sorted(net_game_dir.iterdir())):
        if c.is_file() and c.suffix() == ".game":
            if num_of_all_games < 0:
                num_of_all_games = int(str(c)[-5].split("_")[-1])
            games = pickle.load(str(c))
            for g in games:
                game_buffer.append(g)
            if len(game_buffer) >= 400:
                game_buffer = game_buffer[-400:]
                break
    
    for _ in range(TRAIN_CYCLES):
        #generate 10 new games and replace earliest 10 games with new games (or add if not enough)
        #train for a while (dataset has 400*81=32k)
        for k in range(GAMES_PER_UPDATE):
            game_buffer.append(g.play_game(net))
        if len(game_buffer) >= 400:
            game_buffer = game_buffer[-400:]
        #prepare buffer for training session (needs numpy array of shape n_tx9x9x9)
        #train network
        

if __name__ == "__main__":
    sys.exit(int(main() or 0))
