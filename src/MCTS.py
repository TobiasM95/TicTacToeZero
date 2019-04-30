#!/usr/bin/env python3
import numpy as np
from scipy.special import softmax

NUMBER_SIMULATIONS = 10 #agz is 800
MCTS_TEMPERATURE = 1.0

#states are convnet states
#actions are cnn output action vectors hot state
class Node():

    def __init__(self, state=None, action=None, prior=1.0):
        #root node has no parent and no action
        self.action = action
        self.state = state
        self.visit_count = 0
        self.total_val = 0
        self.mean_val = 0
        self.prior = prior
        self.children = []
        self.to_play = -1

    def update(self, value):
        self.visit_count += 1
        self.total_val += value
        self.mean_val = self.total_val / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

    def get_depth(self):
        depth = 0
        clist = [c for c in self.children]
        while(len(clist) > 0):
            clist = [c for p in clist for c in p.children ]
            depth += 1
        return depth

    def print_node(self, note = ""):
        print("Node details of", note, ":")
        if self.action is not None:
            ai = np.where(self.action == 1)[0]
        else:
            ai = None
        print("Action index:", ai)
        print("State (convnet state, show current position):")
        print(self.state[:,:,-3] + self.state[:,:,-2]*2)
        print("(N, W, Q, P) = (",
              self.visit_count, ",",
              self.total_val, ",",
              self.mean_val, ",",
              self.prior, ")")
        print("To play", self.to_play)
        print("Number of children", len(self.children))
        print("Turns of children", [np.where(c.action == 1)[0] for c in self.children])
        print("Max depth of node", self.get_depth())

def run_mcts(game, network, print_root = False):
    root = Node(state = np.copy(game.get_convnet_input()))

    for _ in range(NUMBER_SIMULATIONS):
        sim_game = game.copy_game()
        node = root
        search_path = [node]

        terminal = False
        while(node.is_expanded()):
            action, node = select_mcts_action(node)
            sim_game.move(sim_game.cnn_action_to_coords(action))
            node.state = sim_game.get_convnet_input()
            search_path.append(node)
            if sim_game.state != -1:
                terminal = True

        value = evaluate_node_and_expand(node, network, sim_game, terminal)

        for node in reversed(search_path):
            v = value if node.to_play == sim_game.turn - 1 else (1 - value)
            node.update(v)
    if print_root:
        root.print_node("Root")
    return choose_action(root), generate_mcts_policy(root)

def select_mcts_action(node):
    #UCTS? algorithm choice
    best_choice = (0, -1.0*np.inf)
    for i, child in enumerate(node.children):
        #in alpha zero paper this is described as slowly(log) growing exploration rate
        #set this here as constant for now
        PUCT_C = 1.0
        PUCT_U = PUCT_C*child.prior*np.sqrt(node.visit_count)/(1+child.visit_count)
        action_value = child.mean_val + PUCT_U
        if action_value >= best_choice[1]:
            best_choice = (i, action_value)

    return node.children[best_choice[0]].action, node.children[best_choice[0]]

def evaluate_node_and_expand(node, network, sim_game, terminal):
    if not terminal:
        #check where nodes stay in existence??
        policy_logits, value  = network.evaluate(
            node.state.reshape(1,9,9,2*sim_game.NUMBER_OF_SAVED_GAME_STATES + 1))
        #value.shape = (1,1) policy_logits.shape = (1,81)
    else:
        if sim_game.state == 0:
            return 0.5
        else:
            return 0
        
    # Expand the node.
    node.to_play = sim_game.turn - 1
    legal_move_indices = sim_game.get_legal_move_indices()
    policy = np.array([np.exp(policy_logits[0,i]) for i in legal_move_indices])
    policy_sum = np.sum(policy)
    for index, p in zip(legal_move_indices, policy):
        action = np.zeros(81)
        action[index] = 1
        node.children.append(Node(state=None, action=action, prior=p / policy_sum))
    return value

def choose_action(root):
    action = np.zeros(81)
    visit_counts = np.array([child.visit_count for child in root.children])
    scaled_counts = np.power(visit_counts, 1.0/MCTS_TEMPERATURE)
    scaled_counts[scaled_counts == 0] = -np.inf
    scaled_counts = softmax(scaled_counts)
    #create softmax distribution
    #choose random move from distribution
    #but for now just pick max move
    index = np.random.choice(scaled_counts.shape[0], 1, p=scaled_counts)[0]
    #print("MCTS", scaled_counts.shape, scaled_counts, len(root.children), index)
    #index = np.argmax(scaled_counts)
    return root.children[index].action
    
def generate_mcts_policy(root):
    pol = np.zeros(81)
    for child in root.children:
        pol[np.where(child.action == 1)[0]] = child.prior
    return pol
