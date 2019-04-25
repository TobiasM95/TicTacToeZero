#!/usr/bin/env python3

NUMBER_SIMULATIONS = 100
MCTS_TEMPERATURE

#states are convnet states
#actions are cnn output action vectors hot state
class Node():

    def __init__(self, state, action=None, prior=1.0):
        #root node has no parent and no action
        self.action = action
        self.state = state
        self.visit_count = 0
        self.total_val = 0
        self.mean_val = 0
        self.prior = prior
        self.children = []

    def update(self, value):
        self.visit_count += 1
        self.total_val += value
        self.mean_val = self.total_val / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

def run_mcts(self, game, network):
    root = Node(game.get_convnet_input())

    for _ in range(NUMBER_SIMULATIONS):
        sim_game = game.copy_game()
        node = root
        search_path = [node]

        while(node.is_expanded):
            action, node = self.select_mcts_action(node)
            sim_game.move(sim_game.cnn_action_to_coords(action))
            search_path.append(node)

        value = self.evaluate_node_and_expand(node, network)

        for node in reversed(search_path):
            node.update(value)

    return self.choose_action(root), self.generate_mcts_policy(root)
#def copy game in utttgame
def select_mcts_action(node):
    #UCTS? algorithm choice

def evaluate_node_and_expand(node, network):
    #check where nodes stay in existence??

def choose_action(self, root):
    action = np.zeros(81)
    visit_counts = np.array([child.visit_count for child in root.children])
    scaled_counts = np.power(visit_counts, 1.0/MCTS_TEMPERATURE)
    #create softmax distribution
    #choose random move from distribution
    #but for now just pick max move
    index = np.argmax(scaled_counts)
    return root.children[index].action
    
def generate_mcts_policy(self, root):
    pol = np.zeros(81)
    for child in root.children:
        pol[np.where(child.action == 1)[0]] = child.prior
    return pol
