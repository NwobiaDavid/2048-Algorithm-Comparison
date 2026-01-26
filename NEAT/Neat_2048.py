# neat_2048.py
import neat
import pickle
import math
from game_logic import Game2048Logic


def normalize_state(state):
    return [0.0 if x == 0 else math.log2(x) / 11.0 for x in state]

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game2048Logic()
    moves = 0
    max_moves = 2000  # prevent infinite loops

    while not game.game_over and moves < max_moves:
        state = game.get_state()
        # Normalize input (log2 for better scale; 0 → 0)
        # inputs = [0 if x == 0 else x.bit_length() - 1 for x in state]  # log2(x) approx via bit_length
        inputs = normalize_state(state)
        
        output = net.activate(inputs)
        direction = output.index(max(output))  # choose action with highest output

        moved = game.move(direction)
        if not moved:
            # Penalize useless moves slightly
            pass
        moves += 1

    # Fitness: combination of score and max tile
    # fitness = game.score + game.get_max_tile()
    max_tile = game.get_max_tile()
    fitness = game.score + 100 * max_tile + 0.5 * game.moves
    
    print(f"Genome ended with max tile: {game.get_max_tile()}, score: {game.score}")
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)