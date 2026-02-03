import neat
import pickle
import math
from math import log2
from .expectimax_search import ExpectimaxSearch
from .game_logic import Game2048Logic

def normalize(arr):
    """Normalize the array using log2 scaling"""
    val = max(arr)
    log_val = log2(val) if val > 0 else 1
    if log_val == 0:
        return [0.0] * len(arr)
    normalized = []
    for i in range(len(arr)):
        if arr[i] != 0:
            normalized.append(log2(arr[i]) / log_val)
        else:
            normalized.append(0.0)
    return normalized

def calc_smoothness(game):
    """Calculate board smoothness - sum of adjacent tile differences"""
    board = game.grid
    smoothness = 0
    
    temp_board = [row[:] for row in board]
    
    for rotation in range(2):
        for i in range(len(temp_board)):
            for j in range(len(temp_board[i])-1):
                if temp_board[i][j] != 0 and temp_board[i][j+1] != 0:
                    current_smoothness = math.fabs(log2(temp_board[i][j]) - log2(temp_board[i][j+1]))
                    smoothness = smoothness - current_smoothness
        
        temp_board = [[temp_board[len(temp_board)-1-j][i] for j in range(len(temp_board))] 
                     for i in range(len(temp_board[0]))]

    return smoothness

def count_monotonicity(game):
    """Count monotonic rows/columns - higher means better organization"""
    board = game.grid
    monotonicity = 0
    
    # Check rows
    for row in board:
        increasing = decreasing = True
        for j in range(len(row)-1):
            if row[j] != 0 and row[j+1] != 0:
                if row[j] < row[j+1]:
                    increasing = False
                elif row[j] > row[j+1]:
                    decreasing = False
        if increasing or decreasing:
            monotonicity += 1
    
    # Check columns
    for col in range(len(board[0])):
        increasing = decreasing = True
        for row in range(len(board)-1):
            if board[row][col] != 0 and board[row+1][col] != 0:
                if board[row][col] < board[row+1][col]:
                    increasing = False
                elif board[row][col] > board[row+1][col]:
                    decreasing = False
        if increasing or decreasing:
            monotonicity += 1
    
    return monotonicity

def get_enhanced_inputs(game):
    """Get enhanced input features for the neural network (24 inputs total)"""
    board = game.get_state()
    
    # Normalized board (16 inputs)
    normalized_board = [0.0 if x == 0 else log2(x) / 11.0 for x in board]
    
    # Max tile value (normalized) - 1 input
    max_tile = max(board) if board else 2
    max_tile_feature = [log2(max_tile) / 11.0 if max_tile > 0 else 0]
    
    # Empty tile count (normalized) - 1 input
    empty_count = sum(1 for x in board if x == 0)
    empty_feature = [empty_count / 16.0]
    
    # Available moves (4 binary inputs) - 4 inputs
    available_moves = []
    for direction in range(4):
        old_grid = [row[:] for row in game.grid]
        old_score = game.score
        can_move = game.move(direction)
        game.grid = old_grid
        game.score = old_score
        available_moves.append(1.0 if can_move else 0.0)
    
    # Smoothness (normalized) - 1 input
    smoothness_val = calc_smoothness(game)
    smoothness = [smoothness_val / 100.0]
    
    # Monotonicity (normalized) - 1 input
    monotonicity_val = count_monotonicity(game)
    monotonicity = [monotonicity_val / 8.0]
    
    # Total: 16 + 1 + 1 + 4 + 1 + 1 = 24 inputs
    return normalized_board + max_tile_feature + empty_feature + available_moves + smoothness + monotonicity

def eval_genome(genome, config, use_search=False, search_depth=2):
    """
    Evaluate a genome
    
    Args:
        genome: NEAT genome to evaluate
        config: NEAT config
        use_search: Whether to use expectimax search (slower but better)
        search_depth: Depth of expectimax search (2-3 recommended)
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game2048Logic()
    
    searcher = ExpectimaxSearch(max_depth=search_depth) if use_search else None
    
    game_over = False
    consecutive_not_moved = 0
    NOT_MOVED_RESTART_THRESHOLD = 10
    moves_made = 0
    max_moves = 2000 if use_search else 1500
    max_tile_achieved = 0
    
    while not game_over and moves_made < max_moves:
        # Get enhanced input features (24 inputs)
        in_neurons = get_enhanced_inputs(game)
        
        if use_search:
            # Use expectimax search guided by neural network
            direction = searcher.search(game, net, get_enhanced_inputs)
            old_grid = [row[:] for row in game.grid]
            old_score = game.score
            moved = game.move(direction)
            
            if not moved:
                game.grid = old_grid
                game.score = old_score
        else:
            # Pure neural network approach (faster, for training)
            output = net.activate(in_neurons)
            output_moves = [(i, output[i]) for i in range(len(output))]
            output_moves = sorted(output_moves, key=lambda x: x[1], reverse=True)
            
            moved = False
            for (direction, weight) in output_moves:
                old_grid = [row[:] for row in game.grid]
                old_score = game.score
                
                move_result = game.move(direction)
                
                if move_result:
                    moved = True
                    break
                else:
                    game.grid = old_grid
                    game.score = old_score

        if moved:
            consecutive_not_moved = 0
            moves_made += 1
            current_max = max(cell for row in game.grid for cell in row if cell != 0)
            max_tile_achieved = max(max_tile_achieved, current_max)
        else:
            consecutive_not_moved = consecutive_not_moved + 1

        if game.is_game_over():
            game_over = True
        elif consecutive_not_moved == NOT_MOVED_RESTART_THRESHOLD:
            game_over = True

    return fitness(game, max_tile_achieved, consecutive_not_moved == NOT_MOVED_RESTART_THRESHOLD)

def fitness(game, max_tile_achieved, timedOut=False):
    score = game.score
    smoothness = calc_smoothness(game)
    monotonicity = count_monotonicity(game)
    empty_tiles = sum(1 for row in game.grid for cell in row if cell == 0)
    
    board = game.grid
    corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
    max_tile = max(cell for row in board for cell in row)
    corner_bonus = 50000 if max_tile in corners else 0
    
    tile_bonus = 0
    if max_tile_achieved >= 2048:
        tile_bonus = 1000000000  # 1 BILLION
    elif max_tile_achieved >= 1024:
        tile_bonus = 50000000    # 50 million
    elif max_tile_achieved >= 512:
        tile_bonus = 2000000     # 2 million
    elif max_tile_achieved >= 256:
        tile_bonus = 100000
    elif max_tile_achieved >= 128:
        tile_bonus = 10000
    else:
        tile_bonus = max_tile_achieved ** 2
    
    fitness_value = (
        score * 10 +
        tile_bonus +
        corner_bonus +
        monotonicity * 1000 +
        empty_tiles * 500 +
        smoothness * 100
    )
    
    if timedOut:
        fitness_value *= 0.01
    
    return max(fitness_value, 1.0)

def eval_genomes(genomes, config):
    """Evaluate all genomes - used during training (no search for speed)"""
    for genome_id, genome in genomes:
        try:
            fitness_value = eval_genome(genome, config, use_search=False)
            genome.fitness = fitness_value
        except Exception as e:
            genome.fitness = -float('inf')
            print(f"Error evaluating genome {genome_id}: {e}")