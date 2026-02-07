import sys
import os
import pygame
import pickle
import neat
import math
import csv
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
local_dir = os.path.dirname(__file__)
pickle_path = os.path.join(local_dir, 'winner.pkl')
# pickle_path = os.path.join(local_dir, 'winner-pureNEAT.pkl')
                           
sys.path.append(parent_dir)

from main_2048 import GAME2048
from expectimax_search import ExpectimaxSearch

GRID_SIZE = 4
TILE_SIZE = 100
GAP = 10
HEADER_HEIGHT = 80
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1)*GAP
T_WIN_SIZE = WINDOW_SIZE + HEADER_HEIGHT

pygame.init()

USE_SEARCH = True
# USE_SEARCH = False 


screen = pygame.display.set_mode((WINDOW_SIZE, T_WIN_SIZE))
title = "NEAT 2048 - WITH SEARCH" if USE_SEARCH else "NEAT 2048 - Pure Network"
pygame.display.set_caption(title)

config_path = os.path.join(local_dir, 'config-feedforward.txt')
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

with open(pickle_path, 'rb') as f:
    winner = pickle.load(f)
    print("Winner fitness:", winner.fitness)
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Initialize expectimax searcher
SEARCH_DEPTH = 3
searcher = ExpectimaxSearch(max_depth=SEARCH_DEPTH) if USE_SEARCH else None


class GameAdapter:
    def __init__(self, game2048_instance=None, grid=None, score=None):
        if game2048_instance is not None:
            self.game = game2048_instance
            self.grid = game2048_instance.grid
            self.score = game2048_instance.score
        else:
            from main_2048 import GAME2048
            self.game = GAME2048()
            self.grid = [row[:] for row in grid] if grid else [[0]*4 for _ in range(4)]
            self.score = score if score is not None else 0
            self.game.grid = self.grid
            self.game.score = self.score
    
    def move(self, direction):
        """Convert direction index to string and call game.move"""
        direction_map = ["left", "right", "up", "down"]
        result = self.game.move(direction_map[direction])
        
        self.grid = self.game.grid
        self.score = self.game.score
        return result
    
    def is_game_over(self):
        return self.game.is_game_over()
    
    def get_state(self):
        """Flatten the grid for neural network input"""
        return [cell for row in self.grid for cell in row]
    
    def copy(self):
        """Create a copy of this adapter"""
        return GameAdapter(grid=self.grid, score=self.score)
    
    def __getattr__(self, name):
        """Forward any other attribute access to the wrapped game"""
        return getattr(self.game, name)


def get_enhanced_inputs(game_adapter):
    """Get enhanced input features - must match training (24 inputs)"""
    state = game_adapter.get_state()
    
    # Normalized board (16 inputs)
    normalized_board = [0.0 if x == 0 else math.log2(x) / 11.0 for x in state]
    
    max_tile = max(state) if state else 2
    max_tile_feature = [math.log2(max_tile) / 11.0 if max_tile > 0 else 0]
    
    empty_count = sum(1 for x in state if x == 0)
    empty_feature = [empty_count / 16.0]
    
    available_moves = []
    for direction in ["left", "right", "up", "down"]:
        old_grid = [row[:] for row in game_adapter.grid]
        old_score = game_adapter.score
        can_move = game_adapter.game.move(direction)
        game_adapter.grid = old_grid
        game_adapter.game.grid = old_grid
        game_adapter.score = old_score
        game_adapter.game.score = old_score
        available_moves.append(1.0 if can_move else 0.0)
    
    smoothness_val = calc_smoothness(game_adapter)
    smoothness = [smoothness_val / 100.0]
    
    monotonicity_val = count_monotonicity(game_adapter)
    monotonicity = [monotonicity_val / 8.0]
    
    return normalized_board + max_tile_feature + empty_feature + available_moves + smoothness + monotonicity


def calc_smoothness(game_adapter):
    """Calculate board smoothness"""
    board = game_adapter.grid
    smoothness = 0
    temp_board = [row[:] for row in board]
    
    for rotation in range(2):
        for i in range(len(temp_board)):
            for j in range(len(temp_board[i])-1):
                if temp_board[i][j] != 0 and temp_board[i][j+1] != 0:
                    current_smoothness = math.fabs(math.log2(temp_board[i][j]) - math.log2(temp_board[i][j+1]))
                    smoothness = smoothness - current_smoothness
        
        temp_board = [[temp_board[len(temp_board)-1-j][i] for j in range(len(temp_board))] 
                     for i in range(len(temp_board[0]))]
    
    return smoothness


def count_monotonicity(game_adapter):
    board = game_adapter.grid
    monotonicity = 0
    
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

def run_experiment(num_games=100):
    # Create benchmarks directory if it doesn't exist
    os.makedirs("benchmarks", exist_ok=True)
    
    # Prepare CSV file path
    search_type = "with_search" if USE_SEARCH else "pure_network"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmarks/neat_{search_type}_results_{timestamp}.csv"
    
    # Header for the CSV file
    headers = ['Game_Number', 'Highest_Tile', 'Final_Score', 'Time_Seconds', 'Number_of_Moves', 'Success']
    
    # List to store results
    results = []
    
    print(f"Starting {num_games} games...")
    print(f"Using {'NEAT with Expectimax Search' if USE_SEARCH else 'Pure NEAT Network'}")
    
    completed_games = 0
    
    while completed_games < num_games:
        game_num = completed_games + 1
        print(f"Running game {game_num}/{num_games}")
        
        # Initialize game without pygame display
        game = GAME2048()
        game.add_random_tile()
        game.add_random_tile()
        
        # Wrap game in adapter for expectimax
        game_adapter = GameAdapter(game)

        clock = pygame.time.Clock()
        running = True
        auto_play = True
        consecutive_not_moved = 0
        NOT_MOVED_THRESHOLD = 10

        # Start timing for this specific game
        start_time = time.time()

        while running and not game.game_over:
            dt = clock.tick(120) / 1000.0  # Higher FPS for faster execution

            # Process events without drawing anything for speed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return  # Exit the entire experiment

            if auto_play and not game.game_over and not game.moving_animation:
                if USE_SEARCH:
                    game_adapter.grid = game.grid
                    game_adapter.score = game.score
                    
                    # Use expectimax search
                    direction_idx = searcher.search(game_adapter, net, get_enhanced_inputs)
                    direction = ["left", "right", "up", "down"][direction_idx]
                    moved = game.move(direction)
                else:
                    game_adapter.grid = game.grid
                    game_adapter.score = game.score
                    inputs = get_enhanced_inputs(game_adapter)
                    output = net.activate(inputs)
                    
                    output_moves = [(i, output[i]) for i in range(len(output))]
                    output_moves.sort(key=lambda x: x[1], reverse=True)
                    
                    moved = False
                    for direction_idx, weight in output_moves:
                        direction = ["left", "right", "up", "down"][direction_idx]
                        moved = game.move(direction)
                        
                        if moved:
                            break
                
                if moved:
                    consecutive_not_moved = 0
                    # Only print occasionally to reduce output overhead
                    if game.moves % 50 == 0:
                        max_tile = max(max(row) for row in game.grid)
                        # print(f"Move {game.moves}, Score: {game.score}, Max Tile: {max_tile}")
                else:
                    consecutive_not_moved += 1
                    if consecutive_not_moved >= NOT_MOVED_THRESHOLD:
                        game.game_over = True
                        break

                if game.is_game_over() and not game.game_over:
                    game.game_over = True
                    break

            if not game.game_over:
                game.update_animation(dt)

            game.draw(screen)
            pygame.display.flip()

        # Calculate game duration
        game_time = time.time() - start_time
        
        # Record game statistics
        highest_tile = max(max(row) for row in game.grid)
        final_score = game.score
        num_moves = game.moves
        success = highest_tile >= 2048
        
        result = {
            'Game_Number': game_num,
            'Highest_Tile': highest_tile,
            'Final_Score': final_score,
            'Time_Seconds': round(game_time, 2),
            'Number_of_Moves': num_moves,
            'Success': success
        }
        
        results.append(result)
        print(f"Game {game_num}: Highest Tile={highest_tile}, Score={final_score}, Time={game_time:.2f}s, Moves={num_moves}, Success={success}")
        
        completed_games += 1
    
    # Write results to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nBenchmarks completed! Results saved to {csv_filename}")
    
    # Print summary statistics
    successful_games = sum(1 for r in results if r['Success'])
    avg_score = sum(r['Final_Score'] for r in results) / len(results)
    avg_time = sum(r['Time_Seconds'] for r in results) / len(results)
    avg_moves = sum(r['Number_of_Moves'] for r in results) / len(results)
    
    print(f"\nSummary:")
    print(f"Total games played: {len(results)}")
    print(f"Successful games (reached 2048): {successful_games}")
    print(f"Success rate: {successful_games/len(results)*100:.2f}%")
    print(f"Average score: {avg_score:.2f}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Average moves: {avg_moves:.2f}")
    
    # Print tile distribution
    tile_counts = {}
    for r in results:
        tile = r['Highest_Tile']
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    print("\nTile distribution:")
    for tile in sorted(tile_counts.keys(), reverse=True):
        print(f"  {tile}: {tile_counts[tile]} games")
        
if __name__ == "__main__":
    # main()
    run_experiment(100)