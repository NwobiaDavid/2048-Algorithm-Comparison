# play_ai.py
import sys
import os
import pygame
import pickle
import neat
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
local_dir = os.path.dirname(__file__)
pickle_path = os.path.join(local_dir, 'winner.pkl')
                           
sys.path.append(parent_dir)

from Main_2048 import GAME2048

GRID_SIZE = 4
TILE_SIZE = 100
GAP = 10
HEADER_HEIGHT = 80
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1)*GAP

T_WIN_SIZE = WINDOW_SIZE + HEADER_HEIGHT


# config_path = "config-feedforward.txt"
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

def get_24_inputs(state):
    # First 16: normalized grid values
    normalized_grid = [0.0 if x == 0 else math.log2(x) / 11.0 for x in state]
    
    # Next 8: additional features (example implementations)
    additional_features = []
    
    # Example: max tile positions (row, col) normalized
    max_val = max(state)
    max_idx = state.index(max_val) if max_val > 0 else 0
    max_row, max_col = max_idx // 4, max_idx % 4
    additional_features.extend([max_row/3.0, max_col/3.0])  # 2 features
    
    # Example: empty cell count (normalized)
    empty_count = state.count(0)
    additional_features.append(empty_count / 16.0)  # 1 feature
    
    # Example: monotonicity features (row/column trends)
    grid_2d = [state[i:i+4] for i in range(0, 16, 4)]
    monotonicity_score = calculate_monotonicity(grid_2d)
    additional_features.extend([monotonicity_score, 1-monotonicity_score])  # 2 features
    
    # Example: clustering penalty (adjacent similar tiles)
    clustering_penalty = calculate_clustering_penalty(grid_2d)
    additional_features.append(clustering_penalty)  # 1 feature
    
    # Pad with zeros to reach 24 total features
    remaining_features = 24 - len(normalized_grid) - len(additional_features)
    additional_features.extend([0.0] * remaining_features)
    
    return normalized_grid + additional_features

def calculate_monotonicity(grid):
    # Calculate how monotonic the rows/columns are
    score = 0
    # Horizontal monotonicity
    for row in grid:
        for i in range(len(row)-1):
            if row[i] >= row[i+1]:
                score += 1
    # Vertical monotonicity  
    for c in range(4):
        for r in range(3):
            if grid[r][c] >= grid[r+1][c]:
                score += 1
    return min(score / 24.0, 1.0)  # Normalize

def calculate_clustering_penalty(grid):
    penalty = 0
    for r in range(4):
        for c in range(4):
            if grid[r][c] != 0:
                # Check adjacent cells
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 4 and 0 <= nc < 4 and grid[nr][nc] != 0:
                        penalty += abs(grid[r][c] - grid[nr][nc])
    return min(penalty / 100.0, 1.0)  # Normalize

def normalize_state(state):
    # return [0.0 if x == 0 else math.log2(x) / 11.0 for x in state]
    return get_24_inputs(state)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, T_WIN_SIZE))  # ← Add these if needed
    pygame.display.set_caption("NEAT 2048")
    
    game = GAME2048()  # Already adds two tiles in __init__

    game.add_random_tile()
    game.add_random_tile()

    clock = pygame.time.Clock()
    running = True
    auto_play = True
    consecutive_not_moved = 0
    NOT_MOVED_THRESHOLD = 10
    attempted_directions = []


    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                elif event.key == pygame.K_r and game.game_over:
                    game.reset_game()

        if auto_play and not game.game_over and not game.moving_animation:
            state = game.get_state()
            # inputs = [0 if x == 0 else x.bit_length() - 1 for x in state]
            inputs = normalize_state(state)
            output = net.activate(inputs)
            
            output_moves = [(i, output[i]) for i in range(len(output))]
            output_moves.sort(key=lambda x: x[1], reverse=True)
            
            moved = False
            for direction_idx, weight in output_moves:
                # Avoid immediately repeating the same direction if it didn't work recently
                if len(attempted_directions) >= 2 and direction_idx == attempted_directions[-1]:
                    continue
                    
                direction = ["left", "right", "up", "down"][direction_idx]
                moved = game.move(direction)
                
                if moved:
                    attempted_directions.append(direction_idx)
                    if len(attempted_directions) > 3:
                        attempted_directions.pop(0)
                    break
                
            if moved:
                consecutive_not_moved = 0
                print(f"Score: {game.score}, Max Tile: {max(max(row) for row in game.grid)}")
            else:
                consecutive_not_moved += 1
                if consecutive_not_moved >= NOT_MOVED_THRESHOLD:
                    game.game_over = True
                    print(f"Game Over! Score: {game.score}, Max Tile: {max(max(row) for row in game.grid)}")

            if game.is_game_over() and not game.game_over:
                game.game_over = True
                print(f"Game Over! Score: {game.score}, Max Tile: {max(max(row) for row in game.grid)}")

        if not game.game_over:
            game.update_animation(dt)

        game.draw(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()