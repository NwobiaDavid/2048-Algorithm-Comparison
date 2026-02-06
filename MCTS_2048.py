import pygame
import random
import math
import numpy as np
from numba import jit
import os
import csv
import time

pygame.font.init()

GRID_SIZE = 4
TILE_SIZE = 100
GAP = 10
HEADER_HEIGHT = 80
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1)*GAP
T_WIN_SIZE = WINDOW_SIZE + HEADER_HEIGHT

pygame.init()
pygame.display.set_caption("Monte Carlo 2048")
TILE_FONT = pygame.font.SysFont("comicsans", 32, bold=True)
OVER_FONT = pygame.font.SysFont("comicsans", 48, bold=True)
MOVES_FONT = pygame.font.SysFont("comicsans", 20, bold=True) 
TIMER_FONT = pygame.font.SysFont("comicsans", 20, bold=True)

screen = pygame.display.set_mode((WINDOW_SIZE, T_WIN_SIZE))

TILE_COLORS = {
    0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
}


@jit(nopython=True)
def slide_and_merge_left(board):
    new_board = board.copy()
    score_delta = 0
    
    for row in range(4):
        pos = 0
        for col in range(4):
            if new_board[row, col] != 0:
                new_board[row, pos] = new_board[row, col]
                if pos != col:
                    new_board[row, col] = 0
                pos += 1
        
        
        for col in range(3):
            if new_board[row, col] != 0 and new_board[row, col] == new_board[row, col + 1]:
                new_board[row, col] *= 2
                score_delta += new_board[row, col]
                new_board[row, col + 1] = 0
                
                for k in range(col + 1, 3):
                    new_board[row, k] = new_board[row, k + 1]
                new_board[row, 3] = 0
    
    return new_board, score_delta

@jit(nopython=True)
def make_move_with_score(board, direction):
    """Ultra-fast move function that returns both new board and score delta"""
    score_delta = 0
    
    if direction == 0:  # Up
        rotated = np.rot90(board, k=1)
        moved, score_delta = slide_and_merge_left(rotated)
        return np.rot90(moved, k=-1), score_delta
    elif direction == 1:  # Right
        flipped = np.fliplr(board)
        moved, score_delta = slide_and_merge_left(flipped)
        return np.fliplr(moved), score_delta
    elif direction == 2:  # Down
        rotated = np.rot90(board, k=-1)
        moved, score_delta = slide_and_merge_left(rotated)
        return np.rot90(moved, k=1), score_delta
    else:  # Left (direction == 3)
        moved, score_delta = slide_and_merge_left(board)
        return moved, score_delta

@jit(nopython=True)
def make_move_numba(board, direction):
    """Ultra-fast move function using numba"""
    new_board, _ = make_move_with_score(board, direction)
    return new_board

@jit(nopython=True)
def is_game_over_numba(board):
    for r in range(4):
        for c in range(4):
            if board[r, c] == 0:
                return False
    
    for r in range(4):
        for c in range(3):
            if board[r, c] == board[r, c + 1]:
                return False
    
    for r in range(3):
        for c in range(4):
            if board[r, c] == board[r + 1, c]:
                return False
    
    return True

@jit(nopython=True)
def boards_equal(board1, board2):
    """Fast board equality check"""
    for r in range(4):
        for c in range(4):
            if board1[r, c] != board2[r, c]:
                return False
    return True

@jit(nopython=True)
def count_empty_tiles(board):
    """Count empty tiles in board"""
    count = 0
    for r in range(4):
        for c in range(4):
            if board[r, c] == 0:
                count += 1
    return count

@jit(nopython=True)
def add_random_tile_numba(board, tile_value, random_pos):
    """Add tile at specific random position (position pre-selected)"""
    new_board = board.copy()
    empty_count = 0
    
    for r in range(4):
        for c in range(4):
            if new_board[r, c] == 0:
                if empty_count == random_pos:
                    new_board[r, c] = tile_value
                    return new_board
                empty_count += 1
    
    return new_board


class MonteCarloPlayer:
    """Monte Carlo simulation player based on the C++ implementation"""
    
    def __init__(self, runs_per_move=50, max_simulation_steps=50):
        self.runs_per_move = runs_per_move
        self.max_simulation_steps = max_simulation_steps
        self.direction_names = ["up", "right", "down", "left"]
    
    def simulate_one_run(self, board):
        """From a given game state, choose random moves until the game is completed."""
        board_copy = np.copy(board)
        first_move = -1
        score = 0
        steps = 0
        
        while not is_game_over_numba(board_copy) and steps < self.max_simulation_steps:
            possible_moves = [0, 1, 2, 3]
            random.shuffle(possible_moves)
            
            move_found = False
            for move_dir in possible_moves:
                new_board, move_score = make_move_with_score(board_copy, move_dir)
                
                if not boards_equal(board_copy, new_board):
                    board_copy = new_board
                    score += move_score
                    
                    if first_move == -1:
                        first_move = move_dir
                    
                    empty_count = count_empty_tiles(board_copy)
                    if empty_count > 0:
                        random_pos = random.randint(0, empty_count - 1)
                        tile_value = 2 if random.random() < 0.9 else 4
                        board_copy = add_random_tile_numba(board_copy, tile_value, random_pos)
                    
                    move_found = True
                    break
            
            if not move_found:
                break
                
            steps += 1
        
        if first_move == -1:
            first_move = 0  # Default to UP
        
        return first_move, score
    
    def pick_move(self, board_list):
        """Use Monte Carlo simulation to pick the best move."""
        board = np.array(board_list, dtype=np.int32)
        
        scores = [0, 0, 0, 0]
        counter = [0, 0, 0, 0]
        
        # Run simulations for each move
        for i in range(self.runs_per_move):
            first_move, final_score = self.simulate_one_run(board)
            scores[first_move] += final_score
            counter[first_move] += 1
        
        best_avg_score = float('-inf')
        best_move = 0
        
        for i in range(4):
            if counter[i] > 0:
                avg_score = scores[i] / counter[i]
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_move = i
        
        return self.direction_names[best_move]

def monte_carlo_search(game_instance, runs_per_move=50):
    """Wrapper function for Monte Carlo player"""
    mc_player = MonteCarloPlayer(runs_per_move=runs_per_move)
    return mc_player.pick_move(game_instance.grid)


class GAME2048:
    def __init__(self):
        self.grid = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        self.score = 0
        self.moves = 0
        self.font = TILE_FONT
        self.moves_font = MOVES_FONT
        self.timer_font = TIMER_FONT
        self.game_over = False
        self.start_time = pygame.time.get_ticks()
        self.game_end_time = 0
        self.max_tile_achieved = 0
    
    def update_max_tile(self):
        """Track the maximum tile achieved"""
        current_max = max(max(row) for row in self.grid)
        if current_max > self.max_tile_achieved:
            self.max_tile_achieved = current_max
            
            max_pos = None
            for r in range(4):
                for c in range(4):
                    if self.grid[r][c] == current_max:
                        max_pos = (r, c)
                        break
                if max_pos:
                    break
            
            empty = sum(1 for row in self.grid for cell in row if cell == 0)
            
            corner_status = "✓ CORNER" if max_pos in [(0,0), (0,3), (3,0), (3,3)] else "⚠ NOT IN CORNER"
            
            if current_max >= 128:
                print(f"🎯 New max: {current_max} at {max_pos} [{corner_status}] | Empty: {empty} | Score: {self.score} | Moves: {self.moves}")
            
            if current_max >= 2048:
                print(f"🎉🎉🎉 ACHIEVED {current_max} TILE! 🎉🎉🎉")
        
        return current_max
        
    def add_random_tile(self):
        empty_tiles = [(r, c) for r in range(4) for c in range(4) if self.grid[r][c] == 0]
        if empty_tiles:
            row, col = random.choice(empty_tiles)
            self.grid[row][col] = 2 if random.random() < 0.9 else 4
    
    def move(self, direction):
        """Optimized move using numpy backend"""
        board = np.array(self.grid, dtype=np.int32)
        direction_map = {"up": 0, "right": 1, "down": 2, "left": 3}
        dir_index = direction_map[direction]
        
        new_board, score_delta = make_move_with_score(board, dir_index)
        
        if boards_equal(board, new_board):
            return False
        
        self.score += score_delta
        
        self.grid = new_board.tolist()
        self.add_random_tile()
        self.moves += 1
        self.update_max_tile()
        
        return True
    
    def is_game_over(self):
        board = np.array(self.grid, dtype=np.int32)
        return is_game_over_numba(board)
    
    def get_elapsed_time(self):
        if self.game_over:
            return self.game_end_time
        return (pygame.time.get_ticks() - self.start_time) / 1000.0
        
    def record_game_end_time(self):
        self.game_end_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
            
    def draw(self, screen):
        screen.fill((187, 173, 160))
        pygame.draw.rect(screen, (187, 173, 160), (0, 0, WINDOW_SIZE, HEADER_HEIGHT))
        
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        moves_text = self.moves_font.render(f"Moves: {self.moves}", True, (255, 255, 255))
        elapsed_time = self.get_elapsed_time()
        time_text = self.timer_font.render(f"Time: {elapsed_time:.1f}s", True, (255, 255, 255))
        
        screen.blit(score_text, (20, 20))
        screen.blit(time_text, (20, 55))
        moves_rect = moves_text.get_rect(topright=(WINDOW_SIZE - 20, 20))
        screen.blit(moves_text, moves_rect)
        
        for row in range(4):
            for col in range(4):
                x = col * TILE_SIZE + (col + 1) * GAP
                y = row * TILE_SIZE + (row + 1) * GAP + HEADER_HEIGHT
                value = self.grid[row][col]
                color = TILE_COLORS.get(value, TILE_COLORS[2048])
                pygame.draw.rect(screen, color, (x, y, TILE_SIZE, TILE_SIZE), 0, 5)
                
                if value != 0:
                    text_color = (119, 110, 101) if value <= 4 else (249, 246, 242)
                    text_surface = self.font.render(str(value), True, text_color)
                    text_rect = text_surface.get_rect(center=(x + TILE_SIZE//2, y + TILE_SIZE//2))
                    screen.blit(text_surface, text_rect)
                        
        if self.game_over:
            overlay = pygame.Surface((WINDOW_SIZE, T_WIN_SIZE))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            game_over_text = OVER_FONT.render("GAME OVER", True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(WINDOW_SIZE//2, T_WIN_SIZE//2 - 40))
            screen.blit(game_over_text, text_rect)
            
            max_tile_text = self.font.render(f"Max Tile: {self.max_tile_achieved}", True, (255, 255, 255))
            max_rect = max_tile_text.get_rect(center=(WINDOW_SIZE//2, T_WIN_SIZE//2 + 10))
            screen.blit(max_tile_text, max_rect)
            
            final_score = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            score_rect = final_score.get_rect(center=(WINDOW_SIZE//2, T_WIN_SIZE//2 + 50))
            screen.blit(final_score, score_rect)
            
            restart_text = TIMER_FONT.render("Press R to Restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(WINDOW_SIZE//2, T_WIN_SIZE//2 + 90))
            screen.blit(restart_text, restart_rect)
                    
    def reset_game(self):
        self.grid = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.max_tile_achieved = 0
        self.start_time = pygame.time.get_ticks()
        self.add_random_tile()
        self.add_random_tile()
        print("\n🎮 New game started!")

def run_experiment(num_games=100):
    # Create experiments directory if it doesn't exist
    os.makedirs("benchmarks", exist_ok=True)
    
    # Prepare CSV file path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmarks/monte_carlo_results_{timestamp}.csv"
    
    # Header for the CSV file
    headers = ['Game_Number', 'Highest_Tile', 'Final_Score', 'Time_Seconds', 'Number_of_Moves', 'Success']
    
    # List to store results
    results = []
    
    print(f"Starting {num_games} games...")
    
    completed_games = 0
    
    while completed_games < num_games:
        game_num = completed_games + 1
        print(f"Running game {game_num}/{num_games}")
        
        game = GAME2048()
        game.add_random_tile()
        game.add_random_tile()
        
        clock = pygame.time.Clock()
        running = True
        
        ai_mode = True
        ai_delay = 0.05  # Faster for experiment efficiency
        last_ai_move_time = 0
        
        # Main game loop for this specific game
        while running and not game.game_over:
            dt = clock.tick(120) / 1000.0  # Higher FPS for faster execution
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return  # Exit the entire experiment
        
            if ai_mode and not game.game_over:
                current_time = pygame.time.get_ticks() / 1000.0
                if current_time - last_ai_move_time >= ai_delay:
                    best_action = monte_carlo_search(game, runs_per_move=25)  # Reduced runs for speed
                    moved = game.move(best_action)
                    
                    if moved:
                        game.game_over = game.is_game_over()
                        if game.game_over:
                            game.record_game_end_time()
                            
                    last_ai_move_time = current_time
            
            game.draw(screen)
            pygame.display.flip()
        
        # Calculate game duration using the internal timer
        game_time = game.get_elapsed_time()
        
        # Record game statistics
        highest_tile = game.max_tile_achieved
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
    
    print(f"\nExperiment completed! Results saved to {csv_filename}")
    
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
    # run()
    run_experiment(10)