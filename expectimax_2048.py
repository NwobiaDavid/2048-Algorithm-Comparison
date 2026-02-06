import pygame
import random
import copy
import time
import os
import csv
from collections import defaultdict
pygame.font.init()

GRID_SIZE = 4
TILE_SIZE = 100
GAP = 10
HEADER_HEIGHT = 80
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1)*GAP

T_WIN_SIZE = WINDOW_SIZE + HEADER_HEIGHT

pygame.init()
pygame.display.set_caption("Expectimax 2048")
TILE_FONT = pygame.font.SysFont("comicsans", 32, bold=True)
OVER_FONT = pygame.font.SysFont("comicsans", 48, bold=True)
MOVES_FONT = pygame.font.SysFont("comicsans", 20, bold=True) 
TIMER_FONT = pygame.font.SysFont("comicsans", 20, bold=True)

screen = pygame.display.set_mode((WINDOW_SIZE, T_WIN_SIZE))

TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50)
}

def get_empty_cells(grid):
    empty_cells = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == 0:
                empty_cells.append((i, j))
    return empty_cells

def tile_exp(grid, r, c):
    val = grid[r][c]
    if val == 0:
        return 0
    exp = 0
    temp_val = val
    while temp_val > 1:
        temp_val >>= 1
        exp += 1
    return exp

def tile_val(grid, r, c):
    return grid[r][c]

def transpose_grid(grid):
    return [[grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]

def flip_h(grid):
    return [row[::-1] for row in grid]

def flip_v(grid):
    return grid[::-1]

def compress_row(row):
    new_row = [num for num in row if num != 0]
    new_row.extend([0] * (len(row) - len(new_row)))
    return new_row

def merge_row(row):
    merged = []
    skip_next = False
    
    for i in range(len(row)):
        if skip_next:
            skip_next = False
            merged.append(0)
            continue
        
        if i < len(row) - 1 and row[i] == row[i + 1] and row[i] != 0:
            merged_value = row[i] * 2
            merged.append(merged_value)
            skip_next = True
        else:
            merged.append(row[i])
            
    merged = [num for num in merged if num != 0]
    merged.extend([0] * (len(row) - len(merged)))
    return merged

def apply_move(grid, direction):
    grid_copy = [row[:] for row in grid]
    original_grid = [row[:] for row in grid_copy]
    
    if direction == "left":
        for i in range(GRID_SIZE):
            compressed_row = compress_row(grid_copy[i])
            merged_row = merge_row(compressed_row)
            grid_copy[i] = merged_row
    elif direction == "right":
        for i in range(GRID_SIZE):
            reversed_row = grid_copy[i][::-1]
            compressed_row = compress_row(reversed_row)
            merged_row = merge_row(compressed_row)
            grid_copy[i] = merged_row[::-1]
    elif direction == "up":
        grid_copy = transpose_grid(grid_copy)
        for i in range(GRID_SIZE):
            compressed_row = compress_row(grid_copy[i])
            merged_row = merge_row(compressed_row)
            grid_copy[i] = merged_row
        grid_copy = transpose_grid(grid_copy)
    elif direction == "down":
        grid_copy = transpose_grid(grid_copy)
        for i in range(GRID_SIZE):
            reversed_row = grid_copy[i][::-1]
            compressed_row = compress_row(reversed_row)
            merged_row = merge_row(compressed_row)
            grid_copy[i] = merged_row[::-1]
        grid_copy = transpose_grid(grid_copy)
    
    return grid_copy, original_grid != grid_copy

def count_empty(grid):
    count = 0
    for row in grid:
        for cell in row:
            if cell == 0:
                count += 1
    return count

def count_distinct_tiles(grid):
    distinct = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                distinct.add(cell)
    return len(distinct)

def get_max_tile(grid):
    max_val = 0
    for row in grid:
        for cell in row:
            if cell > max_val:
                max_val = cell
    return max_val

def score_heuristic(grid):
    score = 0
    for row in grid:
        for val in row:
            if val >= 4:  
                k = 0
                temp_val = val
                while temp_val > 1:
                    temp_val //= 2
                    k += 1
                score += (k - 1) * val
    return score

def merge_heuristic(grid):
    return count_empty(grid)

def corner_heuristic(grid):
    bonuses = []
    
    
    lower_left = (10 * tile_val(grid, 0, 3) + 5 * tile_val(grid, 0, 2) + 2 * tile_val(grid, 0, 1) + 1 * tile_val(grid, 0, 0) +
                  5  * tile_val(grid, 1, 3) + 3 * tile_val(grid, 1, 2) + 1 * tile_val(grid, 1, 1) +
                  2  * tile_val(grid, 2, 3) + 1 * tile_val(grid, 2, 2) +
                  1  * tile_val(grid, 3, 3))
    bonuses.append(lower_left)
    
    upper_left = (10 * tile_val(grid, 3, 3) + 5 * tile_val(grid, 3, 2) + 2 * tile_val(grid, 3, 1) + 1 * tile_val(grid, 3, 0) +
                  5  * tile_val(grid, 2, 3) + 3 * tile_val(grid, 2, 2) + 1 * tile_val(grid, 2, 1) +
                  2  * tile_val(grid, 1, 3) + 1 * tile_val(grid, 1, 2) +
                  1  * tile_val(grid, 0, 3))
    bonuses.append(upper_left)
    
    lower_right = (10 * tile_val(grid, 0, 0) + 5 * tile_val(grid, 0, 1) + 2 * tile_val(grid, 0, 2) + 1 * tile_val(grid, 0, 3) +
                   5  * tile_val(grid, 1, 0) + 3 * tile_val(grid, 1, 1) + 1 * tile_val(grid, 1, 2) +
                   2  * tile_val(grid, 2, 0) + 1 * tile_val(grid, 2, 1) +
                   1  * tile_val(grid, 3, 0))
    bonuses.append(lower_right)
    
    upper_right = (10 * tile_val(grid, 3, 0) + 5 * tile_val(grid, 3, 1) + 2 * tile_val(grid, 3, 2) + 1 * tile_val(grid, 3, 3) +
                   5  * tile_val(grid, 2, 0) + 3 * tile_val(grid, 2, 1) + 1 * tile_val(grid, 2, 2) +
                   2  * tile_val(grid, 1, 0) + 1 * tile_val(grid, 1, 1) +
                   1  * tile_val(grid, 0, 0))
    bonuses.append(upper_right)
    
    return max(bonuses)

def wall_gap_heuristic(grid):
    def _wall_gap_helper(g):
        top = ((tile_exp(g, 3, 3) << 40) + (tile_exp(g, 3, 2) << 36) + (tile_exp(g, 3, 1) << 32) +
               (tile_exp(g, 2, 3) << 20) + (tile_exp(g, 2, 2) << 24) + (tile_exp(g, 2, 1) << 28) +
               (tile_exp(g, 1, 3) << 16) + (tile_exp(g, 1, 2) << 12) + (tile_exp(g, 1, 1) << 8))
        
        bottom = ((tile_exp(g, 0, 0) << 40) + (tile_exp(g, 0, 1) << 36) + (tile_exp(g, 0, 2) << 32) +
                  (tile_exp(g, 1, 0) << 20) + (tile_exp(g, 1, 1) << 24) + (tile_exp(g, 1, 2) << 28) +
                  (tile_exp(g, 2, 0) << 16) + (tile_exp(g, 2, 1) << 12) + (tile_exp(g, 2, 2) << 8))
        
        left = ((tile_exp(g, 0, 3) << 40) + (tile_exp(g, 1, 3) << 36) + (tile_exp(g, 2, 3) << 32) +
                (tile_exp(g, 0, 2) << 20) + (tile_exp(g, 1, 2) << 24) + (tile_exp(g, 2, 2) << 28) +
                (tile_exp(g, 0, 1) << 16) + (tile_exp(g, 1, 1) << 12) + (tile_exp(g, 2, 1) << 8))
        
        right = ((tile_exp(g, 3, 0) << 40) + (tile_exp(g, 2, 0) << 36) + (tile_exp(g, 1, 0) << 32) +
                 (tile_exp(g, 3, 1) << 20) + (tile_exp(g, 2, 1) << 24) + (tile_exp(g, 1, 1) << 28) +
                 (tile_exp(g, 3, 2) << 16) + (tile_exp(g, 2, 2) << 12) + (tile_exp(g, 1, 2) << 8))
        
        return max(top, bottom, left, right)
    
    return max(_wall_gap_helper(grid), _wall_gap_helper(transpose_grid(grid))) + score_heuristic(grid)

def full_wall_heuristic(grid):
    def _full_wall_helper(g):
        top = ((tile_exp(g, 3, 3) << 40) + (tile_exp(g, 3, 2) << 36) + (tile_exp(g, 3, 1) << 32) + (tile_exp(g, 3, 0) << 28) +
               (tile_exp(g, 2, 3) << 12) + (tile_exp(g, 2, 2) << 16) + (tile_exp(g, 2, 1) << 20) + (tile_exp(g, 2, 0) << 24) +
               (tile_exp(g, 1, 3) << 8))
        
        bottom = ((tile_exp(g, 0, 0) << 40) + (tile_exp(g, 0, 1) << 36) + (tile_exp(g, 0, 2) << 32) + (tile_exp(g, 0, 3) << 28) +
                  (tile_exp(g, 1, 0) << 12) + (tile_exp(g, 1, 1) << 16) + (tile_exp(g, 1, 2) << 20) + (tile_exp(g, 0, 3) << 24) +
                  (tile_exp(g, 2, 0) << 8))
        
        left = ((tile_exp(g, 0, 3) << 40) + (tile_exp(g, 1, 3) << 36) + (tile_exp(g, 2, 3) << 32) + (tile_exp(g, 3, 3) << 28) +
                (tile_exp(g, 0, 2) << 12) + (tile_exp(g, 1, 2) << 16) + (tile_exp(g, 2, 2) << 20) + (tile_exp(g, 3, 2) << 24) +
                (tile_exp(g, 0, 1) << 8))
        
        right = ((tile_exp(g, 3, 0) << 40) + (tile_exp(g, 2, 0) << 36) + (tile_exp(g, 1, 0) << 32) + (tile_exp(g, 0, 0) << 28) +
                 (tile_exp(g, 3, 1) << 12) + (tile_exp(g, 2, 1) << 16) + (tile_exp(g, 1, 1) << 20) + (tile_exp(g, 0, 1) << 24) +
                 (tile_exp(g, 3, 2) << 8))
        
        return max(top, bottom, left, right)
    
    return max(_full_wall_helper(grid), _full_wall_helper(transpose_grid(grid))) + score_heuristic(grid)


def skewed_corner_heuristic(grid):
    def _skewed_corner_helper(g):
        top = (16 * tile_val(g, 3, 3) + 10 * tile_val(g, 3, 2) + 6 * tile_val(g, 3, 1) + 3 * tile_val(g, 3, 0) +
               10 * tile_val(g, 2, 3) + 6  * tile_val(g, 2, 2) + 3 * tile_val(g, 2, 1) + 1 * tile_val(g, 2, 0) +
               4  * tile_val(g, 1, 3) + 3  * tile_val(g, 1, 2) + 1 * tile_val(g, 1, 1) +
               1  * tile_val(g, 0, 3) + 1  * tile_val(g, 0, 2))
        
        bottom = (16 * tile_val(g, 0, 0) + 10 * tile_val(g, 0, 1) + 6 * tile_val(g, 0, 2) + 3 * tile_val(g, 0, 3) +
                  10 * tile_val(g, 1, 0) + 6  * tile_val(g, 1, 1) + 3 * tile_val(g, 1, 2) + 1 * tile_val(g, 1, 3) +
                  4  * tile_val(g, 2, 0) + 3  * tile_val(g, 2, 1) + 1 * tile_val(g, 2, 2) +
                  1  * tile_val(g, 3, 0) + 1  * tile_val(g, 3, 1))
        
        left = (16 * tile_val(g, 0, 3) + 10 * tile_val(g, 1, 3) + 6 * tile_val(g, 2, 3) + 3 * tile_val(g, 3, 3) +
                10 * tile_val(g, 0, 2) + 6  * tile_val(g, 1, 2) + 3 * tile_val(g, 2, 2) + 1 * tile_val(g, 3, 2) +
                4  * tile_val(g, 0, 1) + 3  * tile_val(g, 1, 1) + 1 * tile_val(g, 2, 1) +
                1  * tile_val(g, 0, 0) + 1  * tile_val(g, 1, 0))
        
        right = (16 * tile_val(g, 3, 0) + 10 * tile_val(g, 2, 0) + 6 * tile_val(g, 1, 0) + 3 * tile_val(g, 0, 0) +
                 10 * tile_val(g, 3, 1) + 6  * tile_val(g, 2, 1) + 3 * tile_val(g, 1, 1) + 1 * tile_val(g, 0, 1) +
                 4  * tile_val(g, 3, 2) + 3  * tile_val(g, 2, 2) + 1 * tile_val(g, 1, 2) +
                 1  * tile_val(g, 3, 3) + 1  * tile_val(g, 2, 3))
        
        return max(top, bottom, left, right)
    
    return max(_skewed_corner_helper(grid), _skewed_corner_helper(transpose_grid(grid)))


def monotonicity_heuristic(grid):
    
    def calc_row_monotonicity(row_vals):
        score = 0
        
        total_val = sum(max(val, 0) for val in row_vals)
        
        increasing = decreasing = 0
        for i in range(len(row_vals) - 1):
            val1 = row_vals[i]
            val2 = row_vals[i + 1]
            
            if val1 >= val2 and val1 != 0:
                increasing += 1
            if val1 <= val2 and val2 != 0:
                decreasing += 1
        
        monotonic_score = max(increasing, decreasing)
        return monotonic_score * total_val
    
    total_score = 0
    
    for i in range(4):
        row = grid[i]
        total_score += calc_row_monotonicity(row)
    
    transposed = transpose_grid(grid)
    for i in range(4):
        col = transposed[i]
        total_score += calc_row_monotonicity(col)
    
    max_monotonicity = 0
    for i in range(4):
        row = grid[i]
        monotonicity = calc_row_monotonicity(row)
        max_monotonicity = max(max_monotonicity, monotonicity)
    
    transposed = transpose_grid(grid)
    for i in range(4):
        col = transposed[i]
        monotonicity = calc_row_monotonicity(col)
        max_monotonicity = max(max_monotonicity, monotonicity)
    
    total_score += max_monotonicity * 8
    
    total_score += count_empty(grid)
    
    return max(0, total_score)

def evaluate_board(grid, heuristic_idx=2):
    
    heuristics = [
        score_heuristic,
        merge_heuristic,
        corner_heuristic,
        wall_gap_heuristic,
        full_wall_heuristic,
        lambda g: full_wall_heuristic(g),
        skewed_corner_heuristic,
        monotonicity_heuristic
    ]
    
    return heuristics[heuristic_idx](grid)

def expectimax(grid, depth, is_max_node, cache=None, depth_limit=4, fours_count=0, cache_depth=2):

    if cache is None:
        cache = {}
    
    grid_tuple = tuple(map(tuple, grid))
    cache_key = (grid_tuple, depth, is_max_node)
    
    
    if depth >= cache_depth and cache_key in cache:
        cached_result, cached_depth = cache[cache_key]
        if cached_depth >= depth:
            return cached_result, None
    
    
    if is_game_over_static(grid):
        
        raw_score = evaluate_board(grid)
        score = raw_score - (raw_score >> 2)
        result = score, None
        if depth >= cache_depth:
            cache[cache_key] = (result[0], depth)
        return result
    
    if depth == 0 or fours_count >= 4:
        score = evaluate_board(grid)
        result = score, None
        if depth >= cache_depth:
            cache[cache_key] = (result[0], depth)
        return result
    
    if is_max_node:
        
        best_score = float('-inf')
        best_move = None
        
        for idx, direction in enumerate(["up", "down", "left", "right"]):
            new_grid, moved = apply_move(grid, direction)
            
            if moved:
                score, _ = expectimax(new_grid, depth - 1, False, cache, depth_limit, fours_count, cache_depth)
                
                if score >= best_score:
                    best_score = score
                    best_move = idx  
        
        result = best_score, best_move
        if depth >= cache_depth:
            cache[cache_key] = (result[0], depth)
        return result
    else:
        empty_cells = get_empty_cells(grid)
        if not empty_cells:
            score = evaluate_board(grid)
            result = score, None
            if depth >= cache_depth:
                cache[cache_key] = (result[0], depth)
            return result
        
        total_score = 0
        num_empty = len(empty_cells)
        
        for row, col in empty_cells:
            grid_2 = [row[:] for row in grid]
            grid_2[row][col] = 2
            score_2, _ = expectimax(grid_2, depth - 1, True, cache, depth_limit, fours_count, cache_depth)
            total_score += 9 * score_2  
            
            
            grid_4 = [row[:] for row in grid]
            grid_4[row][col] = 4
            score_4, _ = expectimax(grid_4, depth - 1, True, cache, depth_limit, fours_count + 1, cache_depth)
            total_score += 1 * score_4  
            
            
        expected_score = total_score / (num_empty * 10)
        result = expected_score, None
        if depth >= cache_depth:
            cache[cache_key] = (result[0], depth)
        return result

def is_game_over_static(grid):
    for row in grid:
        if 0 in row:
            return False
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE - 1):
            if grid[row][col] == grid[row][col + 1]:
                return False
                
    for row in range(GRID_SIZE - 1):
        for col in range(GRID_SIZE):
            if grid[row][col] == grid[row + 1][col]:
                return False
    return True

def dynamic_depth_picker(grid):
    tile_ct = GRID_SIZE * GRID_SIZE - count_empty(grid)
    
    score = count_distinct_tiles(grid) + (0 if tile_ct <= 6 else (tile_ct - 6) >> 1)
    
    depth = 2
    if score >= 8: depth += 1
    if score >= 11: depth += 1
    if score >= 14: depth += 1
    if score >= 15: depth += 1
    if score >= 17: depth += 1
    if score >= 19: depth += 1
    
    return depth

def get_ai_move(current_grid, depth=-1, heuristic_idx=2): 
    
    depth_to_use = depth if depth > 0 else dynamic_depth_picker(current_grid) - depth if depth < 0 else dynamic_depth_picker(current_grid)
    
    _, move_idx = expectimax(current_grid, depth_to_use, True, cache={}, depth_limit=depth_to_use)
    
    directions = ["up", "down", "left", "right"]
    return directions[move_idx] if move_idx is not None else None

class GAME2048:
    def __init__(self):
        self.grid = [[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
        
        self.score = 0
        self.moves = 0
        self.font = TILE_FONT
        self.moves_font = MOVES_FONT
        self.timer_font = TIMER_FONT
        self.game_over = False
        self.moving_animation = False
        self.animation_progress = 0
        self.animation_direction = None
        self.start_position = {}
        self.end_position = {}
        
        self.start_time = pygame.time.get_ticks()
        self.pause_time = 0
        self.total_paused_time = 0
        self.is_timer_running = True
        
    def add_random_tile(self):
        empty_tiles = []
        
        for row_idx in range(len(self.grid)):
            for col_idx in range(len(self.grid[row_idx])):
                if self.grid[row_idx][col_idx] == 0:
                    empty_tiles.append((row_idx, col_idx))
                    
        if empty_tiles:
            rand_position = random.choice(empty_tiles)
            row, col = rand_position
            
            if random.random() < 0.9:
                self.grid[row][col] = 2
            else:
                self.grid[row][col] = 4
    
    def compress_row(self, row):
        new_row = [num for num in row if num != 0]
        new_row.extend([0] * (len(row) - len(new_row)))
        return new_row
    
    def merge_row(self, row):
        merged = []
        skip_next = False
        
        for i in range(len(row)):
            if skip_next:
                skip_next = False
                merged.append(0)
                continue
            
            if i < len(row) - 1 and row[i] == row[i + 1] and row[i] != 0:
                merged_value = row[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                skip_next = True
            else:
                merged.append(row[i])
                
        merged = [num for num in merged if num != 0]
        merged.extend([0] * (len(row) - len(merged)))
        return merged
    
    def transpose_grid(self):
        return [[self.grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]  
    
    def move_left(self):
        original_grid = [row[:] for row in self.grid]
            
        for i in range(GRID_SIZE):
            compressed_row = self.compress_row(self.grid[i])
            merged_row = self.merge_row(compressed_row)
            self.grid[i] = merged_row
            
        return original_grid != self.grid
    
    def move_right(self):
        original_grid = [row[:] for row in self.grid]
            
        for i in range(GRID_SIZE):
            reversed_row = self.grid[i][::-1]
            compressed_row = self.compress_row(reversed_row)
            merged_row = self.merge_row(compressed_row)
            self.grid[i] = merged_row[::-1]
        
        return original_grid != self.grid
    
    def move_up(self):
        original_grid = [row[:] for row in self.grid]
        self.grid = self.transpose_grid()
        
        for i in range(GRID_SIZE):
            compressed_row = self.compress_row(self.grid[i])
            merged_row = self.merge_row(compressed_row)
            self.grid[i] = merged_row
            
        self.grid = self.transpose_grid()
        
        return original_grid != self.grid
    
    def move_down(self):
        original_grid = [row[:] for row in self.grid]
        self.grid = self.transpose_grid()
        
        for i in range(GRID_SIZE):
            reversed_row = self.grid[i][::-1]
            compressed_row = self.compress_row(reversed_row)
            merged_row = self.merge_row(compressed_row)
            self.grid[i] = merged_row[::-1]
            
        self.grid = self.transpose_grid()
        
        return original_grid != self.grid
      
    def move(self, direction):
        moved = False
        if direction == "left":
            moved = self.move_left()
        elif direction == "right":
            moved = self.move_right()
        elif direction == "up":
            moved = self.move_up()
        elif direction == "down":
            moved = self.move_down()
            
        if moved:
            self.add_random_tile()
            self.moves += 1
        
        return moved
    
    def is_game_over(self):
        for row in self.grid:
            if 0 in row:
                return False
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE - 1):
                if self.grid[row][col] == self.grid[row][col + 1]:
                    return False
                
        for row in range(GRID_SIZE - 1):
            for col in range(GRID_SIZE):
                if self.grid[row][col] == self.grid[row + 1][col]:
                    return False
        return True
    
    def get_elapsed_time(self):
        if self.game_over:
            return self.game_end_time
        elif self.is_timer_running:
            current_time = pygame.time.get_ticks()
            elapsed = (current_time - self.start_time - self.total_paused_time) / 1000.0
            return max(elapsed, 0)
        else:
            return self.pause_time
        
    def pause_timer(self):
        if self.is_timer_running:
            self.current_time = pygame.time.get_ticks()
            self.elapsed_before_pause = (self.current_time - self.start_time - self.total_paused_time) / 1000.0
            self.is_timer_running = False
            self.pause_time = self.elapsed_before_pause
            
    def resume_timer(self):
        if not self.is_timer_running:
            self.pause_start_time = pygame.time.get_ticks()
            self.is_timer_running = True
    
    def record_game_end_time(self):
        current_time = pygame.time.get_ticks()
        self.game_end_time = (current_time - self.start_time - self.total_paused_time) / 1000.0       
            
    def draw(self, screen):
        screen.fill((187, 173, 160))
        pygame.draw.rect(screen, (187, 173, 160), (0, 0, WINDOW_SIZE, HEADER_HEIGHT))
        
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        moves_text = self.moves_font.render(f"Moves: {self.moves}", True, (255, 255, 255))
        
        elasped_time = self.get_elapsed_time()
        time_text = self.timer_font.render(f"Time: {elasped_time:.1f}s", True, (255, 255, 255))
        
        screen.blit(score_text, (20, 20))
        screen.blit(time_text, (20, 55))
        
        moves_rect = moves_text.get_rect(topright=(WINDOW_SIZE - 20, 20))
        screen.blit(moves_text, moves_rect)
        
        if self.moving_animation and self.animation_direction:
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    base_x = col * TILE_SIZE + (col + 1) * GAP
                    base_y = row * TILE_SIZE + (row + 1) * GAP + HEADER_HEIGHT
                    
                    if (row, col) in self.end_position:
                        start_pos = self.start_position.get((row, col), (base_x, base_y))
                        end_pos = self.end_position[(row, col)]
                        
                        current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * self.animation_progress
                        current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * self.animation_progress
                    else:
                        current_x, current_y = base_x, base_y    
                        
                    value = self.grid[row][col]
                    if value != 0:
                        color = TILE_COLORS.get(value, TILE_COLORS[4096])
                        pygame.draw.rect(screen, color, (current_x, current_y, TILE_SIZE, TILE_SIZE), 0, 5)
                    
                        text_color = (119, 110, 101) if value <= 4 else (249, 246, 242)
                        text_surface = self.font.render(str(value), True, text_color)
                        
                        text_rect = text_surface.get_rect(center=(current_x + TILE_SIZE//2, current_y + TILE_SIZE//2))
                        screen.blit(text_surface, text_rect)
        else:
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    x = col * TILE_SIZE + (col + 1) * GAP
                    y = row * TILE_SIZE + (row + 1) * GAP + HEADER_HEIGHT
                    
                    value = self.grid[row][col]
                    color = TILE_COLORS.get(value, TILE_COLORS[4096])
                    
                    pygame.draw.rect(screen, color, (x, y, TILE_SIZE, TILE_SIZE), 0, 5)
                    
                    if value != 0:
                        text_color = (119, 110, 101) if value <= 4 else (249, 246, 242)
                        text_surface = self.font.render(str(value), True, text_color)
                        text_rect = text_surface.get_rect(center=(x + TILE_SIZE//2, y + TILE_SIZE//2))
                        screen.blit(text_surface, text_rect) 
                        
                    
    def animate_move(self, direction):
        self.moving_animation = True
        self.animation_progress = 0
        self.animation_direction = direction
        self.start_position = {}
        self.end_position = {}
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col] != 0:
                    base_x = col * TILE_SIZE + (col + 1) * GAP
                    base_y = row * TILE_SIZE + (row + 1) * GAP
                    self.start_position[(row, col)] = (base_x, base_y)
        
        temp_grid = []
        for row in self.grid:
            temp_grid.append(row[:])
            
        if direction == "left":
            for i in range(GRID_SIZE):
                row = temp_grid[i]
                compressed_row = self.compress_row(row)
                merged_row = self.merge_row(compressed_row)
                temp_grid[i] = merged_row
                
                original_non_zero = []
                for j, val in enumerate(row):
                    if val != 0:
                        original_non_zero.append((i, j))
                        
                final_non_zero = []
                for j, val in enumerate(temp_grid[i]):
                    if val != 0:
                        final_non_zero.append((i, j))
                        
                for idx, (final_pos) in enumerate(final_non_zero):
                    if idx < len(original_non_zero):
                        orig_pos = original_non_zero[idx]
                        final_x = final_pos[1] * TILE_SIZE + (final_pos[1] + 1) * GAP
                        final_y = final_pos[0] * TILE_SIZE + (final_pos[0] + 1) * GAP
                        self.end_position[orig_pos] = (final_x, final_y)
                        
    def update_animation(self, dt):
        if self.moving_animation:
            self.animation_progress += dt * 5
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                
                if self.animation_direction == "left":
                    self.move_left()
                elif self.animation_direction == "right":
                    self.move_right()
                elif self.animation_direction == "up":
                    self.move_up()
                elif self.animation_direction == "down":
                    self.move_down()
                    
                self.moving_animation = False
                self.add_random_tile()
    
    def reset_game(self):
        self.grid = [[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.start_time = pygame.time.get_ticks()
        self.total_paused_time = 0
        self.is_timer_running = True
        self.add_random_tile()
        self.add_random_tile()                   


def run_experiment(num_games=100):
    # Create experiments directory if it doesn't exist
    os.makedirs("benchmarks", exist_ok=True)
    
    # Prepare CSV file path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmarks/expectimax_results_{timestamp}.csv"
    
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
        
        ai_playing = True
        ai_move_delay = 0
        ai_move_interval = 1
        
        # Main game loop for this specific game
        while running and not game.game_over:
            dt = clock.tick(60) / 1000.0
            ai_move_delay += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return  # Exit the entire experiment
                    
            if ai_playing and not game.moving_animation and not game.game_over and ai_move_delay >= ai_move_interval:
                ai_move_delay = 0
                # AI move selection happens here
                ai_move = get_ai_move(game.grid, depth=-1, heuristic_idx=2)
                if ai_move:
                    game.move(ai_move)
                    game.game_over = game.is_game_over()
                    if game.game_over:
                        game.record_game_end_time()
                    
            if not game.game_over:
                game.update_animation(dt)
            
            game.draw(screen)   
            pygame.display.flip()
        
        # Calculate game duration using the internal timer
        game_time = game.get_elapsed_time()
        
        # Record game statistics
        highest_tile = get_max_tile(game.grid)  # Use function instead of method
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
    run_experiment(10)