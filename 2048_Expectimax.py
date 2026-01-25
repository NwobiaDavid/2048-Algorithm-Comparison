import pygame
import random
import copy
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
}

def get_empty_cells(grid):
    empty_cells = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == 0:
                empty_cells.append((i, j))
    return empty_cells

def evaluate_board(grid):
    score = 0
    
    monotonicity_weight = 10
    monotonicity_score = 0
    
    for row in grid:
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                monotonicity_score += 1
    
    for col in range(GRID_SIZE):
        for row in range(GRID_SIZE - 1):
            if grid[row][col] >= grid[row + 1][col]:
                monotonicity_score += 1
                
    score += monotonicity_score * monotonicity_weight
    
    empty_count = len(get_empty_cells(grid))
    score += empty_count * 100
    
    max_title = 0
    for row in grid:
        if max(row) > max_title:
            max_title = max(row)
    
    corner_positions = [(0,0), (0, GRID_SIZE-1), (GRID_SIZE-1, 0), (GRID_SIZE-1, GRID_SIZE-1)]
    
    corners = [grid[r][c] for r, c in corner_positions]
    if max_title in corners:
        bonus = max_title * 5
        score += bonus
        
    smoothness_penalty = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] != 0:
                if j < GRID_SIZE - 1 and grid[i][j+1] != 0:
                    smoothness_penalty += abs(grid[i][j] - grid[i][j+1])
                if i < GRID_SIZE - 1 and grid[i+1][j] != 0:
                    smoothness_penalty += abs(grid[i][j] - grid[i+1][j])
    
    score -= smoothness_penalty * 2
    
    return score
    
    
def expectimax(grid, depth, is_max_node):
    
    if depth == 0:
        return evaluate_board(grid), None
    
    if is_max_node:
        best_score = float('-inf')
        best_move = None
            
        for direction in ["up", "down", "left", "right"]:
            grid_copy = copy.deepcopy(grid)
            
            game_temp = GAME2048()
            game_temp.grid = grid_copy
            
            moved = False
            if direction == "left":
                moved = game_temp.move_left()
            elif direction == "right":
                moved = game_temp.move_right()
            elif direction == "up":
                moved = game_temp.move_up()
            elif direction == "down":
                moved = game_temp.move_down()
            
            if moved:
                score, _ = expectimax(game_temp.grid, depth - 1, False)
                    
                if score > best_score:
                    best_score = score
                    best_move = direction
                        
        return best_score, best_move
    else:
        empty_cells = get_empty_cells(grid)
        if not empty_cells:
            return evaluate_board(grid), None
    
        total_score = 0
    
        for row, col in empty_cells:
            grid_2 = copy.deepcopy(grid)
            grid_2[row][col] = 2
            score_2, _ = expectimax(grid_2, depth - 1, True)
            total_score += 0.9 * score_2 / len(empty_cells)
            
            grid_4 = copy.deepcopy(grid)
            grid_4[row][col] = 4
            score_4, _ = expectimax(grid_4, depth - 1, True)
            total_score += 0.1 * score_4 / len(empty_cells)
            
        return total_score, None
    

def get_ai_move(current_grid, depth=4):
    _, best_move = expectimax(current_grid, depth, True)
    return best_move
    
    
    
    
class GAME2048:
    def __init__(self):
        self.grid = [[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
        
        self.score = 0
        self.moves = 0
        self.font = TILE_FONT
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
        new_row = []
        for num in row:
            if num != 0:
                new_row.append(num)
                
        while len(new_row) < len(row):
            new_row.append(0)
            
        return new_row
    
    def merge_row(self, row):
        merged = []
        skip_next = False
        
        for i in range(len(row)):
            if skip_next:
                skip_next = False
                continue
            
            if i < len(row) - 1 and row[i] == row[i + 1]:
                merged_value = row[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                skip_next = True
            else:
                merged.append(row[i])
                
        while len(merged) < len(row):
            merged.append(0)
            
        return merged
    
    def transpose_grid(self):
        transposed = []
        for _ in range(GRID_SIZE):
            new_row = []
            
            for _ in range(GRID_SIZE):
                new_row.append(0)
            
            transposed.append(new_row)
            
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                transposed[j][i] = self.grid[i][j]
        return transposed          
    
    
    def move_left(self):
        original_grid = []
        for row in self.grid:
            row_copy = row[:]
            original_grid.append(row_copy)
            
        for i in range(GRID_SIZE):
            compressed_row = self.compress_row(self.grid[i])
            merged_row = self.merge_row(compressed_row)
            
            self.grid[i] = merged_row
            
        return original_grid != self.grid
    
    def move_right(self):
        original_grid = []
        for row in self.grid:
            row_copy = row[:]
            original_grid.append(row_copy)
            
        for i in range(GRID_SIZE):
            reversed_row = self.grid[i][::-1]
            
            compressed_row = self.compress_row(reversed_row)
            merged_row = self.merge_row(compressed_row)
            
            self.grid[i] = merged_row[::-1]
        
        return original_grid != self.grid
    
    def move_up(self):
        original_grid = []
        for row in self.grid:
            row_copy = row[:]
            original_grid.append(row_copy)
            
        self.grid = self.transpose_grid()
        
        for i in range(GRID_SIZE):
            compressed_row = self.compress_row(self.grid[i])
            merged_row = self.merge_row(compressed_row)
            
            self.grid[i] = merged_row
            
        self.grid = self.transpose_grid()
        
        return original_grid != self.grid
    
    def move_down(self):
        original_grid = []
        for row in self.grid:
            row_copy = row[:]
            original_grid.append(row_copy)
            
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
            return self.paused_time
        
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
        moves_text = self.font.render(f"Moves: {self.moves}", True, (255, 255, 255))
        
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
                        
                        color = TILE_COLORS.get(value, TILE_COLORS[2048])
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
            text_rect = game_over_text.get_rect(center=(WINDOW_SIZE//2, T_WIN_SIZE//2))
            screen.blit(game_over_text, text_rect)
            
            restart_text = self.font.render("Press R to Restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(WINDOW_SIZE//2, T_WIN_SIZE//2 + 50))
            screen.blit(restart_text, restart_rect)
                    
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
        
def run():
    game = GAME2048()
    game.add_random_tile()
    game.add_random_tile()
    
    clock = pygame.time.Clock()
    running = True
    
    ai_playing = True
    ai_move_delay = 0
    ai_move_interval = 100
    
    while running:
        dt = clock.tick(60) / 1000.0
        ai_move_delay += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game.reset_game()
                    elif event.key == pygame.K_SPACE:
                        ai_playing = not ai_playing
                        print("AI Mode:", "ON" if ai_playing else "OFF")
                        if ai_playing:
                            game.resume_timer()
                        else:
                            game.pause_timer()
                        
        if ai_playing and not game.moving_animation and not game.game_over and ai_move_delay >= ai_move_interval:
            ai_move_delay = 0
            ai_move = get_ai_move(game.grid, depth=4)
            if ai_move:
                game.move(ai_move)
                game.game_over = game.is_game_over()
                if game.game_over:
                    game.record_game_end_time()
                
                                        
        if not game.game_over:
            game.update_animation(dt)
        
        game.draw(screen)   
        pygame.display.flip()
        
    pygame.quit()
    

if __name__ == "__main__":
    run()
