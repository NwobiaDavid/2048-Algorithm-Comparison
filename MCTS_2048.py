import pygame
import random
import math
from copy import deepcopy

pygame.font.init()

GRID_SIZE = 4
TILE_SIZE = 100
GAP = 10
HEADER_HEIGHT = 80
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1)*GAP

T_WIN_SIZE = WINDOW_SIZE + HEADER_HEIGHT

pygame.init()
pygame.display.set_caption("MCTS 2048")
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

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = ["up", "dowm", "left", "right"]    
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=math.sqrt(2)):
        choices_weights = [
            (child.value / child.visits) + 
            exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self):
        action = self.untried_actions.pop(random.randint(0, len(self.untried_actions)-1))

        temp_game = GAME2048()
        temp_game.grid = [row[:] for row in [self.state[i:i+4] for i in range(0, 16, 4)]]
        
        original_grid = [row[:] for row in temp_game.grid]
        moved = temp_game.move(action)
        
        if moved or original_grid != temp_game.grid:
            new_State = temp_game.get_state()
            child_node = MCTSNode(state=new_State, parent=self, action=action)
            self.children.append(child_node)
            return child_node
        else:
            if len(self.untried_actions) > 0:
                return self.expand()
            return None
        
def state_is_terminal(game_instance):
    return game_instance.is_game_over()
    
def calculate_reward(game_instance):
    empty_tiles = sum(1 for row in game_instance.grid for cell in row if cell == 0)
    max_tile = max(max(row) for row in game_instance.grid)
    score = game_instance.score
        
    reward = score + (empty_tiles * 100) + (max_tile * 10)
        
    if game_instance.is_game_over():
        reward -= 1000
            
    return reward
    
def mcts_search(game_instance, iterations=100):
    root = MCTSNode(state=game_instance.get_state())
        
    for _ in range(iterations):
        node = root
        temp_game = GAME2048()
            
        while not state_is_terminal(temp_game) and node.is_fully_expanded():
            node = node.best_child()
            temp_game.grid = [row[:] for row in [node.state[i:i+4] for i in range(0, 16, 4)]]
                
        if not state_is_terminal(temp_game) and len(node.untried_actions) > 0:
            child = node.expand()
            if child is not None:
                node = child
                temp_game.grid = [row[:] for row in [node.state[i:i+4] for i in range(0, 16, 4)]]
                    
                    
        rollout_game = GAME2048()
        rollout_game.grid = [row[:] for row in [node.state[i:i+4] for i in range(0, 16, 4)]]
            
        simulation_steps = 0
        max_simulation_steps = 20
        while not state_is_terminal(rollout_game) and simulation_steps < max_simulation_steps:
            possible_moves = ["up", "down", "left", "right"] 
            random.shuffle(possible_moves)
            move_made = False
                
            for action in possible_moves:
                original_grid = [row[:] for row in rollout_game.grid]  
                moved = rollout_game.move(action)
                if moved or original_grid != rollout_game.grid:
                    move_made = True
                    break
                    
            if not move_made:
                break
            simulation_steps += 1
                
                
        reward = calculate_reward(rollout_game)
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value += reward
            current_node = current_node.parent
                
    if root.children:
        best_child = max(root.children, key=lambda c: c.visits) 
        return best_child.action
    else:
        possible_moves = ["up", "down", "left", "right"]
        random.shuffle(possible_moves)
        for action in possible_moves:
            temp_game = deepcopy(game_instance)
            original_grid = [row[:] for row in temp_game.grid]
            if temp_game.move(action) or original_grid != temp_game.grid:
                return action
        return "up"
        
    
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
        self.game_end_time = 0
        
        
    def get_state(self):
        return [cell for row in self.grid for cell in row]

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
        else:
            return (pygame.time.get_ticks() - self.start_time) / 1000.0
        
    def record_game_end_time(self):
        self.game_end_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
            
            
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
        self.add_random_tile()
        self.add_random_tile()                   
        
def run():
    game = GAME2048()
    game.add_random_tile()
    game.add_random_tile()
    
    clock = pygame.time.Clock()
    running = True
    
    ai_mode = True
    ai_thinking = False
    ai_delay = 0.2
    last_ai_move_time = 0
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if game.game_over:
                    if event.key == pygame.K_r:
                        game.reset_game()
                elif event.key == pygame.K_SPACE:
                    ai_mode = not ai_mode
                    print(f"AI Mode: {'ON' if ai_mode else 'OFF'}")
                else:
                    if not game.moving_animation and not ai_mode:
                        moved = False
                        if event.key == pygame.K_LEFT:
                            moved = game.move("left")
                        elif event.key == pygame.K_RIGHT:
                            moved = game.move("right")
                        elif event.key == pygame.K_UP:
                            moved = game.move("up")
                        elif event.key == pygame.K_DOWN:
                            moved = game.move("down")
                            
                        if moved:
                            game.game_over = game.is_game_over()
                            if game.game_over:
                                game.record_game_end_time()
                                
        if ai_mode and not game.game_over and not game.moving_animation:
            current_time = pygame.time.get_ticks() / 1000.0
            if current_time - last_ai_move_time >= ai_delay:
                best_action = mcts_search(game, iterations=100)
                moved = game.move(best_action)
                
                if moved:
                    game.game_over = game.is_game_over()
                    if game.game_over:
                        game.record_game_end_time()
                        
                last_ai_move_time = current_time
                
        # if not game.game_over:
        #     game.update_animation(dt)
        
        game.draw(screen)
        pygame.display.flip()
        
    pygame.quit()
    

if __name__ == "__main__":
    run()
