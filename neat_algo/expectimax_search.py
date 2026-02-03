import random
import math

class ExpectimaxSearch:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.eval_cache = {}
    
    def search(self, game, net, input_function):
        """
        Perform expectimax search with improved heuristics
        """
        self.eval_cache.clear()
        best_score = float('-inf')
        best_direction = 0
        
        for direction in range(4):
            game_copy = self._copy_game(game)
            moved = game_copy.move(direction)
            
            if moved:
                score = self._expect_node(game_copy, 1, net, input_function)
                
                if score > best_score:
                    best_score = score
                    best_direction = direction
        
        return best_direction
    
    def _max_node(self, game, depth, net, input_function):
        """Player's turn - maximize score"""
        if depth >= self.max_depth or game.is_game_over():
            return self._evaluate_advanced(game, net, input_function)
        
        max_score = float('-inf')
        has_valid_move = False
        
        for direction in range(4):
            game_copy = self._copy_game(game)
            moved = game_copy.move(direction)
            
            if moved:
                has_valid_move = True
                score = self._expect_node(game_copy, depth + 1, net, input_function)
                max_score = max(max_score, score)
        
        if not has_valid_move:
            return self._evaluate_advanced(game, net, input_function)
        
        return max_score
    
    def _expect_node(self, game, depth, net, input_function):
        """Random tile placement - expected value"""
        if depth >= self.max_depth or game.is_game_over():
            return self._evaluate_advanced(game, net, input_function)
        
        empty_cells = [(r, c) for r in range(4) for c in range(4) if game.grid[r][c] == 0]
        
        if not empty_cells:
            return self._evaluate_advanced(game, net, input_function)
        
        total_score = 0.0
        
        if depth <= 1:
            sample_size = min(len(empty_cells), 6)
        else:
            sample_size = min(len(empty_cells), 3)
        
        sampled_cells = random.sample(empty_cells, sample_size) if len(empty_cells) > sample_size else empty_cells
        
        for row, col in sampled_cells:
            for tile_value, probability in [(2, 0.9), (4, 0.1)]:
                game_copy = self._copy_game(game)
                game_copy.grid[row][col] = tile_value
                
                score = self._max_node(game_copy, depth + 1, net, input_function)
                total_score += score * probability
        
        return total_score / len(sampled_cells)
    
    def _evaluate_advanced(self, game, net, input_function):
        """Advanced evaluation combining neural network and heuristics"""
       
        grid_tuple = tuple(tuple(row) for row in game.grid)
        if grid_tuple in self.eval_cache:
            return self.eval_cache[grid_tuple]
        
        inputs = input_function(game)
        output = net.activate(inputs)
        network_score = sum(output) * 500
        
        game_score = game.score
        
        empty_tiles = sum(1 for row in game.grid for cell in row if cell == 0)
        empty_bonus = empty_tiles ** 2 * 100
        
        max_tile = max(cell for row in game.grid for cell in row)
        max_tile_bonus = max_tile * 10
        
        monotonicity_score = self._calculate_monotonicity(game.grid)
        
        smoothness_score = self._calculate_smoothness(game.grid)
        
        corner_score = self._calculate_corner_score(game.grid)
        
        edge_score = self._calculate_edge_score(game.grid)
        
        merge_score = self._calculate_merge_potential(game.grid)
        
        evaluation = (
            network_score +
            game_score * 5 +
            empty_bonus +
            max_tile_bonus +
            monotonicity_score * 2000 +
            smoothness_score * 500 +
            corner_score * 5000 +
            edge_score * 1000 +
            merge_score * 300
        )
        
        self.eval_cache[grid_tuple] = evaluation
        return evaluation
    
    def _calculate_monotonicity(self, grid):
        score = 0
        
        for direction in ['left', 'right', 'up', 'down']:
            monotonic_count = 0
            
            if direction in ['left', 'right']:
                for row in grid:
                    non_zero = [x for x in row if x != 0]
                    if len(non_zero) <= 1:
                        continue
                    
                    if direction == 'left':
                        is_monotonic = all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1))
                    else:
                        is_monotonic = all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1))
                    
                    if is_monotonic:
                        monotonic_count += 1
            else:
                for col in range(4):
                    non_zero = [grid[row][col] for row in range(4) if grid[row][col] != 0]
                    if len(non_zero) <= 1:
                        continue
                    
                    if direction == 'up':
                        is_monotonic = all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1))
                    else:
                        is_monotonic = all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1))
                    
                    if is_monotonic:
                        monotonic_count += 1
            
            score = max(score, monotonic_count)
        
        return score
    
    def _calculate_smoothness(self, grid):
        """Penalty for large differences between adjacent tiles"""
        smoothness = 0
        
        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    value = math.log2(grid[i][j])
                    
                    # Check right neighbor
                    if j < 3 and grid[i][j+1] != 0:
                        neighbor_value = math.log2(grid[i][j+1])
                        smoothness -= abs(value - neighbor_value)
                    
                    # Check down neighbor
                    if i < 3 and grid[i+1][j] != 0:
                        neighbor_value = math.log2(grid[i+1][j])
                        smoothness -= abs(value - neighbor_value)
        
        return smoothness
    
    def _calculate_corner_score(self, grid):
        """Reward having max tile in a corner"""
        max_tile = max(cell for row in grid for cell in row)
        corners = [grid[0][0], grid[0][3], grid[3][0], grid[3][3]]
        
        if max_tile in corners:
            corner_index = corners.index(max_tile)
            
            if corner_index == 0:  # Top-left
                adjacent = [grid[0][1], grid[1][0]]
            elif corner_index == 1:  # Top-right
                adjacent = [grid[0][2], grid[1][3]]
            elif corner_index == 2:  # Bottom-left
                adjacent = [grid[2][0], grid[3][1]]
            else:  # Bottom-right
                adjacent = [grid[2][3], grid[3][2]]
            
            adjacent_bonus = sum(x for x in adjacent if x >= max_tile // 2)
            return max_tile + adjacent_bonus
        
        return 0
    
    def _calculate_edge_score(self, grid):
        """Reward having high tiles on edges"""
        edge_tiles = []
        
        edge_tiles.extend(grid[0])
        edge_tiles.extend(grid[3])
        
        edge_tiles.extend([grid[1][0], grid[2][0]])
        edge_tiles.extend([grid[1][3], grid[2][3]])
        
        edge_tiles.sort(reverse=True)
        return sum(edge_tiles[:8])
    
    def _calculate_merge_potential(self, grid):
        """Reward adjacent tiles that can merge"""
        merge_score = 0
        
        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    if j < 3 and grid[i][j] == grid[i][j+1]:
                        merge_score += grid[i][j]
                    
                    if i < 3 and grid[i][j] == grid[i+1][j]:
                        merge_score += grid[i][j]
        
        return merge_score
    
    def _copy_game(self, game):
        """Create a deep copy of the game state"""
        if hasattr(game, 'copy'):
            return game.copy()
        else:
            game_copy = type(game)()
            game_copy.grid = [row[:] for row in game.grid]
            game_copy.score = game.score
            return game_copy