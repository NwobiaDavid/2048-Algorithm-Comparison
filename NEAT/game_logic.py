import random

GRID_SIZE = 4

class Game2048Logic:
    def __init__(self):
        self.grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.grid[r][c] = 2 if random.random() < 0.9 else 4

    def compress_row(self, row):
        return [x for x in row if x != 0] + [0] * row.count(0)

    def merge_row(self, row):
        merged = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            if i < len(row)-1 and row[i] == row[i+1]:
                merged.append(row[i] * 2)
                self.score += row[i] * 2
                skip = True
            else:
                merged.append(row[i])
        return merged + [0] * (len(row) - len(merged))

    def move_left(self):
        new_grid = []
        for row in self.grid:
            compressed = self.compress_row(row)
            merged = self.merge_row(compressed)
            new_grid.append(merged)
        return new_grid

    def move_right(self):
        new_grid = []
        for row in self.grid:
            rev = row[::-1]
            compressed = self.compress_row(rev)
            merged = self.merge_row(compressed)
            new_grid.append(merged[::-1])
        return new_grid

    def transpose(self, grid):
        return [[grid[r][c] for r in range(GRID_SIZE)] for c in range(GRID_SIZE)]

    def move_up(self):
        transposed = self.transpose(self.grid)
        new_transposed = []
        for row in transposed:
            compressed = self.compress_row(row)
            merged = self.merge_row(compressed)
            new_transposed.append(merged)
        return self.transpose(new_transposed)

    def move_down(self):
        transposed = self.transpose(self.grid)
        new_transposed = []
        for row in transposed:
            rev = row[::-1]
            compressed = self.compress_row(rev)
            merged = self.merge_row(compressed)
            new_transposed.append(merged[::-1])
        return self.transpose(new_transposed)

    def move(self, direction):
        old_grid = [row[:] for row in self.grid]
        if direction == 0:  # left
            self.grid = self.move_left()
        elif direction == 1:  # right
            self.grid = self.move_right()
        elif direction == 2:  # up
            self.grid = self.move_up()
        elif direction == 3:  # down
            self.grid = self.move_down()
        else:
            return False

        if self.grid != old_grid:
            self.add_random_tile()
            self.moves += 1
            return True
        return False

    def is_game_over(self):
        if any(0 in row for row in self.grid):
            return False
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE - 1):
                if self.grid[r][c] == self.grid[r][c+1] or self.grid[c][r] == self.grid[c+1][r]:
                    return False
        return True

    def get_state(self):
        # Flatten grid into list of 16 values
        return [cell for row in self.grid for cell in row]

    def get_max_tile(self):
        return max(max(row) for row in self.grid)