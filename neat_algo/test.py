import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WINDOW_SIZE = 550  # Increased window size to accommodate padding
PADDING = 25
GRID_SIZE = 4
TILE_SIZE = (WINDOW_SIZE - 2 * PADDING - (GRID_SIZE + 1) * 10) // GRID_SIZE  # Adjusted calculation
FONT = pygame.font.SysFont('ComicSans', 32, bold=True)
SMALL_FONT = pygame.font.SysFont('ComicSans', 24, bold=True)
TILE_FONT = pygame.font.SysFont("comicsans", 32, bold=True)

# Colors
COLORS = {
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

# Create initial empty grid
# grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
grid = [
    [2048, 2048, 2048, 2048],
    [2048, 2048, 2048, 2048],
    [0, 0, 2048, 2048],
    [0, 2048, 2048, 2048]
]

# Set up display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("2048 Grid Editor")

def draw_grid():
    screen.fill((187, 173, 160))  # Background color
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = grid[row][col]
            color = COLORS.get(value, (60, 58, 50))
            rect = pygame.Rect(
                PADDING + col * (TILE_SIZE + 10),
                PADDING + row * (TILE_SIZE + 10),
                TILE_SIZE,
                TILE_SIZE
            )
            pygame.draw.rect(screen, color, rect, border_radius=5)
            
            if value != 0:
                text_color = (119, 110, 101) if value <= 4 else (249, 246, 242)
                text = FONT.render(str(value), True, text_color)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

def get_tile_at_pos(pos):
    x, y = pos
    # Calculate relative position within grid area
    rel_x = x - PADDING
    rel_y = y - PADDING
    
    # Check if click is within grid boundaries
    if 0 <= rel_x < (TILE_SIZE + 10) * GRID_SIZE and 0 <= rel_y < (TILE_SIZE + 10) * GRID_SIZE:
        col = min(rel_x // (TILE_SIZE + 10), GRID_SIZE - 1)
        row = min(rel_y // (TILE_SIZE + 10), GRID_SIZE - 1)
        return (row, col)
    return None, None

def toggle_tile_value(row, col):
    values = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    current_idx = values.index(grid[row][col])
    next_idx = (current_idx + 1) % len(values)
    grid[row][col] = values[next_idx]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                row, col = get_tile_at_pos(event.pos)
                if row is not None and col is not None:
                    toggle_tile_value(row, col)
            elif event.button == 3:  # Right click
                row, col = get_tile_at_pos(event.pos)
                if row is not None and col is not None:
                    grid[row][col] = 0
    
    draw_grid()
    pygame.display.flip()

pygame.quit()
sys.exit()