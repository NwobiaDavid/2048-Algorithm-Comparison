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


def normalize_state(state):
    return [0.0 if x == 0 else math.log2(x) / 11.0 for x in state]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, T_WIN_SIZE))  # ← Add these if needed
    pygame.display.set_caption("2048 - AI Playing")
    
    game = GAME2048()  # Already adds two tiles in __init__

    game.add_random_tile()
    game.add_random_tile()

    clock = pygame.time.Clock()
    running = True
    auto_play = True

    while running:
        dt = clock.tick(5) / 1000.0

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
            direction_idx = output.index(max(output))
            direction = ["left", "right", "up", "down"][direction_idx]
            print(f"State: {state}")
            print(f"Outputs: {output}")
            print(f"Chosen: {direction}")

            moved = game.move(direction)
            print(f"Moved: {moved}")
            # if not moved:
            #     # If no move happened, check if the game is truly over
            #     if game.is_game_over():
            #         game.game_over = True
            #         game.record_game_end_time()
            # else:
            #     if game.is_game_over():
            #         game.game_over = True
            #         game.record_game_end_time()
            if not moved:
                if not hasattr(game, '_failed_moves'):
                    game._failed_moves = 0
                game._failed_moves += 1
                if game._failed_moves > 10:
                    game.game_over = True
                    game.record_game_end_time()
            else:
                game._failed_moves = 0

        if not game.game_over:
            game.update_animation(dt)

        game.draw(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()