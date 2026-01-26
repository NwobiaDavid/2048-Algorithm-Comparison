# train.py
import os
import pickle
import neat
from neat import Checkpointer
from Neat_2048 import eval_genomes


def run_neat():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    p.add_reporter(Checkpointer(50))

    winner = p.run(eval_genomes, 1000)  # 100 generations

    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
        print("Winner genome saved!")

if __name__ == '__main__':
    run_neat()