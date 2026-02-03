# train.py
import os
import pickle
import neat
from neat import Checkpointer
from Neat_2048 import eval_genomes, eval_genome


def run_neat():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(Checkpointer(100))
    
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-500')


    winner = p.run(eval_genomes, 1000)  # 500+ generations
    
    print("\nTesting winner...")
    for i in range(10):
        fitness = eval_genome(winner, config)  # ← Use eval_genome (singular)
        print(f"Test {i+1}: {fitness}")
        
    winner_path = os.path.join(local_dir, 'winner.pkl')
    with open(winner_path, 'wb') as f:
        pickle.dump(winner, f)
        print("Winner genome saved!")

if __name__ == '__main__':
    run_neat()