import os
import pickle
import neat
from neat import Checkpointer
from neat_2048 import eval_genomes, eval_genome


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
    # p.add_reporter(Checkpointer(50))

    winner = p.run(eval_genomes, 500)
    
    print("\n" + "="*60)
    print("TESTING WINNER - Pure Neural Network (Fast)")
    print("="*60)
    for i in range(10):
        fitness = eval_genome(winner, config, use_search=False)
        print(f"Test {i+1}: {fitness:,.0f}")
    
    print("\n" + "="*60)
    print("TESTING WINNER - With Expectimax Search (Slow but Better)")
    print("="*60)
    for i in range(10):
        fitness = eval_genome(winner, config, use_search=True, search_depth=2)
        print(f"Test {i+1}: {fitness:,.0f}")
        
    winner_path = os.path.join(local_dir, 'winner.pkl')
    with open(winner_path, 'wb') as f:
        pickle.dump(winner, f)
        print("\nWinner genome saved!")

if __name__ == '__main__':
    run_neat()