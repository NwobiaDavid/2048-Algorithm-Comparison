# 2048 Algorithm Comparison

A comparative study of three AI approaches for playing 2048: **Expectimax**, **Monte Carlo**, and **NEAT**. This project explores which algorithmic strategy achieves the highest scores in this stochastic puzzle game.

> **Note:** This was my first foray into Game AI (a learning project comparing search-based and evolutionary approaches)

## 🎯 Project Overview

The goal of this project was to determine which algorithmic approach yields the highest scores and most consistent wins in the stochastic environment of 2048.

I implemented three distinct agents:

1. **Expectimax Search**
2. **Monte Carlo Simulation**
3. **NEAT (Pure & Hybrid with Expectimax)**

## 🤖 Algorithms

### 1. Expectimax
Probabilistic tree search modeling tile spawn randomness (90% twos, 10% fours). Uses dynamic depth (2-8 levels), corner bias, monotonicity, smoothness heuristics, and memoization for efficiency.

### 2. Monte Carlo Tree Search
Executes 25-50 random playouts per move, selecting actions with highest average outcomes. Numba JIT-compiled for performance.

### 3. NEAT (NeuroEvolution of Augmenting Topologies)
Evolves neural networks over 500 generations with 24-input features (board state, max tile, empty cells, monotonicity, smoothness). Tested both pure neural control and hybrid neural-guided Expectimax.

## 📊 Methodology

To ensure statistically significant comparisons, I conducted a rigorous benchmark:

* **Sample Size:** Each algorithm played **100 full games** to completion (or timeout)
* **Metrics Tracked:** 
  - Success rate (reaching 2048 tile)
  - Final score (average, median, standard deviation)
  - Maximum tile achieved
  - Execution time per game
  - Moves executed before termination
* **Hardware:** All tests run on identical hardware to ensure fair comparison

## 🏆 Results & Comparison


| Algorithm          | Success Rate | Avg Score | Avg Max Tile | Avg Time | Avg Moves |
|--------------------|--------------|-----------|--------------|----------|-----------|
| **Expectimax**     | **79%**      | 47,694    | 2,880        | 48.38s   | 2,276     |
| Monte Carlo        | 15%          | 13,445    | 989          | 42.00s   | 783       |
| NEAT + Expectimax  | 10%          | 11,279    | 829          | 13.82s   | 680       |
| NEAT Pure          | 0%           | 2,433     | 185          | 1.92s    | 1,000     |

![Comparison Summary](benchmarks/graphs/comparison_summary.png)

### Analysis

**Expectimax's dominance** (79% success) demonstrates that explicit probabilistic search with tuned heuristics significantly outperforms other approaches in this environment. Its computational cost (48s/game) translates directly into superior strategic planning.

**Monte Carlo** achieved moderate success (15%) with lower computational overhead, proving effective for opportunistic play but lacking deterministic long-term strategy.

**Hybrid NEAT underperformed** expectations (10%), suggesting that evolved evaluation functions didn't effectively complement the search algorithm. Pure NEAT completely failed (0%), indicating neural networks alone lack sufficient strategic depth for complex board states despite being 25x faster.

**Key insight:** In 2048, computational investment in search depth and heuristic quality directly correlates with performance.

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install pygame neat-python matplotlib numpy numba
```

### Running Agents
```bash
# Expectimax
python expectimax_2048.py

# Monte Carlo  
python mcts_2048.py

# NEAT
python neat_algo/train.py && python neat_algo/play_ai.py
```

## 📝 Author's Note & Future Improvements
As mentioned, this was my first time diving into AI for games. While I am proud of the Expectimax results, I acknowledge there is significant room for optimization:

- **Heuristic optimization** via genetic algorithms or Bayesian optimization
- **Alpha-Beta pruning** for deeper search without exponential cost increase
- **Bitboard representation** for faster state evaluation and Monte Carlo scaling
- **Neural network architecture search** for more effective hybrid approaches
