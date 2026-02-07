# 2048 Algorithm Comparison

A comparative study of three different Artificial Intelligence approaches designed to play the game **2048**. This project implements and analyzes the performance of **Expectimax**, **Monte Carlo**, and **NEAT (NeuroEvolution of Augmenting Topologies)**.

> **Note:** This project serves as my introduction to Game AI. It represents my first attempt at teaching a computer to play a game, exploring various search and evolutionary strategies.
**Key finding:** Explicit probabilistic search (Expectimax) significantly outperformed evolutionary and simulation-based approaches in this stochastic puzzle environment.

## 🎯 Project Overview

The goal of this project was to determine which algorithmic approach yields the highest scores and most consistent wins in the stochastic environment of 2048.

I implemented three distinct agents:

1. **Expectimax Search**
2. **Monte Carlo Simulation**
3. **NEAT (Pure & Hybrid with Expectimax)**

## 🤖 The Algorithms

### 1. Expectimax

This agent treats the game as a tree search problem including "chance" nodes. Since 2048 spawns new tiles (2 or 4) at random locations, standard Minimax is insufficient. Expectimax calculates the weighted average of all possible outcomes based on the probability of tile spawns.

* **Heuristics used:** Multiple evaluation functions including corner bias, monotonicity, smoothness, empty tile maximization, and score-based heuristics
* **Dynamic Depth:** The search depth adapts based on board complexity (ranging from 2-8 levels deep depending on distinct tiles and occupied cells)
* **Caching:** Implements memoization to avoid recalculating the same board states
* **Chance Modeling:** Properly weights the probabilities of 2-tile (90%) and 4-tile (10%) spawns when calculating expected values
* **Optimization:** Uses bit manipulation for efficient board state representation and evaluation

### 2. Monte Carlo Tree Search

This agent uses rapid random simulations to determine the best move. For each possible move from the current position, it executes multiple random games to completion and selects the initial move that leads to the highest average final score.

* **Simulation Count:** Performs 25-50 random playouts per move decision to balance accuracy and speed
* **Optimization:** Implemented with Numba JIT compilation for significant performance improvements
* **Efficiency:** Uses NumPy arrays for fast board state manipulation and move calculations
* **Strategy:** Focuses on long-term reward rather than immediate gains, exploring the game tree through random sampling

### 3. NEAT (NeuroEvolution of Augmenting Topologies)

This approach uses genetic algorithms to evolve a neural network with 24 input features.

* **Input Features:** Enhanced feature set including normalized board state (16 values), max tile value, empty tile count, available moves (4 binary values), smoothness, and monotonicity
* **Pure NEAT:** Direct neural network control using the 24-input feature vector to output move decisions
* **NEAT + Expectimax:** A hybrid approach where the evolved neural network provides evaluation functions to guide the Expectimax search algorithm
* **Fitness Function:** Incorporates multiple factors including score, tile bonuses (with heavy rewards for reaching higher tiles like 2048+), corner positioning of max tiles, monotonicity, empty spaces, and smoothness
* **Training:** Evolves networks over 500 generations with specialized evaluation functions for both pure neural control and hybrid search approaches  


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
| **Expectimax**     | **79.00%**   | 47,694    | 2,880        | 48.38s   | 2,276     |
| Monte Carlo        | 15.00%       | 13,445    | 989          | 42.00s   | 783       |
| NEAT + Expectimax  | 10.00%       | 11,279    | 829          | 13.82s   | 680       |
| NEAT Pure          | 0.00%        | 2,433     | 185          | 1.92s    | 1,000   |



### Key Findings

**🥇 Expectimax Dominates**  
With a **79% success rate** in reaching the 2048 tile, Expectimax demonstrated superior strategic planning through its probabilistic lookahead and sophisticated heuristics. Its median max tile of 2048 (with many runs reaching 4096+) confirms consistent high-level play despite computational cost.

**🥈 Monte Carlo Shows Promise**  
Achieving a **15% success rate** with significantly lower computation time than Expectimax, Monte Carlo proved effective for opportunistic play. Its strength lies in exploring diverse game trajectories through random playouts, though it lacks the deterministic planning needed for consistent high-tile achievement.

**🥉 Hybrid NEAT+Expectimax Underperformed Expectations**  
Despite theoretical advantages of neural-guided search, the hybrid approach achieved only **10% success**—likely due to suboptimal network evaluation functions that didn't complement the search algorithm effectively. This highlights the challenge of integrating learned heuristics with explicit search.

**❌ Pure NEAT Failed to Generalize**  
The pure neural network approach achieved **0% success rate**, consistently stalling around the 128-256 tile range. While extremely fast (1.92s/game), it lacked the strategic depth required for complex board states—demonstrating that raw speed cannot compensate for poor decision quality in this domain.

### Performance Visualizations
![Comparison Summary](benchmarks/graphs/comparison_summary.png)

* **Expectimax:**  Consistently reached 2048+ tiles with the highest scores, demonstrating superior strategic planning through its probabilistic tree search and sophisticated heuristics.
* **Monte Carlo:** Showed moderate performance with 20% success rate, proving effective for short-term planning but struggling with long-term strategy due to random simulation limitations.
* **NEAT + Expectimax:** Performed surprisingly well considering the hybrid approach, achieving 10% success rate by leveraging neural network guidance within the search framework.
* **NEAT Pure:** While fastest in execution time, failed to achieve the target 2048 tile in any game, indicating that pure neural network control without search guidance wasn't sufficient for complex game states.


> **Critical Insight:** Computational cost directly correlates with performance in 2048.

## 🛠️ Installation & Usage

### Prerequisites

* Python 3.x
* `pygame` (for the game interface)
* `neat-python` (for the evolutionary algorithm)
* `matplotlib` (for visualizations)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/NwobiaDavid/2048-Algorithm-Comparison.git

```

2. Have fun



### Running the Agents

To run the Expectimax agent:


```bash
# Run Expectimax agent
python expectimax_2048.py

# Run Monte Carlo agent  
python mcts_2048.py

# Run NEAT agent
python neat_algo/train.py && python neat_algo/play_ai.py
```


## 📝 Author's Note & Future Improvements

As mentioned, this was my first time diving into AI for games. While I am proud of the Expectimax results, I acknowledge there is significant room for optimization:

* **Heuristic Tuning:** The weights for various evaluation functions (monotonicity, smoothness, corner positioning) could be optimized further using genetic algorithms or other hyperparameter optimization techniques.
* **Pruning:** Implementing Alpha-Beta pruning or similar optimization techniques could allow for deeper search depths without increasing computational complexity exponentially.
* **Bitboard Optimization:**  Representing the board using 64-bit integers would significantly speed up operations and enable larger-scale Monte Carlo simulations.

