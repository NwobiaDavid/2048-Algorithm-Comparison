import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

def load_latest_expectimax_data():
    """Load the most recent expectimax results CSV file."""
    # Find all expectimax result files in the experiments folder
    pattern = "benchmarks/expectimax_results_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError("No expectimax results CSV files found in experiments folder!")
    
    # Sort by modification time to get the latest file
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df

def visualize_performance(df):
    """Create comprehensive dashboard visualization."""
    # Calculate Metrics
    df['Moves_Per_Sec'] = df['Number_of_Moves'] / df['Time_Seconds']
    
    # Setup Dashboard
    sns.set_style("whitegrid")
    
    # Defined with Integer keys. 
    # Added 16384 and 32768 just in case your AI is really good!
    tile_colors = {
        1024: '#edc53f', 
        2048: '#edc22e', 
        4096: '#3e3933', 
        8192: '#3c3a32', 
        16384: '#24211d',
        32768: '#000000'
    }
    
    # Force Highest_Tile to be integers so they match the color dictionary keys
    df['Highest_Tile'] = df['Highest_Tile'].astype(int)
    
    # Filter colors to only include tiles actually present in your data
    # This prevents the error if you have a tile not in the list
    present_tiles = df['Highest_Tile'].unique()
    # If a tile shows up that isn't in our color list, give it a default gray
    palette = {tile: tile_colors.get(tile, '#776e65') for tile in present_tiles}

    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f'2048 Expectimax Algorithm Performance Analysis (N={len(df)})', fontsize=20, y=0.98)
    
    gs = fig.add_gridspec(2, 3)

    # --- PLOT 1: Reliability ---
    ax1 = fig.add_subplot(gs[0, 0])
    tile_counts = df['Highest_Tile'].value_counts().sort_index()
    # Use the safe 'palette' variable we created above
    # Convert to DataFrame for proper hue assignment
    tile_counts_df = pd.DataFrame({'Tile': tile_counts.index, 'Count': tile_counts.values})
    sns.barplot(data=tile_counts_df, x='Tile', y='Count', hue='Tile', palette=palette, ax=ax1, edgecolor='black', legend=False)
    ax1.bar_label(ax1.containers[0])
    ax1.set_title('Reliability: Highest Tile Achieved', fontweight='bold')
    ax1.set_ylabel('Number of Games')

    # --- PLOT 2: Score Distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        sns.kdeplot(data=df, x='Final_Score', hue='Highest_Tile', palette=palette, fill=True, ax=ax2, common_norm=False, warn_singular=False)
    except Exception:
        # Fallback to histogram if KDE fails (happens if only 1 data point exists for a tier)
        sns.histplot(data=df, x='Final_Score', hue='Highest_Tile', palette=palette, element='step', ax=ax2)
    ax2.set_title('Score Density by Tile Tier', fontweight='bold')

    # --- PLOT 3: Efficiency Map ---
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(data=df, x='Number_of_Moves', y='Final_Score', hue='Highest_Tile', 
                    palette=palette, style='Highest_Tile', s=80, alpha=0.8, ax=ax3)
    ax3.set_title('Efficiency Map (Linearity Check)', fontweight='bold')

    # --- PLOT 4: Computational Speed ---
    ax4 = fig.add_subplot(gs[1, 0])
    sns.boxplot(y=df['Moves_Per_Sec'], color='#f4f4f4', ax=ax4)
    sns.stripplot(y=df['Moves_Per_Sec'], color='#776e65', alpha=0.4, size=4, ax=ax4)
    ax4.set_title('Processing Speed (Moves/Sec)', fontweight='bold')

    # --- PLOT 5: Time vs Score Bubble Plot ---
    ax5 = fig.add_subplot(gs[1, 1:])
    scatter = ax5.scatter(df['Time_Seconds'], df['Final_Score'], 
                         c=[palette.get(t, '#776e65') for t in df['Highest_Tile']], 
                         s=df['Moves_Per_Sec']*2, 
                         alpha=0.7, edgecolors='white', linewidth=1)
    
    ax5.set_title('Time Efficiency (Duration vs Final Score)', fontweight='bold')
    ax5.set_xlabel('Game Duration (Seconds)')
    ax5.set_ylabel('Final Score')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# def create_survival_curve(df):
#     """Create a histogram showing the distribution of final scores."""
#     plt.figure(figsize=(12, 6))
    
#     # Create bins for the score ranges
#     min_score = int(df['Final_Score'].min())
#     max_score = int(df['Final_Score'].max())
#     bin_size = max(100, (max_score - min_score) // 20)  # Adjust bin size based on data range
    
#     bins = range(min_score, max_score + bin_size, bin_size)
    
#     plt.hist(df['Final_Score'], bins=bins, edgecolor='black', alpha=0.7)
#     plt.xlabel('Final Score')
#     plt.ylabel('Frequency (Number of Games)')
#     plt.title('Distribution of Final Scores (Survival Curve)\nShows Algorithm Consistency/Reliability')
#     plt.grid(True, alpha=0.3, axis='y')
    
#     # Add some statistics as text on the plot
#     mean_score = df['Final_Score'].mean()
#     std_score = df['Final_Score'].std()
#     plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.0f}')
    
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def simulate_growth_trajectory(df):
#     """Simulate growth trajectory by creating artificial move-score pairs."""
#     # Since we don't have per-move data, we'll interpolate between start (0,0) and end (moves, final_score)
#     # For demonstration, let's create sample trajectories based on the average progression
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Sample a few games to show individual trajectories (for visualization)
#     sample_size = min(10, len(df))  # Only show up to 10 sample trajectories
#     sample_df = df.sample(n=sample_size, random_state=42)
    
#     all_trajectories = []
    
#     for idx, row in sample_df.iterrows():
#         moves = int(row['Number_of_Moves'])
#         final_score = row['Final_Score']
        
#         # Create interpolated points from (0,0) to (moves, final_score)
#         if moves > 0:
#             move_points = np.linspace(0, moves, num=min(moves + 1, 100))  # Limit to 100 points for performance
#             score_points = np.interp(move_points, [0, moves], [0, final_score])
            
#             ax.plot(move_points, score_points, alpha=0.6, linewidth=1)
#             # Store for potential average calculation
#             all_trajectories.append((move_points, score_points))
    
#     ax.set_xlabel('Move Number')
#     ax.set_ylabel('Current Score')
#     ax.set_title('Sample Growth Trajectories (Efficiency Visualization)\nShows how quickly scores increase per move')
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Also show an average trajectory if possible
#     if all_trajectories:
#         # Find the maximum number of moves to align all trajectories
#         max_moves = max(int(row['Number_of_Moves']) for _, row in sample_df.iterrows()) if len(sample_df) > 0 else 100
        
#         # Create aligned arrays for averaging
#         aligned_scores = np.zeros((len(all_trajectories), max_moves + 1))
        
#         for i, (moves_arr, scores_arr) in enumerate(all_trajectories):
#             moves_int = [int(m) for m in moves_arr if int(m) <= max_moves]
#             scores_interp = np.interp(range(max_moves + 1), moves_int, scores_arr, right=0)
#             aligned_scores[i] = scores_interp
        
#         # Calculate average score at each move
#         avg_scores = np.mean(aligned_scores, axis=0)
#         moves_range = range(max_moves + 1)
        
#         # Plot the average trajectory
#         plt.figure(figsize=(12, 6))
#         plt.plot(moves_range, avg_scores, color='red', linewidth=2, label='Average Trajectory')
#         plt.fill_between(moves_range, 
#                         np.min(aligned_scores, axis=0), 
#                         np.max(aligned_scores, axis=0), 
#                         alpha=0.2, color='red', label='Min-Max Range')
#         plt.xlabel('Move Number')
#         plt.ylabel('Current Score')
#         plt.title('Average Growth Trajectory\nShows Average Efficiency Across All Games')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()

def main():
    try:
        # Load the data
        df = load_latest_expectimax_data()
        
        # Validate required columns
        required_cols = ['Game_Number', 'Highest_Tile', 'Final_Score', 'Time_Seconds', 'Number_of_Moves', 'Success']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("Creating Comprehensive Performance Dashboard...")
        visualize_performance(df)
        
        # print("Creating Survival Curve (Score Distribution)...")
        # create_survival_curve(df)
        
        # print("Creating Growth Trajectory Visualization...")
        # simulate_growth_trajectory(df)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total games: {len(df)}")
        print(f"Success rate (2048+): {(df['Success'].sum() / len(df)) * 100:.2f}%")
        print(f"Average score: {df['Final_Score'].mean():.2f}")
        print(f"Median score: {df['Final_Score'].median():.2f}")
        print(f"Score std deviation: {df['Final_Score'].std():.2f}")
        print(f"Average time: {df['Time_Seconds'].mean():.2f}s")
        print(f"Average moves: {df['Number_of_Moves'].mean():.2f}")
        print(f"Highest tile achieved: {df['Highest_Tile'].max()}")
        print(f"Most common highest tile: {df['Highest_Tile'].mode()[0] if len(df['Highest_Tile'].mode()) > 0 else 'N/A'}")
        
        # Consistency measure (coefficient of variation)
        cv_score = df['Final_Score'].std() / df['Final_Score'].mean()
        print(f"Score consistency (CV): {cv_score:.3f} ({'Consistent' if cv_score < 0.5 else 'Volatile'})")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()