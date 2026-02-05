import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime

def load_and_compare_experiments():
    """
    Load all benchmark CSV files and create comparison plots
    """
    # Find all CSV files in benchmarks folder
    csv_files = glob.glob("benchmarks/*.csv")
    
    if not csv_files:
        print("No CSV files found in benchmarks folder!")
        return
    
    # Dictionary to hold dataframes for each algorithm
    alg_data = {}
    
    # Load each CSV file and categorize by algorithm
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path).lower()
        
        # Determine algorithm type based on filename
        if 'expectimax' in filename:
            alg_name = 'Expectimax'
        elif 'monte_carlo' in filename or 'monte' in filename:
            alg_name = 'Monte Carlo'
        elif 'neat' in filename and 'with_search' in filename:
            alg_name = 'NEAT + Expectimax'
        elif 'neat' in filename and 'pure' in filename:
            alg_name = 'NEAT Pure'
        elif 'neat' in filename:
            alg_name = 'NEAT'
        else:
            alg_name = 'Unknown'
        
        # Add algorithm column to dataframe
        df['Algorithm'] = alg_name
        if alg_name in alg_data:
            alg_data[alg_name] = pd.concat([alg_data[alg_name], df], ignore_index=True)
        else:
            alg_data[alg_name] = df
    
    print(f"Found {len(alg_data)} different algorithms:")
    for alg, df in alg_data.items():
        print(f"  {alg}: {len(df)} games")
    
    # Create comprehensive comparison
    create_comparison_plots(alg_data)
    
    # Generate summary statistics
    generate_summary_statistics(alg_data)


def create_comparison_plots(data_dict):
    """
    Create various comparison plots for the algorithms
    """
    # Combine all data for plotting
    all_data = pd.concat(data_dict.values(), ignore_index=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Success Rate Comparison
    ax1 = plt.subplot(2, 3, 1)
    success_rates = all_data.groupby('Algorithm')['Success'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(success_rates)), success_rates.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(success_rates))))
    plt.xticks(range(len(success_rates)), success_rates.index, rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Comparison\n(Reached 2048 tile)')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(success_rates.values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Average Score Comparison
    ax2 = plt.subplot(2, 3, 2)
    avg_scores = all_data.groupby('Algorithm')['Final_Score'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_scores)), avg_scores.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_scores))))
    plt.xticks(range(len(avg_scores)), avg_scores.index, rotation=45, ha='right')
    plt.ylabel('Average Score')
    plt.title('Average Score Comparison')
    
    # Add value labels on bars
    for i, v in enumerate(avg_scores.values):
        plt.text(i, v + max(avg_scores.values) * 0.01, f'{int(v):,}', ha='center', va='bottom', fontsize=9)
    
    # 3. Average Max Tile Comparison
    ax3 = plt.subplot(2, 3, 3)
    avg_max_tiles = all_data.groupby('Algorithm')['Highest_Tile'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_max_tiles)), avg_max_tiles.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_max_tiles))))
    plt.xticks(range(len(avg_max_tiles)), avg_max_tiles.index, rotation=45, ha='right')
    plt.ylabel('Average Max Tile')
    plt.title('Average Max Tile Comparison')
    
    # Add value labels on bars
    for i, v in enumerate(avg_max_tiles.values):
        plt.text(i, v + max(avg_max_tiles.values) * 0.01, f'{int(v):,}', ha='center', va='bottom', fontsize=9)
    
    # 4. Average Time Comparison
    ax4 = plt.subplot(2, 3, 4)
    avg_times = all_data.groupby('Algorithm')['Time_Seconds'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_times)), avg_times.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_times))))
    plt.xticks(range(len(avg_times)), avg_times.index, rotation=45, ha='right')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Game Duration Comparison')
    
    # Add value labels on bars
    for i, v in enumerate(avg_times.values):
        plt.text(i, v + max(avg_times.values) * 0.01, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 5. Average Moves Comparison
    ax5 = plt.subplot(2, 3, 5)
    avg_moves = all_data.groupby('Algorithm')['Number_of_Moves'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_moves)), avg_moves.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_moves))))
    plt.xticks(range(len(avg_moves)), avg_moves.index, rotation=45, ha='right')
    plt.ylabel('Average Number of Moves')
    plt.title('Average Moves Comparison')
    
    # Add value labels on bars
    for i, v in enumerate(avg_moves.values):
        plt.text(i, v + max(avg_moves.values) * 0.01, f'{int(v)}', ha='center', va='bottom', fontsize=9)
    
    # 6. Tile Distribution Heatmap
    ax6 = plt.subplot(2, 3, 6)
    # Create a pivot table for tile distribution
    tile_dist = all_data.groupby(['Algorithm', 'Highest_Tile']).size().unstack(fill_value=0)
    sns.heatmap(tile_dist.T, annot=True, fmt='d', cmap='YlOrRd', ax=ax6)
    plt.title('Tile Distribution by Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Highest Tile Achieved')
    
    plt.tight_layout()
    plt.savefig('benchmarks/graphs/comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_statistics(data_dict):
    """
    Generate detailed summary statistics for each algorithm
    """
    print("\n" + "="*80)
    print("DETAILED SUMMARY STATISTICS BY ALGORITHM")
    print("="*80)
    
    # Create a summary DataFrame
    summary_stats = []
    
    for alg_name, df in data_dict.items():
        stats = {
            'Algorithm': alg_name,
            'Games_Played': len(df),
            'Success_Rate': f"{df['Success'].mean():.2%}",
            'Avg_Score': f"{df['Final_Score'].mean():,.0f}",
            'Median_Score': f"{df['Final_Score'].median():,.0f}",
            'Std_Score': f"{df['Final_Score'].std():,.0f}",
            'Max_Score': f"{df['Final_Score'].max():,.0f}",
            'Min_Score': f"{df['Final_Score'].min():,.0f}",
            'Avg_Max_Tile': f"{df['Highest_Tile'].mean():,.0f}",
            'Median_Max_Tile': f"{df['Highest_Tile'].median():,.0f}",
            'Avg_Time': f"{df['Time_Seconds'].mean():.2f}s",
            'Avg_Moves': f"{df['Number_of_Moves'].mean():.0f}",
        }
        summary_stats.append(stats)
    
    # Create summary DataFrame and sort by success rate
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('Success_Rate', ascending=False, key=lambda x: pd.to_numeric(x.str.rstrip('%')) / 100)
    
    print(summary_df.to_string(index=False))
    
    # Also save to CSV
    summary_df.to_csv('benchmarks/algorithm_comparison_summary.csv', index=False)
    print(f"\nDetailed summary saved to: benchmarks/algorithm_comparison_summary.csv")
    
    # Create individual statistics for each algorithm
    print("\n" + "="*80)
    print("INDIVIDUAL ALGORITHM ANALYSIS")
    print("="*80)
    
    for alg_name, df in data_dict.items():
        print(f"\n{alg_name.upper()}:")
        print(f"  Total Games: {len(df)}")
        print(f"  Success Rate: {df['Success'].mean():.2%}")
        print(f"  Score Range: {df['Final_Score'].min():,} - {df['Final_Score'].max():,}")
        print(f"  Average Score: {df['Final_Score'].mean():,.0f}")
        print(f"  Max Tile Range: {df['Highest_Tile'].min():,} - {df['Highest_Tile'].max():,}")
        print(f"  Average Max Tile: {df['Highest_Tile'].mean():,.0f}")
        print(f"  Average Time: {df['Time_Seconds'].mean():.2f}s")
        print(f"  Average Moves: {df['Number_of_Moves'].mean():.0f}")
        
        # Tile distribution for this algorithm
        tile_counts = df['Highest_Tile'].value_counts().sort_index(ascending=False)
        print("  Tile Distribution:")
        for tile, count in tile_counts.items():
            percentage = (count / len(df)) * 100
            print(f"    {tile}: {count} games ({percentage:.1f}%)")
        
        print("-" * 40)


def create_detailed_comparison_plots(data_dict):
    """
    Create additional detailed comparison plots
    """
    all_data = pd.concat(data_dict.values(), ignore_index=True)
    
    # Score distribution by algorithm
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=all_data, x='Algorithm', y='Final_Score')
    plt.title('Score Distribution by Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('benchmarks/score_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Max tile distribution by algorithm
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=all_data, x='Algorithm', y='Highest_Tile')
    plt.title('Max Tile Distribution by Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('benchmarks/max_tile_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Time vs Score scatter plot colored by algorithm
    plt.figure(figsize=(14, 8))
    for alg in all_data['Algorithm'].unique():
        subset = all_data[all_data['Algorithm'] == alg]
        plt.scatter(subset['Time_Seconds'], subset['Final_Score'], label=alg, alpha=0.6)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Final Score')
    plt.title('Time vs Score by Algorithm')
    plt.legend()
    plt.tight_layout()
    plt.savefig('benchmarks/time_vs_score.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    load_and_compare_experiments()
    
    # Optionally create additional detailed plots
    # Uncomment the line below if you want more detailed plots
    # create_detailed_comparison_plots(data_dict)