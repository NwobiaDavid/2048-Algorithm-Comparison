import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

def load_and_compare_experiments():
    csv_files = glob.glob("benchmarks/*.csv")
    
    if not csv_files:
        print("No CSV files found in benchmarks folder!")
        return
    
    alg_data = {}
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path).lower()
        
        # Check if this is the summary CSV file (skip it)
        if 'comparison_summary' in filename:
            continue
            
        # Check if required columns exist
        required_cols = ['Final_Score', 'Highest_Tile', 'Time_Seconds', 'Number_of_Moves', 'Success']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file_path} - missing required columns")
            continue
        
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
        
        df['Algorithm'] = alg_name
        if alg_name in alg_data:
            alg_data[alg_name] = pd.concat([alg_data[alg_name], df], ignore_index=True)
        else:
            alg_data[alg_name] = df
    
    if not alg_data:
        print("No valid data found in CSV files!")
        return
    
    print(f"Found {len(alg_data)} different algorithms:")
    for alg, df in alg_data.items():
        print(f"  {alg}: {len(df)} games")
    
    create_comparison_plots(alg_data)
    generate_summary_statistics(alg_data)

def create_comparison_plots(data_dict):
    all_data = pd.concat(data_dict.values(), ignore_index=True)
    
    # Only keep rows with required columns
    required_cols = ['Final_Score', 'Highest_Tile', 'Time_Seconds', 'Number_of_Moves', 'Success', 'Algorithm']
    all_data = all_data[required_cols]
    all_data = all_data.dropna(subset=['Final_Score', 'Highest_Tile', 'Time_Seconds', 'Number_of_Moves'])
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    ax1 = plt.subplot(2, 3, 1)
    success_rates = all_data.groupby('Algorithm')['Success'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(success_rates)), success_rates.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(success_rates))))
    plt.xticks(range(len(success_rates)), success_rates.index, rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Comparison\n(Reached 2048 tile)')
    plt.ylim(0, 1)
    
    for i, v in enumerate(success_rates.values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = plt.subplot(2, 3, 2)
    avg_scores = all_data.groupby('Algorithm')['Final_Score'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_scores)), avg_scores.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_scores))))
    plt.xticks(range(len(avg_scores)), avg_scores.index, rotation=45, ha='right')
    plt.ylabel('Average Score')
    plt.title('Average Score Comparison')
    
    valid_scores = [v for v in avg_scores.values if not np.isnan(v)]
    if valid_scores:
        max_score = max(valid_scores)
        for i, v in enumerate(avg_scores.values):
            if not np.isnan(v):
                plt.text(i, v + max_score * 0.01, f'{int(v):,}', ha='center', va='bottom', fontsize=9)
    
    ax3 = plt.subplot(2, 3, 3)
    avg_max_tiles = all_data.groupby('Algorithm')['Highest_Tile'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_max_tiles)), avg_max_tiles.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_max_tiles))))
    plt.xticks(range(len(avg_max_tiles)), avg_max_tiles.index, rotation=45, ha='right')
    plt.ylabel('Average Max Tile')
    plt.title('Average Max Tile Comparison')
    
    valid_tiles = [v for v in avg_max_tiles.values if not np.isnan(v)]
    if valid_tiles:
        max_tile = max(valid_tiles)
        for i, v in enumerate(avg_max_tiles.values):
            if not np.isnan(v):
                plt.text(i, v + max_tile * 0.01, f'{int(v):,}', ha='center', va='bottom', fontsize=9)
    
    ax4 = plt.subplot(2, 3, 4)
    avg_times = all_data.groupby('Algorithm')['Time_Seconds'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_times)), avg_times.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_times))))
    plt.xticks(range(len(avg_times)), avg_times.index, rotation=45, ha='right')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Game Duration Comparison')
    
    valid_times = [v for v in avg_times.values if not np.isnan(v)]
    if valid_times:
        max_time = max(valid_times)
        for i, v in enumerate(avg_times.values):
            if not np.isnan(v):
                plt.text(i, v + max_time * 0.01, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
    
    ax5 = plt.subplot(2, 3, 5)
    avg_moves = all_data.groupby('Algorithm')['Number_of_Moves'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(avg_moves)), avg_moves.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(avg_moves))))
    plt.xticks(range(len(avg_moves)), avg_moves.index, rotation=45, ha='right')
    plt.ylabel('Average Number of Moves')
    plt.title('Average Moves Comparison')
    
    valid_moves = [v for v in avg_moves.values if not np.isnan(v)]
    if valid_moves:
        max_moves = max(valid_moves)
        for i, v in enumerate(avg_moves.values):
            if not np.isnan(v):
                plt.text(i, v + max_moves * 0.01, f'{int(v)}', ha='center', va='bottom', fontsize=9)
    
    ax6 = plt.subplot(2, 3, 6)
    tile_dist = all_data.groupby(['Algorithm', 'Highest_Tile']).size().unstack(fill_value=0)
    sns.heatmap(tile_dist.T, annot=True, fmt='d', cmap='YlOrRd', ax=ax6)
    plt.title('Tile Distribution by Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Highest Tile Achieved')
    
    plt.tight_layout()
    os.makedirs('benchmarks/graphs', exist_ok=True)
    plt.savefig('benchmarks/graphs/comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(data_dict):
    print("\n" + "="*80)
    print("DETAILED SUMMARY STATISTICS BY ALGORITHM")
    print("="*80)
    
    summary_stats = []
    
    for alg_name, df in data_dict.items():
        # Only keep rows with required columns
        required_cols = ['Final_Score', 'Highest_Tile', 'Time_Seconds', 'Number_of_Moves', 'Success']
        df = df[required_cols]
        df_clean = df.dropna(subset=required_cols)
        
        stats = {
            'Algorithm': alg_name,
            'Games_Played': len(df_clean),
            'Success_Rate': f"{df_clean['Success'].mean():.2%}" if len(df_clean) > 0 else "0.00%",
            'Avg_Score': f"{df_clean['Final_Score'].mean():,.0f}" if len(df_clean) > 0 else "0",
            'Median_Score': f"{df_clean['Final_Score'].median():,.0f}" if len(df_clean) > 0 else "0",
            'Std_Score': f"{df_clean['Final_Score'].std():,.0f}" if len(df_clean) > 0 else "0",
            'Max_Score': f"{df_clean['Final_Score'].max():,.0f}" if len(df_clean) > 0 else "0",
            'Min_Score': f"{df_clean['Final_Score'].min():,.0f}" if len(df_clean) > 0 else "0",
            'Avg_Max_Tile': f"{df_clean['Highest_Tile'].mean():,.0f}" if len(df_clean) > 0 else "0",
            'Median_Max_Tile': f"{df_clean['Highest_Tile'].median():,.0f}" if len(df_clean) > 0 else "0",
            'Avg_Time': f"{df_clean['Time_Seconds'].mean():.2f}s" if len(df_clean) > 0 else "0.00s",
            'Avg_Moves': f"{df_clean['Number_of_Moves'].mean():.0f}" if len(df_clean) > 0 else "0",
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('Success_Rate', ascending=False, key=lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce').fillna(0) / 100)
    
    print(summary_df.to_string(index=False))
    
    os.makedirs('benchmarks', exist_ok=True)
    summary_df.to_csv('benchmarks/algorithm_comparison_summary.csv', index=False)
    print(f"\nDetailed summary saved to: benchmarks/algorithm_comparison_summary.csv")
    
    print("\n" + "="*80)
    print("INDIVIDUAL ALGORITHM ANALYSIS")
    print("="*80)
    
    for alg_name, df in data_dict.items():
        required_cols = ['Final_Score', 'Highest_Tile', 'Time_Seconds', 'Number_of_Moves', 'Success']
        df = df[required_cols]
        df_clean = df.dropna(subset=required_cols)
        
        print(f"\n{alg_name.upper()}:")
        print(f"  Total Games: {len(df_clean)}")
        success_rate = df_clean['Success'].mean() if len(df_clean) > 0 else 0
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Score Range: {df_clean['Final_Score'].min() if len(df_clean) > 0 else 0:,} - {df_clean['Final_Score'].max() if len(df_clean) > 0 else 0:,}")
        print(f"  Average Score: {df_clean['Final_Score'].mean() if len(df_clean) > 0 else 0:,.0f}")
        print(f"  Max Tile Range: {df_clean['Highest_Tile'].min() if len(df_clean) > 0 else 0:,} - {df_clean['Highest_Tile'].max() if len(df_clean) > 0 else 0:,}")
        print(f"  Average Max Tile: {df_clean['Highest_Tile'].mean() if len(df_clean) > 0 else 0:,.0f}")
        print(f"  Average Time: {df_clean['Time_Seconds'].mean() if len(df_clean) > 0 else 0:.2f}s")
        print(f"  Average Moves: {df_clean['Number_of_Moves'].mean() if len(df_clean) > 0 else 0:.0f}")
        
        if len(df_clean) > 0:
            tile_counts = df_clean['Highest_Tile'].value_counts().sort_index(ascending=False)
            print("  Tile Distribution:")
            for tile, count in tile_counts.items():
                percentage = (count / len(df_clean)) * 100
                print(f"    {tile}: {count} games ({percentage:.1f}%)")
        else:
            print("  Tile Distribution: No valid data")
        
        print("-" * 40)

if __name__ == "__main__":
    load_and_compare_experiments()