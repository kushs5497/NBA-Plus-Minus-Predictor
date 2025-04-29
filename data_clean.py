import pandas as pd
import os

def combine_box_scores():
    # Read regular season data
    regular_season_parts = [
        'regular_season_box_scores_2010_2024_part_1.csv',
        'regular_season_box_scores_2010_2024_part_2.csv',
        'regular_season_box_scores_2010_2024_part_3.csv'
    ]
    
    # Read playoff data
    playoff_data = pd.read_csv('play_off_box_scores_2010_2024.csv')
    playoff_data['game_type'] = 'playoff'
    
    # Combine regular season parts
    regular_season_data = pd.concat(
        [pd.read_csv(file) for file in regular_season_parts],
        ignore_index=True
    )
    regular_season_data['game_type'] = 'regular_season'
    
    # Combine regular season and playoff data
    combined_data = pd.concat(
        [regular_season_data, playoff_data],
        ignore_index=True
    )
    
    # Save combined data
    combined_data.to_csv('combined_box_scores_2010_2024.csv', index=False)
    print(f"Combined data saved to combined_box_scores_2010_2024.csv")
    print(f"Total rows: {len(combined_data)}")
    print(f"Regular season games: {len(regular_season_data)}")
    print(f"Playoff games: {len(playoff_data)}")

if __name__ == "__main__":
    combine_box_scores()
