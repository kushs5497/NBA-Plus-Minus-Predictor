import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_boxscore(game_id, season):
    try:
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        df = boxscore.player_stats.get_data_frame()
        df['SEASON'] = season
        print(f"Fetched {game_id}")
        return df
    except Exception as e:
        print(f"Failed {game_id}: {e}")
        return pd.DataFrame()

def process_season(season, max_threads=5):
    print(f"Processing season: {season}")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = gamefinder.get_data_frames()[0].drop_duplicates(subset='GAME_ID')
    game_ids = games['GAME_ID'].tolist()

    season_df = pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(fetch_boxscore, gid, season) for gid in game_ids]

        for future in as_completed(futures):
            df = future.result()
            season_df = pd.concat([season_df, df], ignore_index=True)
            time.sleep(0.4)  # still throttle a bit

    return season_df

# Run for all seasons
seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2015, 2023)]
full_df = pd.DataFrame()

for season in seasons:
    season_df = process_season(season)
    full_df = pd.concat([full_df, season_df], ignore_index=True)

# Save it
full_df.to_csv("nba_box_scores_2015_to_2023.csv", index=False)
print("All data saved to CSV.")
