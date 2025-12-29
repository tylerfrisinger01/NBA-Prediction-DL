import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2 
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_position_data(positions_path):
    """
    Load and standardize position data from separate CSV.
    Convert full position names to abbreviations and keep the last position for each player.
    """
    if not os.path.exists(positions_path):
        print(f"Position file not found: {positions_path}")
        return None

    def robust_csv_load(file_path):
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✓ Successfully loaded positions with {encoding} encoding")
                return df
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback with error replacement
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            print("✓ Loaded positions with UTF-8 error replacement")
            return df
        except Exception as e:
            print(f"✗ Failed to load positions: {e}")
            return None

    positions_df = robust_csv_load(positions_path)
    if positions_df is None:
        return None

    print(f"Loaded position data: {len(positions_df)} rows")
    column_map = {}
    for col in positions_df.columns:
        col_lower = col.lower()
        if 'fname' in col_lower or 'first' in col_lower:
            column_map[col] = 'first_name'
        elif 'lname' in col_lower or 'last' in col_lower:
            column_map[col] = 'last_name'
        elif 'playerid' in col_lower or 'id' in col_lower:
            column_map[col] = 'player_id'
        elif 'position' in col_lower or 'pos' in col_lower:
            column_map[col] = 'position'
        elif 'player' in col_lower and 'id' not in col_lower:
            column_map[col] = 'player'

    if column_map:
        positions_df = positions_df.rename(columns=column_map)

    if 'player' not in positions_df.columns:
        possible_player_cols = ['Player', 'NAME', 'name', 'player_name', 'PlayerName']
        for col in possible_player_cols:
            if col in positions_df.columns:
                positions_df['player'] = positions_df[col]
                break
        else:
            for col in positions_df.columns:
                if col not in ['position', 'Rk', 'Rank', 'rk', 'rank'] and 'id' not in col.lower():
                    positions_df['player'] = positions_df[col]
                    break

    required_cols = ['player', 'position']
    if 'player_id' in positions_df.columns:
        required_cols.append('player_id')

    missing_cols = [col for col in required_cols if col not in positions_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in position data: {missing_cols}")
        return None

    
    def convert_position_to_abbreviation(position):
        if pd.isna(position):
            return None

        position_str = str(position).upper().strip()
        if '-' in position_str:
            primary_position = position_str.split('-')[0].strip()
        else:
            primary_position = position_str

        position_mapping = {
            'POINT GUARD': 'PG', 'SHOOTING GUARD': 'SG', 'SMALL FORWARD': 'SF',
            'POWER FORWARD': 'PF', 'CENTER': 'C', 'GUARD': 'G', 'FORWARD': 'F',
            'PG': 'PG', 'SG': 'SG', 'SF': 'SF', 'PF': 'PF', 'C': 'C', 'G': 'G', 'F': 'F'
        }

        if primary_position in position_mapping:
            return position_mapping[primary_position]

        for full_name, abbrev in position_mapping.items():
            if full_name in primary_position:
                return abbrev

        return primary_position if len(primary_position) <= 2 else None

    positions_df['position'] = positions_df['position'].apply(convert_position_to_abbreviation)
    unique_positions = positions_df.drop_duplicates(subset=['player'], keep='last')
    
    return unique_positions[required_cols]

def merge_position_data(merged_df, positions_df):
    """Merge position data into main dataframe using player names only."""
    print("Merging position data...")
    merged_df = merged_df.copy()
    positions_df = positions_df.copy()
    
    merged_df['player'] = merged_df['player'].astype(str).str.strip()
    positions_df['player'] = positions_df['player'].astype(str).str.strip()

    merged_df = pd.merge(merged_df, positions_df, on='player', how='left')
    
    # Remove duplicates if any (keeping first occurrence)
    merged_df = merged_df.drop_duplicates()
    return merged_df

def add_player_specific_features(merged_df):
    """Add player-specific baseline features."""
    merged_df['player_career_avg_pts'] = merged_df.groupby('player')['PTS'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )

    if 'away_trad' in merged_df.columns:
        merged_df['player_avg_vs_opponent'] = merged_df.groupby(['player', 'away_trad'])['PTS'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        ).fillna(merged_df['player_career_avg_pts'])
    else:
        merged_df['player_avg_vs_opponent'] = merged_df['player_career_avg_pts']

    if 'home_game' in merged_df.columns:
        home_avg = merged_df[merged_df['home_game'] == 1].groupby('player')['PTS'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        away_avg = merged_df[merged_df['home_game'] == 0].groupby('player')['PTS'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        merged_df['home_away_scoring_diff'] = (home_avg - away_avg).fillna(0)
    else:
        merged_df['home_away_scoring_diff'] = 0

    return merged_df

def add_opponent_features(merged_df):
    """Add features about the opponent's defensive strength."""
    if 'away_trad' in merged_df.columns:
        merged_df['opponent_avg_points_allowed'] = merged_df.groupby('away_trad')['PTS'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        merged_df['opponent_recent_defense'] = merged_df.groupby('away_trad')['PTS'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
    else:
        merged_df['opponent_avg_points_allowed'] = merged_df['PTS'].mean()
        merged_df['opponent_recent_defense'] = merged_df['PTS'].mean()

    return merged_df

def add_matchup_history_vectorized(merged_df):
    """VECTORIZED: Calculate matchup history"""
    print("Calculating matchup history...")
    defender_mask = merged_df['likely_defender'].notna()
    if not defender_mask.any():
        merged_df['historical_pts_vs_defender'] = 0
        merged_df['num_previous_matchups'] = 0
        merged_df['pts_std_vs_defender'] = 0
        return merged_df

    merged_df['matchup_key'] = merged_df['player'] + '_vs_' + merged_df['likely_defender'].fillna('None')
    merged_df = merged_df.sort_values(['matchup_key', 'date'])
    matchup_groups = merged_df.groupby('matchup_key')

    merged_df['historical_pts_vs_defender'] = matchup_groups['PTS'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    ).fillna(0)
    merged_df['num_previous_matchups'] = matchup_groups['PTS'].transform(
        lambda x: x.expanding(min_periods=1).count().shift(1)
    ).fillna(0)
    merged_df['pts_std_vs_defender'] = matchup_groups['PTS'].transform(
        lambda x: x.expanding(min_periods=2).std().shift(1)
    ).fillna(0)

    merged_df.drop('matchup_key', axis=1, inplace=True)
    return merged_df

def add_defender_stats_vectorized(merged_df):
    """VECTORIZED: Calculate defender stats"""
    print("Calculating defender stats...")
    defensive_stats = ['STL', 'BLK', 'DEFRTG']
    merged_df = merged_df.sort_values(['player', 'date'])

    for stat in defensive_stats:
        if stat in merged_df.columns:
            merged_df[f'{stat}_rolling_10'] = merged_df.groupby('player')[stat].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
            )

    recent_stats = merged_df.sort_values('date').groupby('player').tail(1)
    recent_stats_lookup = recent_stats.set_index('player')[[f'{stat}_rolling_10' for stat in defensive_stats if f'{stat}_rolling_10' in recent_stats.columns]]

    for stat in defensive_stats:
        stat_col = f'{stat}_rolling_10'
        if stat_col in recent_stats_lookup.columns:
            merged_df[f'defender_avg_{stat}'] = merged_df['likely_defender'].map(
                recent_stats_lookup[stat_col]
            ).fillna(0)
        else:
            merged_df[f'defender_avg_{stat}'] = 0

    merged_df['defender_games_considered'] = 10
    for stat in defensive_stats:
        if f'{stat}_rolling_10' in merged_df.columns:
            merged_df.drop(f'{stat}_rolling_10', axis=1, inplace=True)

    return merged_df

def add_player_matchup_features_minutes_only(merged_df):
    """Fallback: Uses minutes based matching when position data is not available."""
    print("Using minutes only matching...")
    if 'team_trad' not in merged_df.columns or 'away_trad' not in merged_df.columns:
        return merged_df

    merged_df['game_id'] = merged_df['date'].astype(str) + '_' + merged_df['home_trad'] + '_' + merged_df['away_trad']

    def get_players_with_minutes(game_group):
        return game_group[['player', 'MIN', 'team_trad']].to_dict('records')

    game_players = merged_df.groupby('game_id').apply(
        lambda x: get_players_with_minutes(x), include_groups=False
    ).reset_index(name='players')
    
    game_players_map = {row['game_id']: row['players'] for _, row in game_players.iterrows()}

    def find_likely_defender_by_minutes(player_row, game_players_map):
        game_id = player_row['game_id']
        if game_id not in game_players_map: return None
        
        all_players = game_players_map[game_id]
        opponent_team = player_row['away_trad'] if player_row['team_trad'] == player_row['home_trad'] else player_row['home_trad']
        
        opponent_players = [p for p in all_players if p['team_trad'] == opponent_team]
        if not opponent_players: return None

        best_match = None
        min_minutes_diff = float('inf')
        for player in opponent_players:
            minutes_diff = abs(player['MIN'] - player_row['MIN'])
            if minutes_diff < min_minutes_diff and player['MIN'] > 0:
                min_minutes_diff = minutes_diff
                best_match = player
        return best_match['player'] if best_match else None

    print("Finding defensive matchups based on minutes played...")
    merged_df['likely_defender'] = merged_df.apply(lambda x: find_likely_defender_by_minutes(x, game_players_map), axis=1)
    
    merged_df = add_defender_stats_vectorized(merged_df)
    merged_df = add_matchup_history_vectorized(merged_df)
    merged_df.drop('game_id', axis=1, inplace=True)
    return merged_df

def add_player_matchup_features_with_position_fast(merged_df):
    """Uses position data for efficient and realistic matching."""
    print("player matchup features with position based matching...")
    if 'team_trad' not in merged_df.columns or 'away_trad' not in merged_df.columns:
        return merged_df
    if 'position' not in merged_df.columns or merged_df['position'].isna().all():
        return add_player_matchup_features_minutes_only(merged_df)

    merged_df['game_id'] = merged_df['date'].astype(str) + '_' + merged_df['home_trad'] + '_' + merged_df['away_trad']
    print("Pre-caching opponent data...")
    game_opponent_cache = {}

    for game_id, game_data in merged_df.groupby('game_id'):
        home_team = game_data['home_trad'].iloc[0]
        away_team = game_data['away_trad'].iloc[0]
        home_players = game_data[game_data['team_trad'] == home_team]
        away_players = game_data[game_data['team_trad'] == away_team]
        
        home_by_position = {}
        away_by_position = {}
        for position, group in home_players.groupby('position'):
            home_by_position[position] = group[['player', 'MIN']].to_dict('records')
        for position, group in away_players.groupby('position'):
            away_by_position[position] = group[['player', 'MIN']].to_dict('records')

        game_opponent_cache[game_id] = {
            'home_team': home_team, 'away_team': away_team,
            'home_by_position': home_by_position, 'away_by_position': away_by_position
        }

    def find_defender_fast(player_row):
        game_id = player_row['game_id']
        if game_id not in game_opponent_cache: return None
        cache = game_opponent_cache[game_id]
        
        opponent_positions = cache['away_by_position'] if player_row['team_trad'] == cache['home_team'] else cache['home_by_position']
        player_position = player_row['position']
        
        opponent_candidates = opponent_positions.get(player_position, [])
        if not opponent_candidates:
            position_groups = {
                'PG': ['PG', 'SG'], 'SG': ['SG', 'PG', 'SF'], 'SF': ['SF', 'SG', 'PF'],
                'PF': ['PF', 'SF', 'C'], 'C': ['C', 'PF']
            }
            for pos in position_groups.get(player_position, []):
                if pos in opponent_positions:
                    opponent_candidates.extend(opponent_positions[pos])
                    break
        
        if not opponent_candidates:
            for pos_players in opponent_positions.values():
                opponent_candidates.extend(pos_players)

        if not opponent_candidates: return None
        
        best_match = None
        min_minutes_diff = float('inf')
        for candidate in opponent_candidates:
            if candidate['MIN'] > 0:
                minutes_diff = abs(candidate['MIN'] - player_row['MIN'])
                if minutes_diff < min_minutes_diff:
                    min_minutes_diff = minutes_diff
                    best_match = candidate
        return best_match['player'] if best_match else None

    print("Finding defensive matchups...")
    merged_df['likely_defender'] = merged_df.apply(find_defender_fast, axis=1)
    
    merged_df = add_defender_stats_vectorized(merged_df)
    merged_df = add_matchup_history_vectorized(merged_df)
    merged_df.drop('game_id', axis=1, inplace=True)
    return merged_df

def add_player_matchup_features(merged_df):
    if 'position' in merged_df.columns and merged_df['position'].notna().any():
        return add_player_matchup_features_with_position_fast(merged_df)
    else:
        return add_player_matchup_features_minutes_only(merged_df)

def add_enhanced_opponent_features(merged_df):
    print("Adding enhanced opponent features with matchup context...")
    if 'away_trad' not in merged_df.columns:
        return merged_df

    merged_df['opponent_defensive_strength'] = merged_df.groupby('away_trad')['PTS'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    if 'STL' in merged_df.columns:
        merged_df['opponent_wing_defense'] = merged_df.groupby('away_trad')['STL'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
    if 'BLK' in merged_df.columns:
        merged_df['opponent_paint_defense'] = merged_df.groupby('away_trad')['BLK'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
    if 'DEFRTG' in merged_df.columns:
        merged_df['opponent_defensive_trend'] = merged_df.groupby('away_trad')['DEFRTG'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
    return merged_df

def add_missing_context_features(merged_df):
    print("Adding missing context features...")
    merged_df = merged_df.sort_values(['player', 'date'])
    merged_df['days_rest'] = merged_df.groupby('player')['date'].diff().dt.days.fillna(7)
    merged_df['is_back_to_back'] = (merged_df['days_rest'] == 1).astype(int)
    merged_df['is_3_in_4'] = (merged_df['days_rest'] <= 1).astype(int)

    if 'away_trad' in merged_df.columns:
        merged_df['opponent_defensive_form_5g'] = merged_df.groupby('away_trad')['PTS'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )

    merged_df['MIN_volatility_10g'] = merged_df.groupby('player')['MIN'].transform(
        lambda x: x.rolling(window=10, min_periods=1).std().shift(1)
    ).fillna(0)

    if 'PACE' in merged_df.columns:
        merged_df['pace_adjustment'] = (
            merged_df['PACE'] / merged_df.groupby('player')['PACE'].transform(
                lambda x: x.expanding(min_periods=5).mean().shift(1)
            )
        ).fillna(1.0)
    return merged_df

def add_player_role_clusters(merged_df):
    print("Adding player role clusters...")
    player_archetypes = merged_df.groupby('player').agg({
        'PTS': ['mean', 'std'], 'MIN': ['mean', 'std'],
        'USG%': 'mean', 'AST': 'mean', 'REB': 'mean'
    }).round(3)
    player_archetypes.columns = ['_'.join(col).strip() for col in player_archetypes.columns.values]
    player_archetypes = player_archetypes.reset_index()

    def classify_role(row):
        if row['PTS_mean'] > 20 and row['MIN_mean'] > 32: return 'star'
        elif row['PTS_mean'] > 10 and row['MIN_mean'] > 25: return 'starter'
        elif row['MIN_mean'] > 15: return 'rotation'
        else: return 'bench'

    player_archetypes['player_role'] = player_archetypes.apply(classify_role, axis=1)
    merged_df = pd.merge(merged_df, player_archetypes[['player', 'player_role']], on='player', how='left')
    role_dummies = pd.get_dummies(merged_df['player_role'], prefix='role')
    merged_df = pd.concat([merged_df, role_dummies], axis=1)
    return merged_df

def add_outlier_detection_features(merged_df):
    print("outlier detection features...")
    merged_df['career_high'] = merged_df.groupby('player')['PTS'].transform(
        lambda x: x.expanding(min_periods=1).max().shift(1)
    )
    merged_df['pts_vs_career_high'] = merged_df['player_career_avg_pts'] / merged_df['career_high'].replace(0, 1)

    if 'pts_last_3_avg' not in merged_df.columns:
        merged_df['pts_last_3_avg'] = merged_df.groupby('player')['PTS'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
    merged_df['recent_spike'] = (
        merged_df['pts_last_3_avg'] - merged_df['player_career_avg_pts']
    ) / merged_df['PTS_cum_std'].replace(0, 1)

    if 'defender_avg_DEFRTG' in merged_df.columns:
        merged_df['matchup_exploit'] = (
            merged_df['player_career_avg_pts'] / merged_df['defender_avg_DEFRTG'].replace(0, 1)
        ) * 100
    return merged_df

def build_advanced_dnn_model(input_shape):
    """
    Simpler model to reduce overfitting.
    """
    
    reg = l2(0.01)
    
    model = Sequential([
        
        Dense(64, activation='relu', kernel_regularizer=reg, input_shape=[input_shape]),
        BatchNormalization(),
        Dropout(0.3),
        
        
        Dense(32, activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.2),
        
        
        Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mean_absolute_error'])
    return model

def load_and_preprocess_data(game_logs_path, advanced_stats_path, positions_path=None):
    """Loads and preprocesses player stats."""
    if not os.path.exists(game_logs_path) or not os.path.exists(advanced_stats_path):
        print(f"Error: One or both files not found.")
        return None, None, None, None, None

    try:
        game_logs = pd.read_csv(game_logs_path)
        advanced_stats = pd.read_csv(advanced_stats_path)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None, None, None, None, None

    game_logs.rename(columns={'Player': 'player', 'Date': 'date'}, inplace=True)
    advanced_stats.rename(columns={'Player': 'player', 'Date': 'date'}, inplace=True)
    game_logs['date'] = pd.to_datetime(game_logs['date'], errors='coerce')
    advanced_stats['date'] = pd.to_datetime(advanced_stats['date'], errors='coerce')

    merged_df = pd.merge(game_logs, advanced_stats, on=['player', 'date', 'MIN'], how='inner', suffixes=('_trad', '_adv'))
    merged_df = merged_df.drop_duplicates(subset=['player', 'date']).sort_values(by=['player', 'date'])

    if positions_path:
        positions_df = load_position_data(positions_path)
        if positions_df is not None:
            merged_df = merge_position_data(merged_df, positions_df)
        else:
            merged_df['position'] = None
    else:
        merged_df['position'] = None

    # Base essential columns
    base_essential_cols = ['player', 'date', 'MIN', 'PTS', 'FGA', 'AST', 'REB', 'TOV',
                          'TS%', 'USG%', 'NETRTG', 'AST%', 'REB%', 'PACE', 'PIE',
                          'FG%', '3P%', 'FT%', 'STL', 'BLK', 'DEFRTG', 'OREB%', 'DREB%']

    home_col = next((col for col in ['home_trad', 'home_adv'] if col in merged_df.columns), None)
    away_col = next((col for col in ['away_trad', 'away_adv'] if col in merged_df.columns), None)
    win_col = next((col for col in ['win_trad', 'win_adv'] if col in merged_df.columns), None)
    team_col = next((col for col in ['team_trad', 'team_adv'] if col in merged_df.columns), None)

    essential_cols = base_essential_cols + [c for c in [home_col, away_col, win_col, team_col] if c]
    merged_df.dropna(subset=essential_cols, inplace=True)

    if merged_df.empty: return None, None, None, None, None

    stats_to_average = ['MIN', 'FGA', 'AST', 'REB', 'TOV', 'TS%', 'USG%', 'NETRTG', 'AST%', 'REB%', 'PACE', 'PIE']
    rolling_window = 7
    
    for stat in stats_to_average:
        merged_df[f'{stat}_last_{rolling_window}'] = merged_df.groupby('player')[stat].transform(
             lambda x: x.rolling(window=rolling_window, min_periods=1).mean().shift(1)
        )
        merged_df[f'{stat}_cum_avg'] = merged_df.groupby('player')[stat].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )

    merged_df['PTS_cum_std'] = merged_df.groupby('player')['PTS'].transform(
        lambda x: x.expanding(min_periods=1).std().shift(1).fillna(0)
    )

    shooting_stats = ['FG%', '3P%', 'FT%']
    for stat in shooting_stats:
        if stat in merged_df.columns:
            merged_df[f'{stat}_last_7'] = merged_df.groupby('player')[stat].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).mean().shift(1)
            )

    defensive_stats = ['STL', 'BLK', 'DEFRTG', 'OREB%', 'DREB%']
    for stat in defensive_stats:
        if stat in merged_df.columns:
            merged_df[f'{stat}_last_7'] = merged_df.groupby('player')[stat].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).mean().shift(1)
            )

    if home_col and team_col:
        merged_df['home_game'] = (merged_df[team_col] == merged_df[home_col]).astype(int)

    if win_col:
        if merged_df[win_col].dtype == 'object':
            merged_df[win_col] = (merged_df[win_col] == 'W').astype(int)
        merged_df['win_streak'] = merged_df.groupby('player')[win_col].transform(
            lambda x: x.rolling(window=10, min_periods=1).sum().shift(1)
        )

    if 'USG%_last_5' in merged_df.columns and 'TS%_last_5' in merged_df.columns:
        merged_df['high_usage_efficiency'] = (merged_df['USG%_last_5'] * merged_df['TS%_last_5']).fillna(0)

    merged_df['MIN_volatility'] = merged_df.groupby('player')['MIN'].transform(
        lambda x: x.rolling(window=10, min_periods=1).std().shift(1).fillna(0)
    )

    merged_df = add_player_specific_features(merged_df)
    merged_df = add_opponent_features(merged_df)
    merged_df = add_player_matchup_features(merged_df)
    merged_df = add_enhanced_opponent_features(merged_df)
    merged_df = add_missing_context_features(merged_df)
    merged_df = add_player_role_clusters(merged_df)
    merged_df = add_outlier_detection_features(merged_df)

    features = []
    for stat in stats_to_average:
        features.append(f'{stat}_cum_avg')
        features.append(f'{stat}_last_{rolling_window}')
    features.append('PTS_cum_std')

    for stat in shooting_stats + defensive_stats:
        if f'{stat}_last_7' in merged_df.columns:
            features.append(f'{stat}_last_7')

    context_list = ['home_game', 'win_streak', 'high_usage_efficiency', 'MIN_volatility']
    for feat in context_list:
        if feat in merged_df.columns: features.append(feat)

    new_features = [
        'player_career_avg_pts', 'player_avg_vs_opponent', 'home_away_scoring_diff',
        'opponent_avg_points_allowed', 'opponent_recent_defense',
        'defender_avg_STL', 'defender_avg_BLK', 'defender_avg_DEFRTG', 'defender_games_considered',
        'historical_pts_vs_defender', 'num_previous_matchups', 'pts_std_vs_defender',
        'opponent_defensive_strength', 'opponent_wing_defense', 'opponent_paint_defense', 'opponent_defensive_trend',
        'days_rest', 'is_back_to_back', 'is_3_in_4', 'opponent_defensive_form_5g', 'MIN_volatility_10g', 'pace_adjustment',
        'career_high', 'pts_vs_career_high', 'recent_spike', 'matchup_exploit'
    ]
    
    role_features = [col for col in merged_df.columns if col.startswith('role_')]
    
    for feature in new_features + role_features:
        if feature in merged_df.columns:
            features.append(feature)

    target = 'PTS'
    cleaned_df = merged_df[essential_cols + features].dropna()

    if cleaned_df.empty: return None, None, None, None, None

    X = cleaned_df[features]
    y = cleaned_df[[target]]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, feature_scaler, target_scaler, cleaned_df

def time_based_split(df, test_size=0.2):
    df_sorted = df.sort_values('date')
    split_idx = int(len(df) * (1 - test_size))
    train_indices = df_sorted.index[:split_idx]
    test_indices = df_sorted.index[split_idx:]
    return train_indices, test_indices

def enhanced_evaluation(model, X_test, y_test, target_scaler, df_test_metadata):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    mae_unscaled = mae * target_scaler.scale_[0]

    predictions_scaled = model.predict(X_test, verbose=0)
    predictions_unscaled = target_scaler.inverse_transform(predictions_scaled)
    actuals_unscaled = target_scaler.inverse_transform(y_test)

    mse = np.mean((actuals_unscaled - predictions_unscaled) ** 2)
    rmse = np.sqrt(mse)
    errors = np.abs(actuals_unscaled - predictions_unscaled)
    
    print(f"\n--- Enhanced Evaluation ---")
    print(f"Test Set Mean Absolute Error: {mae_unscaled:.2f} PTS")
    print(f"Root Mean Squared Error: {rmse:.2f} PTS")
    print(f"Within 3 points: {np.mean(errors <= 3) * 100:.1f}%")
    print(f"Within 5 points: {np.mean(errors <= 5) * 100:.1f}%")
    print(f"Within 8 points: {np.mean(errors <= 8) * 100:.1f}%")

    return predictions_unscaled, actuals_unscaled


def create_visualizations(model, history, X_test, y_test, target_scaler, predictions_unscaled, actuals_unscaled, df_test_metadata, feature_names=None):
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NBA Player Points Prediction - Model Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Training History
    if history is not None:
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Actual vs Predicted Scatter Plot
    axes[0, 1].scatter(actuals_unscaled, predictions_unscaled, alpha=0.6, s=10)
    max_val = max(actuals_unscaled.max(), predictions_unscaled.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_title('Actual vs Predicted Points')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residuals Plot
    residuals = actuals_unscaled - predictions_unscaled
    axes[0, 2].scatter(predictions_unscaled, residuals, alpha=0.6, s=10)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_title('Residuals vs Predicted')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Error Distribution
    errors = np.abs(residuals)
    axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Prediction Errors')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Accuracy by Point Ranges
    error_ranges = [0, 3, 5, 8, 15, 50]
    accuracy_per_range = []
    range_labels = []
    for i in range(len(error_ranges)-1):
        lower, upper = error_ranges[i], error_ranges[i+1]
        in_range = np.sum((errors >= lower) & (errors < upper))
        accuracy_per_range.append((in_range / len(errors)) * 100)
        range_labels.append(f'{lower}-{upper}')

    bars = axes[1, 1].bar(range_labels, accuracy_per_range, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Prediction Accuracy by Error Range')
    for bar, value in zip(bars, accuracy_per_range):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{value:.1f}%', ha='center')

    # 6. MAE by Actual Points Range
    point_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 50)]
    mae_per_range = []
    range_labels_points = []
    for lower, upper in point_ranges:
        mask = (actuals_unscaled >= lower) & (actuals_unscaled < upper)
        if np.sum(mask) > 0:
            mae_per_range.append(np.mean(errors[mask.flatten()]))
            range_labels_points.append(f'{lower}-{upper}')
        else:
            mae_per_range.append(0)
            range_labels_points.append(f'{lower}-{upper}')

    bars2 = axes[1, 2].bar(range_labels_points, mae_per_range, color='lightcoral', edgecolor='black')
    axes[1, 2].set_title('MAE by Actual Points Range')

    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    try:
        sample_size = min(1000, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[sample_indices]
        y_sample = y_test[sample_indices]

        perm_importance = permutation_importance(
            model, X_sample, y_sample,
            n_repeats=5, random_state=42, scoring='neg_mean_absolute_error'
        )

        if feature_names is None:
            feature_labels = [f'Feature {i}' for i in range(X_test.shape[1])]
        else:
            feature_labels = feature_names

        
        top_n = 30
        sorted_idx = perm_importance.importances_mean.argsort()[::-1][:top_n]

        plt.figure(figsize=(12, 14)) 
        plt.barh(range(len(sorted_idx)),
                perm_importance.importances_mean[sorted_idx],
                xerr=perm_importance.importances_std[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_labels[i] for i in sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis() 
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Could not compute feature importance: {e}")

def analyze_recent_player_performance(model, X_test, y_test, target_scaler, cleaned_df, test_indices, top_n=10, recent_years=1):
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = target_scaler.inverse_transform(predictions_scaled)
    actuals = target_scaler.inverse_transform(y_test)

    results_df = cleaned_df.loc[test_indices, ['player', 'PTS', 'date']].copy()
    results_df['Predicted_PTS'] = predictions
    results_df['Error'] = np.abs(results_df['PTS'] - results_df['Predicted_PTS'])

    if 'date' in results_df.columns:
        results_df['date'] = pd.to_datetime(results_df['date'])
        cutoff_date = results_df['date'].max() - pd.DateOffset(years=recent_years)
        recent_results_df = results_df[results_df['date'] >= cutoff_date]
        print(f"Analyzing {len(recent_results_df['player'].unique())} recent players (since {cutoff_date.strftime('%Y-%m-%d')})")
    else:
        recent_results_df = results_df

    player_stats = recent_results_df.groupby('player').agg({
        'PTS': ['count', 'mean'], 'Error': 'mean'
    }).round(2)
    player_stats.columns = ['Game_Count', 'Avg_Points', 'MAE']
    player_stats = player_stats[player_stats['Game_Count'] >= 5].sort_values('MAE')

    print(f"\n--- RECENT PLAYER PERFORMANCE ANALYSIS (Last {recent_years} year(s)) ---")
    print(f"Overall MAE for recent players: {player_stats['MAE'].mean():.2f} ± {player_stats['MAE'].std():.2f}")

    print(f"\nTop {top_n} Best Predicted Recent Players:")
    for i, (player, row) in enumerate(player_stats.head(top_n).iterrows(), 1):
        print(f"  {i:2d}. {player:<25} MAE: {row['MAE']:.2f} (Avg: {row['Avg_Points']:.1f} pts)")

    print(f"\nTop {top_n} Worst Predicted Recent Players:")
    for i, (player, row) in enumerate(player_stats.tail(top_n).iterrows(), 1):
        print(f"  {i:2d}. {player:<25} MAE: {row['MAE']:.2f} (Avg: {row['Avg_Points']:.1f} pts)")

    return player_stats

def predict_player_vs_specific_opponent(player_name, opponent_team, model, feature_scaler, target_scaler, cleaned_df):
    print(f"Predicting points for {player_name} vs {opponent_team}...")
    
    matchups = pd.DataFrame()
    if 'away_trad' in cleaned_df.columns:
        matchups = cleaned_df[(cleaned_df['player'] == player_name) & (cleaned_df['away_trad'] == opponent_team)]
    elif 'away_adv' in cleaned_df.columns:
        matchups = cleaned_df[(cleaned_df['player'] == player_name) & (cleaned_df['away_adv'] == opponent_team)]

    if matchups.empty:
        print(f"No historical matchups found. Using recent form.")
        player_games = cleaned_df[cleaned_df['player'] == player_name]
        if player_games.empty:
            print("Player not found."); return None
        latest_game = player_games.sort_values('date').iloc[-1]
    else:
        latest_game = matchups.sort_values('date').iloc[-1]

    training_features = feature_scaler.feature_names_in_ if hasattr(feature_scaler, 'feature_names_in_') else cleaned_df.columns
    
    # Ensure all features exist
    current_features = latest_game.index
    for f in training_features:
        if f not in current_features: latest_game[f] = 0
            
    features = latest_game[training_features].values.reshape(1, -1)
    prediction = target_scaler.inverse_transform(model.predict(feature_scaler.transform(features), verbose=0))[0][0]

    print(f"\n=== PREDICTION: {player_name} vs {opponent_team} ===")
    print(f"Predicted Points: {prediction:.1f}")
    return prediction

def simple_prediction_interface(model, feature_scaler, target_scaler, cleaned_df):
    print("\n" + "="*50 + "\nNBA PLAYER vs OPPONENT PREDICTOR\n" + "="*50)
    available_players = sorted(cleaned_df['player'].unique())
    opponent_col = 'away_trad' if 'away_trad' in cleaned_df.columns else 'away_adv'
    available_teams = sorted(cleaned_df[opponent_col].unique()) if opponent_col in cleaned_df.columns else []

    while True:
        player_name = input("\nPlayer name (or 'quit'): ").strip()
        if player_name.lower() == 'quit': break
        
        if player_name not in available_players:
            matches = [p for p in available_players if player_name.lower() in p.lower()]
            if matches:
                confirm = input(f"Did you mean: {matches[0]}? (y/n): ")
                if confirm.lower() == 'y': player_name = matches[0]
                else: continue
            else: print("Player not found."); continue

        opponent_team = input("Opponent team: ").strip()
        if opponent_team not in available_teams:
            matches = [t for t in available_teams if opponent_team.lower() in t.lower()]
            if matches:
                confirm = input(f"Did you mean: {matches[0]}? (y/n): ")
                if confirm.lower() == 'y': opponent_team = matches[0]
                else: continue

        try:
            predict_player_vs_specific_opponent(player_name, opponent_team, model, feature_scaler, target_scaler, cleaned_df)
        except Exception as e:
            print(f"Error: {e}")

def main():
    game_logs_file = 'traditional.csv'
    advanced_stats_file = 'advanced.csv'
    positions_file = 'positions.csv'

    X_scaled, y_scaled, feature_scaler, target_scaler, cleaned_df = load_and_preprocess_data(
        game_logs_file, advanced_stats_file, positions_file
    )
    if X_scaled is None: print("Data loading failed."); return

    feature_names = list(feature_scaler.feature_names_in_)

    train_indices, test_indices = time_based_split(cleaned_df, test_size=0.2)
    X_train = X_scaled[cleaned_df.index.get_indexer(train_indices)]
    X_test = X_scaled[cleaned_df.index.get_indexer(test_indices)]
    y_train = y_scaled[cleaned_df.index.get_indexer(train_indices)]
    y_test = y_scaled[cleaned_df.index.get_indexer(test_indices)]
    df_test_metadata = cleaned_df.loc[test_indices, ['player', 'date', 'PTS']].reset_index(drop=True)

    model = build_advanced_dnn_model(X_train.shape[1])
    print("--- Model Summary ---")
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.0005)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=0.00001)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
## -- Model training -- ##
    print("\n--- Training Model ---")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,  
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    if os.path.exists('best_model.h5'): model.load_weights('best_model.h5')

    predictions, actuals = enhanced_evaluation(model, X_test, y_test, target_scaler, df_test_metadata)
    
    print("\n--- Creating Performance Visualizations ---")
    create_visualizations(model, history, X_test, y_test, target_scaler,
                         predictions, actuals, df_test_metadata, feature_names=feature_names)

    print("\n--- Analyzing Player-Level Performance ---")
    analyze_recent_player_performance(model, X_test, y_test, target_scaler,
                                            cleaned_df, test_indices, top_n=15, recent_years=1)
    
    simple_prediction_interface(model, feature_scaler, target_scaler, cleaned_df)

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main()
