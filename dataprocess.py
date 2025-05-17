"""
Trevor Garrood and Evan Bodenstab
CSE 163
Using NHL Regular Season Result Data to Predict Playoff Outcomes
This program takes the raw data and calculates various different
stats that we will use in our analysis
"""
import pandas as pd


def prod_init_df(df, df1, df2):
    """
    produces the inital dataset we will use for our analysis
    by combining the three datasets we determined help provide
    the richest analysis
    """
    df = df.merge(df1, left_on='team_id', right_on='team_id')
    df = df.merge(df2, left_on='game_id', right_on='game_id')
    return df


def season_data_generator(season, team_abbv, df):
    """
    Takes in season year, team abbreviation, and the df
    returns a dictionary with various data calculations
    that will be used in our analysis
    """
    season_data = {}
    df_ploffs = df.copy()
    df = df[df['type'] == 'R']
    is_year = df['season'] == season
    is_team = df['abbreviation'] == team_abbv
    df_s = df[is_year & is_team]
    season_data['Season'] = season
    season_data['Team'] = team_abbv
    season_data['GF'] = df_s['goals'].sum()
    season_data['GA'] = (df_s['home_goals'].sum() +
                         df_s['away_goals'].sum()) \
        - season_data['GF']
    season_data['Total Season Wins'] = df_s.won.sum()
    highest_streak, num_streaks = get_win_streaks(df_s)
    season_data['Winstreaks'] = num_streaks
    season_data['Longest Win Streak'] = highest_streak
    season_data['Faceoff Percentage'] = \
        round(df_s['faceOffWinPercentage'].mean(), 2)
    season_data['PIM'] = df_s['pim'].sum()
    season_data['Hits'] = df_s['hits'].sum()
    # This 'k' value was determined by NHL's data science team
    k = 2.15
    # Win ratio is determined by pythagorean expectation (gf / gf + ga)
    season_data['Win Ratio'] = round((season_data['GF'] ** k)
                                     / ((season_data['GA'] ** k)
                                     + (season_data['GF'] ** k)), 2)
    games_in_season = 82
    season_data['Expected Season Wins'] = \
        round(season_data['Win Ratio'] * games_in_season, 2)
    df_ploffs = df_ploffs[df_ploffs['type'] == 'P']
    is_year = df_ploffs['season'] == season
    is_team = df_ploffs['abbreviation'] == team_abbv
    df_ploffs_s = df_ploffs[is_year & is_team]
    games_in_ploffs = 16
    season_data['Total Playoff Wins'] = df_ploffs_s.won.sum()
    season_data['Expected Playoff Wins'] = \
        round(season_data['Win Ratio'] * games_in_ploffs, 2)
    season_data['Playoff Round'] = \
        int(round(season_data['Total Playoff Wins'] / 4))
    season_data['Total Games'] = df_s['won'].count() + \
        df_ploffs_s['won'].count()
    return season_data


def get_win_streaks(df):
    """
    Takes in a dataframe and returns the number of winstreaks
    and the length of the longest win streak
    """
    streaks = \
        df['won'].groupby(df['won'].ne(df['won'].shift()).cumsum()).cumcount()
    highest_streak = streaks.max()
    num_streaks = len(streaks[streaks == 1])
    return highest_streak, num_streaks


def get_processed_data(seasons, df, df1):
    """
    Where seasons is all seasons we have data for, df is the
    dataframe of all goals, wins, etc... and df1 is a dataframe
    of teams, we calculate the season stats for every team and ever
    season (except 20122013 which was the NHL lockout) and put it into
    one csv file with the team as the index
    """
    team_data = []
    if type(seasons) is list:
        for season in seasons:
            for team in df1['abbreviation']:
                team_data.append(season_data_generator(season, team, df))
        df4 = pd.DataFrame(team_data).set_index('Team')
    elif type(seasons) is int:
        for team in df1['abbreviation']:
            team_data.append(season_data_generator(seasons, team, df))
    df4 = pd.DataFrame(team_data).set_index('Team')
    return df4
