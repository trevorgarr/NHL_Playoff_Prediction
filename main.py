"""
Trevor Garrood and Evan Bodenstab
CSE 163
Using NHL Regular Season Result Data to Predict Playoff Outcomes
This program runs our data anaylsis on NHL data. It takes the
calculated data from data_process.py and outputs plots and values
that are used to answer out data questions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_process import prod_init_df
from data_process import get_processed_data


def pythag_expectation(df4):
    """
    Creates a plot of number of wins we expects vs number of actual wins
    as well as returns the r squared and rms error for the correlation
    """
    plt.figure(1)
    df4 = df4.dropna()
    sns.scatterplot(x='Total Season Wins', y='Expected Season Wins',
                    data=df4, alpha=0.5, palette="muted")
    plt.xlabel('Total Season Wins')
    plt.ylabel('Expected Season Wins')
    plt.title('Actual vs. Expected Total Season Wins')
    r_squared = get_r_squared(df4['Total Season Wins'],
                              df4['Expected Season Wins'])
    rms_error = get_rms_error(df4['Total Season Wins'],
                              df4['Expected Season Wins'])
    text_str = \
        'R_Squared =' + str(r_squared) + '\n' + ' RMS = ' + str(rms_error)
    plt.text(35, 50, text_str, {'color': 'k', 'fontsize': 11},
             va="top", ha="right")
    plt.savefig('pythag_expect.png', bbox_inches='tight')


def length_of_season(df, df1):
    """
    Creates a plot of total number of wins vs number of total games in
    the previous year as well as returns the r squared and rms error
    for the correlation
    """
    plt.figure(2)
    df5 = get_processed_data(20132014, df, df1).dropna()
    df6 = get_processed_data(20142015, df, df1).dropna()
    plt.scatter(df6['Total Season Wins'], df5['Total Games'])
    plt.ylim(80, 110)
    plt.xlim(20, 55)
    plt.xlabel('Total Season Wins')
    plt.ylabel('Length of Previous Season')
    plt.title('Total wins vs. Length of Previous Season')
    r_squared = get_r_squared(df6['Total Season Wins'],
                              df5['Total Games'])
    rms_error = get_rms_error(df6['Total Season Wins'],
                              df5['Total Games'])
    text_str = \
        'R_Squared =' + str(r_squared) + '\n' + ' RMS = ' + str(rms_error)
    plt.text(35, 105, text_str, {'color': 'k', 'fontsize': 11},
             va="top", ha="right")
    plt.savefig('length_of_season.png', bbox_inches='tight')


def winstreaks(df4):
    """
    Creates a plot of total number of winstreaks vs total number of wins
    as well as returns the r squared and rms error for the correlation
    """
    df4 = df4.dropna()
    plt.figure(3)
    sns.scatterplot(x='Total Season Wins', y='Winstreaks',
                    data=df4, alpha=0.5, palette="muted")
    plt.ylim(12, 26)
    plt.xlim(18, 64)
    plt.xlabel('Total Season Wins')
    plt.ylabel('Winstreaks')
    plt.title('Total Season Wins vs. Number of Winstreaks')
    r_squared = get_r_squared(df4['Total Season Wins'], df4['Winstreaks'])
    rms_error = get_rms_error(df4['Total Season Wins'], df4['Winstreaks'])
    text_str = \
        'R_Squared =' + str(r_squared) + '\n' + ' RMS = ' + str(rms_error)
    plt.text(64, 24, text_str, {'color': 'k', 'fontsize': 11},
             va="top", ha="right")
    plt.savefig('winstreaks.png', bbox_inches='tight')


def long_winstreaks(df4):
    """
    Creates a plot of total number of winstreaks vs total number of wins
    as well as returns the r squared and rms error for the correlation
    """
    df4 = df4.dropna()
    plt.figure(4)
    sns.scatterplot(x='Total Season Wins', y='Longest Win Streak',
                    data=df4, alpha=0.5, palette="muted")
    plt.xlabel('Total Season Wins')
    plt.ylabel('Longest Winstreak')
    plt.title('Total Season Wins vs. Longest Winstreak')
    r_squared = get_r_squared(df4['Total Season Wins'],
                              df4['Longest Win Streak'])
    rms_error = get_rms_error(df4['Total Season Wins'],
                              df4['Longest Win Streak'])
    text_str = \
        'R_Squared =' + str(r_squared) + '\n' + ' RMS = ' + str(rms_error)
    plt.text(64, 16, text_str, {'color': 'k', 'fontsize': 11},
             va="top", ha="right")
    plt.savefig('long_winstreaks.png', bbox_inches='tight')


def get_r_squared(column_one, column_two):
    """
    Calculates the r squared value of two columns where r squared is the
    proportion of the variance in the dependent variable that is
    predictable from the independent variable
    """
    correlation_matrix = np.corrcoef(column_one, column_two)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2
    return round(r_squared, 2)


def get_rms_error(column_one, column_two):
    """
    Calculates the rms error of two columns where rms is the root mean
    squared difference
    """
    rms_error = math.sqrt(mean_squared_error(column_one, column_two))
    return round(rms_error, 2)


def predict_playoffs(df4, year, fig_num):
    """
    Predicts how many playoff games each team in the NHL will win for a
    given year and then plots the data in a heatmap for easier viewing
    also takes in the figure number to seperate the plots for different
    years
    """
    df = df4.copy().dropna()
    df1 = df4.copy().dropna()
    df = df.loc[:, df.columns != 'Playoff Round']
    df1 = df1.loc[:, df1.columns != 'Playoff Round']
    df = df[df['Season'] != year]
    features_train = df.loc[:, df.columns != 'Total Playoff Wins']
    labels_train = df['Total Playoff Wins']
    df1 = df1[df1['Season'] == year]
    features_test = df1.loc[:, df1.columns != 'Total Playoff Wins']
    labels_test = df1['Total Playoff Wins']
    model = DecisionTreeClassifier(max_depth=7)
    model.fit(features_train, labels_train)
    model.predict(features_train)
    test_predictions = model.predict(features_test)
    test_accuracy = round(accuracy_score(labels_test, test_predictions), 2)
    df_out = pd.DataFrame(labels_test)
    df_out['Predicted'] = test_predictions
    df_out = df_out.loc[~(df_out == 0).all(axis=1)]
    plt.figure(fig_num, facecolor='w', edgecolor='k')
    sns.heatmap(df_out, annot=True, cmap='viridis', cbar=False)
    text_str = \
        'Test Accuracy: ' + str(test_accuracy)
    plt.text(-.2, 0, text_str, {'color': 'k', 'fontsize': 11},
             va="top", ha="right")
    plt.savefig('playoff_prediction_' + str(year) + '.png',
                bbox_inches='tight')


def main():
    seasons = [20112012, 20132014, 20142015, 20152016, 20162017, 20172018,
               20182019]
    df = pd.read_csv('game_teams_stats.csv')
    df1 = pd.read_csv('team_info.csv')
    df2 = pd.read_csv('game.csv')
    df = prod_init_df(df, df1, df2)
    df4 = get_processed_data(seasons, df, df1)
    pythag_expectation(df4)
    length_of_season(df, df1)
    winstreaks(df4)
    long_winstreaks(df4)
    predict_playoffs(df4, 20182019, 5)
    predict_playoffs(df4, 20172018, 6)
    predict_playoffs(df4, 20162017, 7)


if __name__ == '__main__':
    main()
