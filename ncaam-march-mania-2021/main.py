# Imports
import warnings
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

warnings.filterwarnings('ignore')

# link: https://www.kaggle.com/readoc/ncaam-prediction-stage-2

kenpom = pd.read_csv('input/Mkenpom2021.xls')

kenpom.Team = kenpom.Team.str.lower()
teams = pd.read_csv('input/MDataFiles_Stage2/MTeamSpellings.csv', encoding='cp1252')
kenpom = pd.merge(kenpom, teams, left_on=['Team'], right_on=['TeamNameSpelling'], how='left')
kenpom = kenpom.drop(columns=['TeamNameSpelling', 'Team'])

numerical_cols = ['AdjustO', 'AdjustD', 'AdjustT', 'Luck']
kenpom = kenpom[['TeamID', 'Year'] + numerical_cols].dropna(subset=['TeamID', 'Year'])
kenpom.TeamID = kenpom.TeamID.astype(int)
kenpom = kenpom.rename(columns={'Year': 'Season'})


def get_moving_averages(df, team_col, target_col):
    totals = defaultdict(int)
    weight_sums = defaultdict(int)
    df_ = df.set_index(['Season', team_col])
    found = 0
    not_found = 0
    year_weights = {2019: 128, 2018: 64, 2017: 32, 2016: 16, 2015: 8, 2014: 4, 2013: 2, 2012: 1, 2011: 0.5, 2010: 0.25}
    for team in df[team_col].unique():
        for year in [2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]:
            wt = year_weights[year]

            df_year = df[df.Season == year]
            if (year, team) in df_.index:
                found += 1
                weight_sums[team] += wt
                totals[team] += wt * df_.loc[year, team][target_col]
            else:
                not_found += 1
        try:
            totals[team] = round(totals[team] / weight_sums[team], 3)
        except:
            continue
    print('found for: ', found * 100 / (found + not_found))
    return totals


f38 = pd.read_csv('input/538ratingsMen.csv')
f38.TeamID = f38.TeamID.astype(int)
f38 = f38[['Season', 'TeamID', '538rating']]

team_to_f38_rating = get_moving_averages(f38, 'TeamID', '538rating')


def get_f38(team_id):
    return team_to_f38_rating[team_id] if team_id in team_to_f38_rating else 0


kenpom['f38'] = kenpom.TeamID.apply(get_f38)

season_df = pd.read_csv('input/MDataFiles_Stage2/MRegularSeasonCompactResults.csv')
tourney_df = pd.read_csv('input/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')
ordinals_df = pd.read_csv('input/MDataFiles_Stage2/MMasseyOrdinals.csv').rename(columns={'RankingDayNum': 'DayNum'})

# Get the last available data from each system previous to the tournament
ordinals_df = ordinals_df.groupby(['SystemName', 'Season', 'TeamID']).last().reset_index()
ordinals_df['Rating'] = 100 - 4 * np.log(ordinals_df['OrdinalRank'] + 1) - ordinals_df['OrdinalRank'] / 22
ref_system = 'POM'
ordinals_df = ordinals_df[ordinals_df.SystemName == ref_system]
massey_df = ordinals_df.set_index(['Season', 'TeamID'])['Rating']

seeds = pd.read_csv('input/MDataFiles_Stage2/MNCAATourneySeeds.csv')
seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds = seeds.set_index(['Season', 'TeamID'])
seeds = seeds.drop('Seed', 1)


def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT',
                 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
                 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'

    df.columns = [x.replace('W', 'T1_').replace('L', 'T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L', 'T1_').replace('W', 'T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location == 'N', 'location'] = '0'
    output.loc[output.location == 'H', 'location'] = '1'
    output.loc[output.location == 'A', 'location'] = '-1'
    output.location = output.location.astype(int)

    output['PointDiff'] = output['T1_Score'] - output['T2_Score']

    return output


regular_results = pd.read_csv("input/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv")
tourney_results = pd.read_csv('input/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')
regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)
boxscore_cols = ['T1_FGM', 'T1_Stl', 'T2_FGM', 'T2_Stl']
season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].mean().reset_index()
season_statistics_T1 = season_statistics.copy()
season_statistics_T2 = season_statistics.copy()

season_statistics_T1.columns = ["T1_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in
                                list(season_statistics_T1.columns)]
season_statistics_T2.columns = ["T2_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in
                                list(season_statistics_T2.columns)]
season_statistics_T1.columns.values[0] = "Season"
season_statistics_T2.columns.values[0] = "Season"

team_to_fgm = get_moving_averages(season_statistics_T1, 'T1_TeamID', 'T1_FGM')
team_to_stl = get_moving_averages(season_statistics_T1, 'T1_TeamID', 'T1_Stl')


def get_stl(team_id):
    return team_to_stl[team_id] if team_id in team_to_stl else 0


def get_fgm(team_id):
    return team_to_fgm[team_id] if team_id in team_to_fgm else 0


kenpom['STL'] = kenpom.TeamID.apply(get_stl)
kenpom['FGM'] = kenpom.TeamID.apply(get_fgm)
kenpom = kenpom.set_index(['Season', 'TeamID'])

kenpom = kenpom.drop('f38', 1)  # f38 has a lot of missing data


def get_kenpom_data(seasons, team_set1, team_set2):
    found_in_kenpom = 0
    found_in_massey = 0
    found_in_seeds = 0
    all_features = []
    for season, team1, team2 in tqdm(zip(seasons, team_set1, team_set2), total=len(team_set1)):

        team1_kenpom_features = kenpom.mean().values
        team2_kenpom_features = kenpom.mean().values
        if (season, team1) in kenpom.index:
            team1_kenpom_features = kenpom.loc[season, team1].values
            found_in_kenpom += 1
        if (season, team2) in kenpom.index:
            team2_kenpom_features = kenpom.loc[season, team2].values
            found_in_kenpom += 1

        team1_massey = team2_massey = 73
        if (season, team1) in massey_df.index:
            team1_massey = massey_df.loc[season, team1]
            found_in_massey += 1
        if (season, team2) in massey_df.index:
            team2_massey = massey_df.loc[season, team2]
            found_in_massey += 1

        team1_seed = team2_seed = 10
        if (season, team1) in seeds.index:
            team1_seed = seeds.loc[season, team1]
            found_in_seeds += 1
        if (season, team2) in seeds.index:
            team2_seed = seeds.loc[season, team2]
            found_in_seeds += 1

        team1_features = np.append(team1_kenpom_features, team1_massey)
        team1_features = np.append(team1_features, team1_seed)
        team2_features = np.append(team2_kenpom_features, team2_massey)
        team2_features = np.append(team2_features, team2_seed)

        features = np.concatenate((team2_features, team1_features))
        massey_pred = 1. / (1e-6 + 10 ** ((team1_massey - team2_massey) / 15))
        features = np.append(features, massey_pred)
        all_features.append(features)

    return {
        'X': np.array(all_features),
        'kenpom_found': found_in_kenpom / (2 * len(team_set1) + 1e-6),
        'massey_found': found_in_massey / (2 * len(team_set1) + 1e-6),
        'seeds_found': found_in_seeds / (2 * len(team_set1) + 1e-6)
    }


FAST_RUN = False

DEBUG = False

all_seasons = pd.read_csv('input/MDataFiles_Stage2/MRegularSeasonCompactResults.csv')
lgbm_parameters = {
    'objective': 'binary',
    'metric': 'binary_logloss',
}

from_year = 2007
if FAST_RUN:
    from_year = 2012
all_seasons = all_seasons[all_seasons.Season > from_year]
test = pd.read_csv('input/MDataFiles_Stage2/MSampleSubmissionStage2.csv')
test['Season'] = test['ID'].apply(lambda x: int(x.split('_')[0]))
test['TeamID1'] = test['ID'].apply(lambda x: int(x.split('_')[1]))
test['TeamID2'] = test['ID'].apply(lambda x: int(x.split('_')[2]))
test = test.drop(['Pred', 'ID'], axis=1)

test_pred = np.zeros(len(test))
# test_pred = []
test_pred_mse = []

SPLITS = 5
if FAST_RUN:
    SPLITS = 3

kf = KFold(n_splits=SPLITS, shuffle=True)
y_preds = []
for year in test['Season'].unique():
    if FAST_RUN and year > 2015:
        break
    season_df = all_seasons[all_seasons.Season < year]  # training on past data
    team1_wins = np.random.randint(0, 2, len(season_df))
    winner_teams = season_df.WTeamID.values
    loser_teams = season_df.LTeamID.values
    team_set1 = np.where(team1_wins == 1, winner_teams, loser_teams)
    team_set2 = np.where(team1_wins == 0, winner_teams, loser_teams)
    seasons = season_df.Season.values
    res = get_kenpom_data(seasons, team_set1, team_set2)
    X_year = res['X']
    y_year = team1_wins

    score_diff = (season_df.WScore.values - season_df.LScore.values) / (
            season_df.WScore.values + season_df.LScore.values)
    y_year_mse = np.where(team1_wins == 1, score_diff, -score_diff)

    print('Kenpom data found for : ', res['kenpom_found'] * 100)
    print('Massey data found for: ', res['massey_found'] * 100)
    print('Seeds found for: ', res['seeds_found'] * 100)

    test_year = test[test['Season'] == year]

    lgbm_val_pred = np.zeros(len(y_year))
    lgbm_val_pred_mse = np.zeros(len(y_year))
    lgbm_test_pred = np.zeros(len(test_year))
    lgbm_test_pred_mse = np.zeros(len(test_year))
    logloss = []
    losses_mse = []

    for trn_idx, val_idx in kf.split(X_year, y_year):
        x_train_idx = X_year[trn_idx]
        x_valid_idx = X_year[val_idx]

        y_valid_idx = y_year[val_idx]
        y_train_idx = y_year[trn_idx]
        y_valid_idx_mse = y_year_mse[val_idx]
        y_train_idx_mse = y_year_mse[trn_idx]

        lgbm_model = LGBMRegressor(**lgbm_parameters)
        lgbm_model_mse = LGBMRegressor(metric='mse')

        lgbm_model.fit(x_train_idx, y_train_idx, eval_set=((x_valid_idx, y_valid_idx)), verbose=False,
                       early_stopping_rounds=100)
        lgbm_model_mse.fit(x_train_idx, y_train_idx_mse, eval_set=((x_valid_idx, y_valid_idx_mse)), verbose=False,
                           early_stopping_rounds=100)

        seasons, team_set1, team_set2 = test_year.values.transpose()
        X_test = get_kenpom_data(seasons, team_set1, team_set2)['X']

        lgbm_test_pred += lgbm_model.predict(X_test) / SPLITS
        lgbm_test_pred_mse = lgbm_model_mse.predict(X_test) / SPLITS

        y_pred = lgbm_model.predict(x_valid_idx)
        y_preds.append(y_pred)

        y_pred_mse = lgbm_model_mse.predict(x_valid_idx)
        y_pred_mse = np.where(y_pred_mse > 0, 1, 0)
        y_pred = (19 * y_pred + y_pred_mse) / 20
        logloss.append(log_loss(y_valid_idx, y_pred))
        losses_mse.append(mean_squared_error(y_valid_idx_mse, y_pred_mse))

    test_pred += lgbm_test_pred.tolist()
    test_pred_mse += lgbm_test_pred_mse.tolist()
    print('Year_Predict:', year, 'Log_Loss:', np.mean(logloss))

model1_preds = test_pred

# Read Data
tourney_result = pd.read_csv('input/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')
tourney_seed = pd.read_csv('input/MDataFiles_Stage2/MNCAATourneySeeds.csv')
season_result = pd.read_csv('input/MDataFiles_Stage2/MRegularSeasonCompactResults.csv')
test_df = pd.read_csv('input/MDataFiles_Stage2/MSampleSubmissionStage2.csv')
submission_df = pd.read_csv('input/MDataFiles_Stage2/MSampleSubmissionStage2.csv')

# deleting unnecessary columns
tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
# Merge Seed
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'],
                          how='left')
tourney_result.rename(columns={'Seed': 'WSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'],
                          how='left')
tourney_result.rename(columns={'Seed': 'LSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)


def get_seed(x):
    return int(x[1:3])


tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))
tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))
# Merge Score
season_win_result = season_result[['Season', 'WTeamID', 'WScore']]
season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]
season_win_result.rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score'}, inplace=True)
season_lose_result.rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score'}, inplace=True)
season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)
season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'],
                          how='left')
tourney_result.rename(columns={'Score': 'WScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'],
                          how='left')
tourney_result.rename(columns={'Score': 'LScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)
tourney_win_result.rename(columns={'WSeed': 'Seed1', 'LSeed': 'Seed2', 'WScoreT': 'ScoreT1', 'LScoreT': 'ScoreT2'},
                          inplace=True)
tourney_lose_result = tourney_win_result.copy()
tourney_lose_result['Seed1'] = tourney_win_result['Seed2']
tourney_lose_result['Seed2'] = tourney_win_result['Seed1']
tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']
tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']
tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']
tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']
tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0
tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)
train_df = tourney_result
# Get Test
test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed': 'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed': 'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score': 'ScoreT1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score': 'ScoreT2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

test_df['result'] = np.NaN


class Base_Model(object):

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=False):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = 'result'
        self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        self.y_pred, self.model = self.fit()

    def train_model(self, train_set, val_set):
        raise NotImplementedError

    def get_cv(self):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return cv.split(self.train_df, self.train_df[self.target])

    def get_params(self):
        raise NotImplementedError

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def convert_x(self, x):
        return x

    def fit(self):
        oof_pred = np.zeros((len(train_df),))
        y_pred = np.zeros((len(test_df),))
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            print('Fold:', fold + 1)
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model = self.train_model(train_set, val_set)

            conv_x_val = self.convert_x(x_val)
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)

            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
        return y_pred, model


class Lgb_Model(Base_Model):

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, train_set, 10000, valid_sets=[train_set, val_set])

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set

    def get_params(self):
        params = {'num_leaves': 400,
                  'min_child_weight': 0.034,
                  'feature_fraction': 0.379,
                  'bagging_fraction': 0.418,
                  'min_data_in_leaf': 106,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.0068,
                  "boosting_type": "gbdt",
                  "bagging_seed": 11,
                  "metric": 'logloss',
                  'reg_alpha': 0.3899,
                  'reg_lambda': 0.648,
                  'random_state': 47,
                  }
        return params


class Xgb_Model(Base_Model):

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return xgb.train(self.params, train_set,
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')],
                         verbose_eval=verbosity, early_stopping_rounds=100)

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set

    def convert_x(self, x):
        return xgb.DMatrix(x)

    def get_params(self):
        params = {'colsample_bytree': 0.8,
                  'learning_rate': 0.01,
                  'max_depth': 3,
                  'subsample': 1,
                  'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'min_child_weight': 3,
                  'gamma': 0.25,
                  'n_estimators': 5000}
        return params


class Catb_Model(Base_Model):

    def train_model(self, train_df, test_df):
        verbosity = 100 if self.verbose else 0
        clf = CatBoostClassifier(**self.params)
        clf.fit(train_df['X'],
                train_df['y'],
                eval_set=(test_df['X'], test_df['y']),
                verbose=verbosity,
                cat_features=self.categoricals)
        return clf

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def get_params(self):
        params = {'loss_function': 'Logloss',
                  'task_type': "CPU",
                  'iterations': 5000,
                  'od_type': "Iter",
                  'depth': 3,
                  'colsample_bylevel': 0.5,
                  'early_stopping_rounds': 300,
                  'l2_leaf_reg': 18,
                  'random_seed': 42,
                  'use_best_model': True
                  }
        return params


features = train_df.columns
features = [x for x in features if x not in ['result']]
print(features)
categoricals = []

# cat_model = Catb_Model(train_df, test_df, features, categoricals=categoricals)
lgb_model = Lgb_Model(train_df, test_df, features, categoricals=categoricals)
xgb_model = Xgb_Model(train_df, test_df, features, categoricals=categoricals)

weights = {
    'model_1': 2,
    'lgb_model': 24,
    'xgb_model': 8
}

X1, X2, X3 = np.array(model1_preds), lgb_model.y_pred, xgb_model.y_pred
W1, W2, W3 = weights['model_1'], weights['lgb_model'], weights['xgb_model']
final_preds = (W1 * X1 + W2 * X2 + W3 * X3) / (W1 + W2 + W3)

submission = pd.read_csv('input/MDataFiles_Stage2/MSampleSubmissionStage2.csv')
submission.Pred = final_preds
submission.to_csv('output/submission.csv', index=False)
