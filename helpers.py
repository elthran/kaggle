import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from scipy.stats import stats

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, ElasticNetCV, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV, RandomizedSearchCV, cross_val_score, \
    KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor


def set_styling():
    sns.set()  # will display the plot and the axis ticks with the white background
    plt.style.use('dark_background')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def load_csv_data(filename="data.csv", id=None, dtype=None, nrows=None, usecols=None):
    df = pd.read_csv(filename, dtype=dtype, nrows=nrows, usecols=usecols)
    if id:
        df = df.set_index(keys=id)
    return df


def load_all_csv_data(train_filename="train.csv", test_filename="test.csv", id=None, dtype=None, nrows=None, parse_dates=[]):
    train = pd.read_csv(train_filename, dtype=dtype, nrows=nrows, parse_dates=parse_dates)
    test = pd.read_csv(test_filename, dtype=dtype, parse_dates=parse_dates)
    df = pd.concat([train, test], sort=False)
    if id:
        df = df.set_index(keys=id)
    return df


def save_data_as_csv(df, filename="cleaned_train.csv"):
    df.to_csv(filename)


def describe_data(df):
    print(f"\nDataframe contains {len(df)} rows and {len(df.columns)} columns")
    print(f"\nDescription:\n{df.describe()}")
    print("\nBasic Info:")  # Needs to be separate calls to run sequentially
    print(f"{df.info()}")  # Above
    print(f"\nNull counts per column:\n{df.isnull().sum()}")
    print("\nUnique value counts per column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")
    print(f"\nFirst x rows:\n{df.head(25)}")


def pair_plot_on_feature(df, target):
    train = df[df[target].notnull()]
    plotting = sns.pairplot(train, hue=target, palette="colorblind")
    plt.show()


def violin_plot_feature_target(df, target, feature):
    plotting = sns.violinplot(y=target, x=feature, data=df)
    plt.show()


def correlation_matrix(df, target):
    train = df[df[target].notnull()]
    # print(train.drop(columns=[target]).corrwith(df[target]))
    correlation = train.corr()
    sns.heatmap(correlation,
                xticklabels=correlation.columns,
                yticklabels=correlation.columns)
    plt.show()


def replace_na_with_mean(df, feature_to_replace):
    df[feature_to_replace].fillna((df[feature_to_replace].mean()), inplace=True)


def replace_na_with_mode(df, feature_to_replace):
    df[feature_to_replace].fillna(df[feature_to_replace].mode().iloc[0], inplace=True)


def replace_with_median_of_other_column(df, feature_to_replace, other_column):
    df[feature_to_replace] = df.groupby(other_column)[feature_to_replace].transform(lambda x: x.fillna(x.median()))
    return df


def convert_non_zero_to_boolean(df, columns):
    for column in columns:
        df[column] = df[column].fillna(0)
        df[f"Has{column}"] = df[column].map(lambda x: 1 if x > 0 else 0)
    df.drop(columns, inplace=True, axis=1)
    return df


def label_encoding_categories(df, columns):
    for column in columns:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes
    return df


def bin_column(df, column_to_bin, num_bins):
    """Replaces the column value with the bin index"""
    df[f"binned_{column_to_bin}"] = pd.qcut(df[column_to_bin], num_bins)
    label = LabelEncoder()
    df[f"{column_to_bin}"] = label.fit_transform(df[f"binned_{column_to_bin}"])
    df.drop([f"binned_{column_to_bin}"], axis=1, inplace=True)
    return df


def one_hot_encode(df, columns):
    df = pd.get_dummies(df, columns=columns)
    return df


def remove_overfitted_colums(df):
    """I don't fully understand how this works yet."""
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    print(f"Dropping {overfit} columns for being overfitted.")
    df.drop(overfit, inplace=True, axis=1)


def split_data(df, feature):
    train = df[df[feature].notnull()]
    test = df[~df[feature].notnull()].drop([feature], axis=1)
    train_features = train.drop([feature], axis=1)
    train_targets = train[feature]
    data_array = np.isnan(train_features).any()
    for column in data_array:
        if column is True:
            print(f"{column} contains N/A data.")
    return train_features, train_targets, test


def standardize_data(train_features, test_features, method="standard"):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Scaler undfeined")
    scaler.fit(train_features)
    train_features = pd.DataFrame(scaler.transform(train_features),
                                  index=train_features.index,
                                  columns=train_features.columns)
    test_features = pd.DataFrame(scaler.transform(test_features),
                                 index=test_features.index,
                                 columns=test_features.columns)
    # This method is cleaner but broken
    # train_features = scaler.fit_transform(train_features)
    # test_features = scaler.transform(test_features)
    return train_features, test_features


def pca_fit_data(train_features, test_features):
    "Requires that the standardize_data uses StandardScaler"
    print(train_features.shape)
    pca = PCA(0.85)
    pca.fit(train_features)
    train_features = pd.DataFrame(pca.transform(train_features),
                                  index=train_features.index)
    test_features = pd.DataFrame(pca.transform(test_features),
                                 index=test_features.index)
    print(pca.n_components_)
    print(pca.explained_variance_ratio_)
    return train_features, test_features


def rmsle(targets, target_predictions):
    return np.sqrt(mean_squared_log_error(targets, target_predictions))


def cv_rmse(model, features, targets):
    rmse = np.sqrt(-cross_val_score(model, features, targets, scoring="neg_mean_squared_error",
                                    cv=KFold(n_splits=10, shuffle=True)))
    return (rmse)


def fit_all_categorical_models(train_features, train_targets):
    model_configs = [{"name": "k_neighbors",
                      "model": KNeighborsClassifier(),
                      "params": {"n_neighbors": [i for i in range(10, 12)]}},
                     {"name": "logistic_regression",
                      "model": LogisticRegression(),
                      "params": {"max_iter": [1000, 2000]}},
                     {"name": "gaussian naive bayes",
                      "model": GaussianNB(),
                      "params": {}},
                     {"name": "linear support vector machine",
                      "model": LinearSVC(),
                      "params": {"max_iter": [2000, 4000]}},
                     {"name": "decision tree classifier",
                      "model": DecisionTreeClassifier(),
                      "params": {"min_samples_leaf": [i for i in range(5, 10)]}}
                     ]
    best_models = []
    all_scores = []
    for model_config in model_configs:
        grid_search = GridSearchCV(model_config["model"],
                                   param_grid=model_config["params"],
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(train_features, train_targets)
        print(f"Best parameters for {model_config['name']} are {grid_search.best_params_}"
              f" with score {grid_search.best_score_}")
        best_models.append(grid_search.best_estimator_)
        all_scores.append(grid_search.best_score_)
    print(f"Overall average score: {sum(all_scores) / len(all_scores)}")
    return best_models


def fit_all_regression_models(train_features, train_targets):
    model_configs = [
        {"name": "Random Forest",
         "model": RandomForestRegressor(),
         "params": {
             'bootstrap': [True],
             'criterion': ["mse"],
             'max_depth': [32],
             'max_features': ['sqrt'],
             'min_samples_leaf': [1],
             'min_samples_split': [2],
             'n_estimators': [5000]}
         },
        {"name": "Decision Tree",
         "model": DecisionTreeRegressor(),
         "params": {"criterion": ["mae"],
                    "splitter": ["best"],
                    "max_depth": [None],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "max_features": ["auto"]},
         "notes": "Not good with continuous variables."
         },
        {"name": "Gradient Regressor",
         "model": GradientBoostingRegressor(),
         "params": {"n_estimators": [1000],
                    "criterion": ['mse'],
                    "learning_rate": [0.01],
                    "max_depth": [5],
                    "max_features": ['sqrt'],
                    "min_samples_leaf": [2],
                    "min_samples_split": [2]}
         },
        {"name": "XGBR Regressor",
         "model": XGBRegressor(),
         "params": {"n_estimators": [10000],
                    "learning_rate": [0.01],
                    "max_depth": [3],
                    "min_child_weight": [1]}
         }
    ]
    best_models = []
    all_scores = []
    for model_config in model_configs:
        grid_search = GridSearchCV(estimator=model_config["model"],
                                   param_grid=model_config["params"],
                                   n_jobs=-1,
                                   verbose=1,
                                   refit=True)
        grid_search.fit(train_features, train_targets)
        print(f"Best parameters for {model_config['name']} are {grid_search.best_params_}"
              f" with score {round(100 * grid_search.best_score_, 2)}%")
        best_models.append(grid_search.best_estimator_)
        all_scores.append(grid_search.best_score_)

        print("Root Mean Squared Logarithmic Error:",
              rmsle(train_targets, grid_search.best_estimator_.predict(train_features)))
        # test_score = cv_rmse(grid_search.best_estimator_, train_features, train_targets)
        # print(f"{model_config['name']}: {test_score.mean()} {test_score.std()}")

    print(f"\nOverall average score: {round(100 * sum(all_scores) / len(all_scores), 2)}%")

    # Remove the worst model from the list
    # del best_models[all_scores.index(min(all_scores))]
    # del all_scores[all_scores.index(min(all_scores))]

    # Only keep the best model
    best_models = best_models[all_scores.index(max(all_scores)):all_scores.index(max(all_scores)) + 1]
    all_scores = all_scores[all_scores.index(max(all_scores)):all_scores.index(max(all_scores)) + 1]

    print(f"\nAverage score of selected models: {round(100 * sum(all_scores) / len(all_scores), 2)}%")

    return best_models


def average_predictions(models, test_features, np_value="int", method="mode"):
    predictions = np.column_stack([model.predict(test_features) for model in models])
    print(f"Predictions of first 5 rows: {predictions[:5]}")
    if method == "mode":
        averaged_predictions = stats.mode(predictions, axis=1)[0].astype(np_value)
    elif method == "mean":
        averaged_predictions = np.average(predictions, axis=1).astype(np_value)
    else:
        raise Exception("Method undefined")
    print(f"Averaged predictions of first 5 rows: {averaged_predictions[:5]}")
    return np.hstack(averaged_predictions)


def output_prediction(id_name, id_column, target_name, predictions):
    output = pd.DataFrame({id_name: id_column, target_name: predictions})
    output.to_csv('submission.csv', index=False)
    output.head()
