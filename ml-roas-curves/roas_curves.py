import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date
from sklearn.ensemble import RandomForestRegressor

from redshift_utilities import RedshiftHelper
from math_functions import MathHelper
from utilities import Utilities

# Set pandas options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set matplotlib options
plt.interactive(False)
plt.close('all')

game_datemapper = {
    "streetrace": "2021-03-25",
    "peggyscove": "2020-09-03",
    "big_hit": "2019-01-13",
    "hydrostone": "2020-05-13",
    "idle_golf": "2018-12-03",
}


class RoasCurve:

    def __init__(self, pull_data=False, train_model=False, training_end_date=None):

        self.math_helper = MathHelper()
        self.utilities = Utilities()
        self.training_end_date = training_end_date if training_end_date else date.today()

        if train_model:
            self.model = None
            self.build_model(pull_data)
        else:
            self.model = joblib.load("trained_model.sav")

        self.predict_curve()

    def build_model(self, pull_data):
        full_df = pd.DataFrame()
        for game, start_date in game_datemapper.items():
            if pull_data or not os.path.isfile(f"{game}_df.feather"):
                df = self.get_roas_df(game, start_date).reset_index()
                df.to_feather(f"{game}_df.feather")
            else:
                df = pd.read_feather(f"{game}_df.feather")
            full_df = full_df.append(df, ignore_index=True)
        full_df.set_index('index', inplace=True)
        self.train_model(full_df)

    def get_roas_df(self, game, start_date):
        redshift_helper = RedshiftHelper(game, start_date, self.training_end_date)
        redshift_helper.set_redshift_connection()
        user_summary_df = redshift_helper.get_user_summary_df()
        spend_df = redshift_helper.get_spend_df()
        revenue_df = redshift_helper.get_revenue_df()
        redshift_helper.redshift_engine.close()

        # Merge revenue with user summary
        df = revenue_df.merge(user_summary_df, how='outer', on=['teamid'])

        df["day_revenue"] = df["day_iap_revenue"] + df["day_tapjoy_revenue"] + df["day_ad_revenue"]
        df.drop(["day_iap_revenue", "day_tapjoy_revenue", "day_ad_revenue"], axis=1, inplace=True)

        # Get number of installs per date and ua type
        install_info_df = df.assign(
            organic_installs=np.where(df['ua_type'] == 'Organic', df.teamid, 0),
            ua_installs=np.where(df['ua_type'] == 'UA', df.teamid, 0),
            android_count=np.where(df['is_android'] == 1, df.teamid, 0),
            us_count=np.where(df['is_us'] == 1, df.teamid, 0),
            row_count=np.where(df['is_row'] == 1, df.teamid, 0),
            users_retained_day1=np.where(df['retention1'] == 1, df.teamid, 0),
            users_retained_day7=np.where(df['retention7'] == 1, df.teamid, 0),
        ).groupby('install_date').agg({'organic_installs': "nunique",
                                       'ua_installs': "nunique",
                                       'android_count': "nunique",
                                       'us_count': "nunique",
                                       'row_count': "nunique",
                                       'users_retained_day1': "nunique",
                                       'users_retained_day7': "nunique"}).reset_index()

        revenue_info_df = df.assign(
            day_organic_revenue=np.where(df['ua_type'] == 'Organic', df.day_revenue, 0),
            day_ua_revenue=np.where(df['ua_type'] == 'UA', df.day_revenue, 0)
        ).groupby(['install_date', 'clean_date']).agg({'day_ua_revenue': "sum",
                                                       'day_organic_revenue': "sum"}).reset_index()

        df = revenue_info_df.merge(install_info_df, how='inner', on=["install_date"])

        # Merge with spend to get CPI column
        df = df.merge(spend_df, how="outer", on=["install_date"])
        df["days_since_install"] = (df["clean_date"] - df["install_date"]).dt.days

        # Aggregate data per install date and days since install date
        df = df.groupby(["install_date", "days_since_install"]) \
            .agg(ua_installs=pd.NamedAgg(column="ua_installs", aggfunc=np.mean),
                 organic_installs=pd.NamedAgg(column="organic_installs", aggfunc=np.mean),
                 users_retained_day1=pd.NamedAgg(column="users_retained_day1", aggfunc=np.mean),
                 users_retained_day7=pd.NamedAgg(column="users_retained_day7", aggfunc=np.mean),
                 android_count=pd.NamedAgg(column="android_count", aggfunc="sum"),
                 us_count=pd.NamedAgg(column="us_count", aggfunc="sum"),
                 row_count=pd.NamedAgg(column="row_count", aggfunc="sum"),
                 spend=pd.NamedAgg(column="spend", aggfunc=np.mean),
                 day_ua_revenue=pd.NamedAgg(column="day_ua_revenue", aggfunc="sum"),
                 day_organic_revenue=pd.NamedAgg(column="day_organic_revenue", aggfunc="sum")).reset_index()

        df = df.fillna({"day_ua_revenue": 0, "day_organic_revenue": 0, "spend": 0, "retention1": 1})
        df['spend'] = df['spend'].apply(np.int64)

        df = df[(df["ua_installs"] > 25) & (df["spend"] > 25)]
        df["all_revenue"] = df["day_ua_revenue"] + df["day_organic_revenue"]
        df = df.sort_values(['install_date', 'days_since_install'], ascending=[True, True])
        df["running_revenue_total"] = df.groupby('install_date')['all_revenue'].cumsum()
        df["roas"] = df["running_revenue_total"] / df["spend"]
        df['roas'] = df.groupby(['install_date'])['roas'].ffill()

        # Remove outliers for day_0 and day_7
        for day in [0, 7]:
            ten_percentile = df[df["days_since_install"] == day].roas.quantile(0.1)
            ninety_percentile = df[df["days_since_install"] == day].roas.quantile(0.9)
            for install_date in df.install_date.unique():
                on_day_df = df[(df["install_date"] == install_date) & (df["days_since_install"] == day)]
                if on_day_df["roas"].max() < ten_percentile or on_day_df["roas"].max() > ninety_percentile:
                    df = df[df["install_date"] != install_date]

        df["retention1"] = (df["users_retained_day1"] / (df["ua_installs"] + df["organic_installs"])).astype(int)
        df["retention7"] = (df["users_retained_day7"] / (df["ua_installs"] + df["organic_installs"])).astype(int)

        all_game_names = [game for game, start_date in game_datemapper.items()]
        for game_ in all_game_names:
            if game_ == game:
                df[f'is_{game_}'] = 1
            else:
                df[f'is_{game_}'] = 0

        plt.scatter(x=df["days_since_install"],
                    y=df["roas"],
                    c=df["install_date"])
        plt.colorbar(label="Install Date", orientation="vertical")
        plt.title(f'Training data for {game} with install date heatmap')
        plt.show()

        df.drop(["install_date",
                 "users_retained_day1",
                 "users_retained_day7",
                 "running_revenue_total",
                 "day_ua_revenue",
                 "day_organic_revenue",
                 "all_revenue"], axis=1, inplace=True)

        return df

    def train_model(self, df):
        train = df.drop(["roas"], axis=1)
        y = df["roas"]
        print("Train looks like:\n\n", train.sample(5))
        if self.model is None:
            self.model = RandomForestRegressor(warm_start=True,
                                               n_estimators=10,
                                               max_features=2)
        self.model.fit(train, y)
        joblib.dump(self.model, "trained_model.sav")

    def predict_curve(self):
        game = 'hydrostone'
        predict_end_date = '2021-07-15'  # str(date.today())
        redshift_helper_2 = RedshiftHelper(game, self.training_end_date, predict_end_date)
        redshift_helper_2.set_redshift_connection()
        user_summary_df = redshift_helper_2.get_user_summary_df()
        spend_df = redshift_helper_2.get_spend_df()
        revenue_df = redshift_helper_2.get_revenue_df()
        redshift_helper_2.redshift_engine.close()

        # Merge revenue with user summary
        roas_df = revenue_df.merge(user_summary_df, how='outer', on=['teamid'])

        roas_df["day_revenue"] = roas_df["day_iap_revenue"] + roas_df["day_tapjoy_revenue"] + roas_df["day_ad_revenue"]
        roas_df.drop(["day_iap_revenue", "day_tapjoy_revenue", "day_ad_revenue"], axis=1, inplace=True)

        # Get number of installs per date and ua type
        install_info_df = roas_df.assign(
            organic_installs=np.where(roas_df['ua_type'] == 'Organic', roas_df.teamid, 0),
            ua_installs=np.where(roas_df['ua_type'] == 'UA', roas_df.teamid, 0),
            android_count=np.where(roas_df['is_android'] == 1, roas_df.teamid, 0),
            us_count=np.where(roas_df['is_us'] == 1, roas_df.teamid, 0),
            row_count=np.where(roas_df['is_row'] == 1, roas_df.teamid, 0),
            users_retained_day1=np.where(roas_df['retention1'] == 1, roas_df.teamid, 0),
            users_retained_day7=np.where(roas_df['retention7'] == 1, roas_df.teamid, 0),
        ).groupby('install_date').agg({'organic_installs': "nunique",
                                       'ua_installs': "nunique",
                                       'android_count': "nunique",
                                       'us_count': "nunique",
                                       'row_count': "nunique",
                                       'users_retained_day1': "nunique",
                                       'users_retained_day7': "nunique"}).reset_index()

        revenue_info_df = roas_df.assign(
            day_organic_revenue=np.where(roas_df['ua_type'] == 'Organic', roas_df.day_revenue, 0),
            day_ua_revenue=np.where(roas_df['ua_type'] == 'UA', roas_df.day_revenue, 0)
        ).groupby(['install_date', 'clean_date']).agg({'day_organic_revenue': "sum",
                                                       'day_ua_revenue': "sum"}).reset_index()

        roas_df = revenue_info_df.merge(install_info_df, how='inner', on=["install_date"])

        # Merge with spend to get CPI column
        roas_df = roas_df.merge(spend_df, how="outer", on=["install_date"])
        roas_df["days_since_install"] = (roas_df["clean_date"] - roas_df["install_date"]).dt.days

        # Aggregate data per install date and days since install date
        roas_df = roas_df.groupby(["install_date", "days_since_install"]) \
            .agg(ua_installs=pd.NamedAgg(column="ua_installs", aggfunc=np.mean),
                 organic_installs=pd.NamedAgg(column="organic_installs", aggfunc=np.mean),
                 users_retained_day1=pd.NamedAgg(column="users_retained_day1", aggfunc=np.mean),
                 users_retained_day7=pd.NamedAgg(column="users_retained_day7", aggfunc=np.mean),
                 android_count=pd.NamedAgg(column="android_count", aggfunc="sum"),
                 us_count=pd.NamedAgg(column="us_count", aggfunc="sum"),
                 row_count=pd.NamedAgg(column="row_count", aggfunc="sum"),
                 spend=pd.NamedAgg(column="spend", aggfunc=np.mean),
                 day_ua_revenue=pd.NamedAgg(column="day_ua_revenue", aggfunc="sum"),
                 day_organic_revenue=pd.NamedAgg(column="day_organic_revenue", aggfunc="sum")).reset_index()

        roas_df = roas_df.fillna({"day_ua_revenue": 0, "day_organic_revenue": 0, "spend": 0})
        roas_df['spend'] = roas_df['spend'].apply(np.int64)

        roas_df["retention1"] = (roas_df["users_retained_day1"] / (roas_df["ua_installs"] + roas_df["organic_installs"])).astype(int)
        roas_df["retention7"] = (roas_df["users_retained_day7"] / (roas_df["ua_installs"] + roas_df["organic_installs"])).astype(int)

        all_game_names = [game for game, start_date in game_datemapper.items()]
        for game_ in all_game_names:
            if game_ == game:
                roas_df[f'is_{game_}'] = 1
            else:
                roas_df[f'is_{game_}'] = 0

        df = roas_df.copy()
        df = df[(df["ua_installs"] > 25) & (df["spend"] > 25)]
        df["all_revenue"] = df["day_ua_revenue"] + df["day_organic_revenue"]
        df = df.sort_values(['install_date', 'days_since_install'], ascending=[True, True])
        df["running_revenue_total"] = df.groupby('install_date')['all_revenue'].cumsum()
        df["roas"] = df["running_revenue_total"] / df["spend"]
        df['roas'] = df.groupby(['install_date'])['roas'].ffill()
        df = df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()
        plt.scatter(x=df["days_since_install"],
                    y=df['roas'])
        plt.colorbar(label="UA Installs", orientation="vertical")
        plt.title(f'Actual ROAS {game} by install count')
        plt.show()


        roas_df.drop(["install_date",
                      "users_retained_day1",
                      "users_retained_day7",
                      "day_ua_revenue",
                      "day_organic_revenue"], axis=1, inplace=True)

        print("Predictions looks like:\n\n", roas_df.sample(5))

        predictions = self.model.predict(roas_df)

        roas_df["predicted_roas"] = predictions

        plt.scatter(x=roas_df["days_since_install"],
                    y=roas_df['predicted_roas'],
                    c=roas_df["ua_installs"])
        plt.colorbar(label="UA Installs", orientation="vertical")
        plt.title(f'Prediction of {game} by install count')
        plt.show()
