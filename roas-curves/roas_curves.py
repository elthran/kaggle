import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date, timedelta

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


class RoasCurve:

    def __init__(self, config, min_num_installs=200, use_cached_data=False, plot_all=False, comprehensive_run=False):
        self.earliest_np_date = pd.Timestamp(date(int(config['start_date'][:4]),
                                                  int(config['start_date'][5:7]),
                                                  int(config['start_date'][8:])))
        self.latest_np_end_date = pd.Timestamp(date(int(config['end_date'][:4]),
                                                    int(config['end_date'][5:7]),
                                                    int(config['end_date'][8:])))
        self.min_num_installs = min_num_installs
        self.data_blend = config.get('data_blend')
        self.data_function = config.get('data_function')
        self.max_train_ages = [config.get('max_train_age')] if config.get('max_train_age') else [30, 45, 60, 90]

        self.plot_all = plot_all
        self.redshift_helper = RedshiftHelper(config)
        self.math_helper = MathHelper()
        self.utilities = Utilities()

        self.raw_roas_df = self.load_cached_roas_df() if use_cached_data else self.build_roas_df_from_redshift()
        self.blended_roas_df = None
        self.predictions = {}
        self.plot_all_raw_data()

        self.choose_blended_data()
        if self.data_blend or comprehensive_run:
            self.choose_math_function()
        if (self.data_function and len(self.max_train_ages) == 1) or comprehensive_run:
            self.run_breakeven_curves()

    def build_roas_df_from_redshift(self):
        self.redshift_helper.set_redshift_connection()
        user_summary_df = self.redshift_helper.get_user_summary_df()
        spend_df = self.redshift_helper.get_spend_df()
        revenue_df = self.redshift_helper.get_revenue_df()
        self.redshift_helper.redshift_engine.close()

        # Merge revenue with user summary
        roas_df = revenue_df.merge(user_summary_df, how='outer', on=['teamid'])

        roas_df["day_revenue"] = roas_df["day_iap_revenue"] + roas_df["day_tapjoy_revenue"] + roas_df["day_ad_revenue"]

        # Get number of installs per date and ua type
        install_info_df = roas_df.assign(
            organic_installs=np.where(roas_df['ua_type'] == 'Organic', roas_df.teamid, 0),
            ua_installs=np.where(roas_df['ua_type'] == 'UA', roas_df.teamid, 0),
        ).groupby('install_date').agg({'organic_installs': "nunique",
                                       'ua_installs': "nunique"}).reset_index()

        revenue_info_df = roas_df.assign(
            day_organic_revenue=np.where(roas_df['ua_type'] == 'Organic', roas_df.day_revenue, 0),
            day_ua_revenue=np.where(roas_df['ua_type'] == 'UA', roas_df.day_revenue, 0)
        ).groupby(['install_date', 'clean_date']).agg({'day_organic_revenue': "sum",
                                                       'day_ua_revenue': "sum"}).reset_index()

        roas_df = revenue_info_df.merge(install_info_df, how='inner', on=["install_date"])

        # Merge with spend to get CPI column
        roas_df = roas_df.merge(spend_df, how="outer", on=["install_date"])
        roas_df["cpi"] = roas_df["spend"] / roas_df["ua_installs"]

        roas_df["days_since_install"] = (roas_df["clean_date"] - roas_df["install_date"]).dt.days

        # Aggregate data per install date and days since install date
        roas_df = roas_df.groupby(["install_date", "days_since_install"]) \
            .agg(cpi=pd.NamedAgg(column="cpi", aggfunc=np.mean),
                 ua_installs=pd.NamedAgg(column="ua_installs", aggfunc=np.mean),
                 organic_installs=pd.NamedAgg(column="organic_installs", aggfunc=np.mean),
                 spend=pd.NamedAgg(column="spend", aggfunc=np.mean),
                 day_ua_revenue=pd.NamedAgg(column="day_ua_revenue", aggfunc="sum"),
                 day_organic_revenue=pd.NamedAgg(column="day_organic_revenue", aggfunc="sum")).reset_index()

        # Fill in null days
        nan_filled_df = pd.DataFrame()
        for install_date in roas_df['install_date'].unique():
            same_install_age_df = roas_df[roas_df['install_date'] == install_date].copy()
            # max_age = (self.latest_np_end_date - install_date).days
            max_age = same_install_age_df["days_since_install"].max()
            same_install_age_df = same_install_age_df.set_index('days_since_install')
            # Add null rows for missing days
            same_install_age_df = same_install_age_df.reindex(range(0, max_age)).reset_index()
            # Fill null values for missing days
            same_install_age_df = same_install_age_df.fillna({"day_ua_revenue": 0,
                                                              "day_organic_revenue": 0})
            same_install_age_df = same_install_age_df.ffill(axis=0)
            nan_filled_df = nan_filled_df.append(same_install_age_df)
        roas_df = nan_filled_df

        roas_df = roas_df[(roas_df["ua_installs"] > self.min_num_installs) & (roas_df["spend"] > 0)]

        organic_multiplier = self.get_organic_lift(roas_df)

        # Calculate roas values
        roas_df = roas_df.sort_values(['install_date', 'days_since_install'], ascending=[True, True])
        roas_df["running_ua_revenue_total"] = roas_df.groupby('install_date')['day_ua_revenue'].cumsum()
        roas_df["roas"] = roas_df["running_ua_revenue_total"] * organic_multiplier / roas_df["spend"]
        roas_df['roas'] = roas_df.groupby(['install_date'])['roas'].ffill()

        # Assign each day to a weekly cohort, with the cohorts starting on day 1
        offset_week_day = 7 - self.earliest_np_date.weekday()
        roas_df["offset_install_date"] = roas_df['install_date'] + timedelta(days=offset_week_day)
        roas_df['install_cohort'] = roas_df['offset_install_date'].dt.to_period('W').apply(lambda r: r.start_time)
        roas_df["install_cohort"] = roas_df['install_cohort'] - timedelta(days=offset_week_day)
        roas_df.drop("offset_install_date", axis=1, inplace=True)

        # Add install age (days since the first install date)
        roas_df["install_age"] = (roas_df['install_date'] - self.earliest_np_date).dt.days + 1

        def get_weighted_roas(df=None, weight_column=None, name=None):
            weighted_df = pd.DataFrame()
            for player_age in roas_df['days_since_install'].unique():
                same_age_df = roas_df[roas_df['days_since_install'] == player_age].copy()
                same_age_df["weighted_roas"] = same_age_df["roas"] * same_age_df[weight_column]
                same_age_df["weight"] = same_age_df[weight_column].sum()
                same_age_df[name] = same_age_df['weighted_roas'].sum() / same_age_df['weight'].max()
                weighted_df = weighted_df.append(same_age_df[["install_date",
                                                              "days_since_install",
                                                              name]])
            return roas_df.merge(weighted_df, how='outer', on=["install_date", "days_since_install"])

        roas_df = get_weighted_roas(df=roas_df, weight_column="install_age", name="age_weighted_roas")
        roas_df = get_weighted_roas(df=roas_df, weight_column="ua_installs", name="install_weighted_roas")

        roas_df.to_feather("roas_df.feather")

        return roas_df

    @staticmethod
    def load_cached_roas_df():
        return pd.read_feather("roas_df.feather")

    def get_organic_lift(self, roas_df):
        organic_df = roas_df.groupby("install_date") \
            .agg(day_ua_revenue=pd.NamedAgg(column="day_ua_revenue", aggfunc="sum"),
                 day_organic_revenue=pd.NamedAgg(column="day_organic_revenue", aggfunc="sum")).reset_index()
        organic_df["organic_percent"] = organic_df["day_organic_revenue"] / (
                organic_df["day_organic_revenue"] + organic_df["day_ua_revenue"])
        if self.plot_all:
            plt.plot(organic_df["install_date"], organic_df["organic_percent"], linewidth=4)
            plt.title(f'Percent of revenue that is organic by install date')
            plt.show()
        organic_df = organic_df.groupby(lambda x: True) \
            .agg(day_ua_revenue=pd.NamedAgg(column="day_ua_revenue", aggfunc="sum"),
                 day_organic_revenue=pd.NamedAgg(column="day_organic_revenue", aggfunc="sum")).reset_index()
        organic_df["organic_percent"] = organic_df["day_organic_revenue"] / (
                organic_df["day_organic_revenue"] + organic_df["day_ua_revenue"])
        raw_organic_revenue_percent = int(100 * organic_df['organic_percent'].max())
        if raw_organic_revenue_percent > 20 or raw_organic_revenue_percent < 15:
            print(f"Organics are {raw_organic_revenue_percent}% of revenue. "
                  f"This indicates an abnormality and will be adjusted.")
        organic_revenue_percent = max(15, min(20, raw_organic_revenue_percent))  # Keep it within 15-20%
        print(f"organic_revenue_percent is being set as {organic_revenue_percent}")
        return 100 / (100 - organic_revenue_percent)

    def plot_all_raw_data(self):
        sns.lineplot(data=self.raw_roas_df, x='days_since_install', y='roas', hue='install_cohort', ci=False)
        plt.title(f'Raw data cohorted by week')
        plt.show()

        plt.scatter(x=self.raw_roas_df["days_since_install"],
                    y=self.raw_roas_df["roas"],
                    c=self.raw_roas_df["install_age"])
        plt.colorbar(label="Install Date", orientation="vertical")
        plt.title(f'Raw data by install date heatmap')
        plt.show()

    def get_roas_mean_df(self):
        return self.raw_roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()

    def get_roas_median_df(self):
        return self.raw_roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.median)).reset_index()

    def get_roas_age_weighted_df(self):
        return self.raw_roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="age_weighted_roas", aggfunc=np.mean)).reset_index()

    def get_roas_install_weighted_df(self):
        return self.raw_roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="install_weighted_roas", aggfunc=np.mean)).reset_index()

    def get_roas_weighted_median_split_df(self, split_date=30):
        new_df = self.raw_roas_df.copy()
        new_df["roas"] = np.where(new_df["days_since_install"] < split_date, new_df["age_weighted_roas"],
                                  new_df["roas"])
        return new_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()

    def choose_blended_data(self):
        datasets_to_fit = [
            {"data_name": "mean", "data": self.get_roas_mean_df},
            {"data_name": "age-weighted", "data": self.get_roas_age_weighted_df},
            {"data_name": "median", "data": self.get_roas_median_df},
            {"data_name": "install-weighted", "data": self.get_roas_install_weighted_df},
            {"data_name": "age-weighted-median-split", "data": self.get_roas_weighted_median_split_df},
        ]
        assert self.data_blend in [dataset.get("data_name") for dataset in datasets_to_fit] + [None], \
            f"Unknown data_blend {self.data_blend}. Try using 'mean' or 'median'"
        for dataset in datasets_to_fit:
            df = dataset.get("data")()
            df["roas"] = df["roas"]
            X = df.days_since_install.values
            y = df.roas.values
            if dataset.get("data_name") == self.data_blend:
                self.blended_roas_df = df
                plt.plot(X, y, color=self.utilities.get_color("black"), label=dataset.get("data_name"), linewidth=4)
            else:
                plt.plot(X, y, color=self.utilities.get_color(), label=dataset.get("data_name"), linewidth=2, alpha=0.5)
        plt.title(f'Different averages of the raw data')
        plt.legend(loc="lower right")
        plt.show()

        if not self.data_blend:
            self.utilities.print_blend_message(datasets_to_fit)

    def choose_math_function(self):
        functions_to_fit = self.math_helper.get_functions(self.data_function)
        X = self.blended_roas_df.days_since_install.values
        y = self.blended_roas_df.roas.values
        y_pred = None

        for function_dict in functions_to_fit:
            self.math_helper.function, function_name = function_dict.get("function"), function_dict.get("name")
            sns.lineplot(x=X, y=y, data=self.blended_roas_df, color="k", label="Training Data")
            self.utilities.reset_color()
            for max_age in self.max_train_ages:
                # Fit the curve up to the max train age
                max_age_df = self.blended_roas_df[self.blended_roas_df["days_since_install"] < max_age]
                X_max_age = max_age_df.days_since_install.values
                Y_max_age = max_age_df.roas.values
                self.math_helper.fit_curve(X_max_age, Y_max_age)

                # Predict the first 30 days
                X_0_to_30 = self.blended_roas_df[
                    self.blended_roas_df["days_since_install"] < 30].days_since_install.values
                y_0_to_30 = self.blended_roas_df[self.blended_roas_df["days_since_install"] < 30].roas.values
                y_pred_0_to_30 = self.math_helper.run_prediction(X_0_to_30)

                # Predict the data after 30 days
                X_30_plus = self.blended_roas_df[
                    self.blended_roas_df["days_since_install"] >= 30].days_since_install.values
                y_30_plus = self.blended_roas_df[self.blended_roas_df["days_since_install"] >= 30].roas.values
                y_pred_30_plus = self.math_helper.run_prediction(X_30_plus)

                # Calculate the error metrics
                self.utilities.calculate_error_metrics(function_name=function_name,
                                                       dataset_tested_on=self.data_blend,
                                                       max_fit_age=max_age,
                                                       y_0_to_30=y_0_to_30,
                                                       y_pred_0_to_30=y_pred_0_to_30,
                                                       y_30_plus=y_30_plus,
                                                       y_pred_30_plus=y_pred_30_plus)

                # Calculate prediction on full data for graphing
                y_pred = self.math_helper.run_prediction(X)
                plt.plot(X, y_pred, color=self.utilities.get_color(), linewidth=2, label=f"max training age: {max_age}")

            if len(self.max_train_ages) == 1:  # Add the max age to the graph if it was chosen
                plt.axvline(x=self.max_train_ages[0], ymin=0, ymax=1, color="k", label="Max train age")
            if y_pred[-1] > 0.8:  # If the data gets near breakeven, add it to the graph for a visual cue
                plt.axhline(y=1, color="k", label="Breakeven")
            plt.title(f'{function_name} on data {self.data_blend}')
            plt.xlim([0, 45])
            plt.legend(loc="lower right")
            plt.show()

        self.utilities.print_final_error_metrics()
        if not self.data_function:
            self.utilities.print_function_message(functions_to_fit)
        if not len(self.max_train_ages) == 1:
            self.utilities.print_train_age_message(self.max_train_ages)

    def run_breakeven_curves(self):
        self.predictions["natural"] = self.math_helper.run_prediction([i for i in range(0, 46)])
        # sns.lineplot(x=[i for i in range(len(self.predictions["natural"]))], y=self.predictions["natural"], color="b",
        #              ci=False,
        #              label="Jacob predict")
        for day in [45]:
            day_1_lift_required = 1 / self.predictions["natural"][day]
            self.predictions[day] = [prediction * day_1_lift_required for prediction in self.predictions["natural"]]
            sns.lineplot(x=[i for i in range(len(self.predictions[day]))],
                         y=self.predictions[day],
                         color="k",
                         ci=False,
                         label=f"Jacob - Day {day} BE")
        plt.axhline(y=1, c="k")
        plt.title("2020-09-23-app-level-box-office.json")

        graeme_df = self.plot_graemes_be_curves()
        sns.lineplot(x=graeme_df.age, y=graeme_df.roas, linestyle="dashed",
                     color="k", ci=False, label=f"Graeme - Day 45 BE")

        future_df = self.plot_future_data()
        # future_X = future_df.days_since_install.values
        # future_y = future_df.roas.values
        # sns.lineplot(x=future_X, y=future_y, color="k", ci=False, label=f"Actual data")
        sns.lineplot(data=future_df, x='days_since_install', y='roas', hue='install_cohort', ci=False)

        plt.show()




    # Below are completely unneeded functions to validate vs Graeme's curves
    def plot_graemes_be_curves(self):
        df = pd.read_csv("graeme.csv")
        graeme_be_df = df[df["target_scenario"] == f"be45"]
        graeme_be_df["age"] = graeme_be_df["age"] - 1
        return graeme_be_df

    # Below are completely unneeded functions to validate vs Graeme's curves
    def plot_future_data(self):
        self.redshift_helper_2 = RedshiftHelper({
            'game_name': 'hydrostone',
            'start_date': '2020-09-23',
            'end_date': '2021-01-23'})
        self.earliest_np_date = pd.Timestamp(date(2020, 9, 23))
        self.latest_np_end_date = pd.Timestamp(date(2021, 1, 23))
        self.redshift_helper_2.set_redshift_connection()
        user_summary_df = self.redshift_helper_2.get_user_summary_df()
        spend_df = self.redshift_helper_2.get_spend_df()
        revenue_df = self.redshift_helper_2.get_revenue_df()
        self.redshift_helper_2.redshift_engine.close()

        # Merge revenue with user summary
        roas_df = revenue_df.merge(user_summary_df, how='outer', on=['teamid'])

        roas_df["day_revenue"] = roas_df["day_iap_revenue"] + roas_df["day_tapjoy_revenue"] + roas_df[
            "day_ad_revenue"]

        # Get number of installs per date and ua type
        install_info_df = roas_df.assign(
            organic_installs=np.where(roas_df['ua_type'] == 'Organic', roas_df.teamid, 0),
            ua_installs=np.where(roas_df['ua_type'] == 'UA', roas_df.teamid, 0),
        ).groupby('install_date').agg({'organic_installs': "nunique",
                                       'ua_installs': "nunique"}).reset_index()

        revenue_info_df = roas_df.assign(
            day_organic_revenue=np.where(roas_df['ua_type'] == 'Organic', roas_df.day_revenue, 0),
            day_ua_revenue=np.where(roas_df['ua_type'] == 'UA', roas_df.day_revenue, 0)
        ).groupby(['install_date', 'clean_date']).agg({'day_organic_revenue': "sum",
                                                       'day_ua_revenue': "sum"}).reset_index()

        roas_df = revenue_info_df.merge(install_info_df, how='inner', on=["install_date"])

        # Merge with spend to get CPI column
        roas_df = roas_df.merge(spend_df, how="outer", on=["install_date"])
        roas_df["cpi"] = roas_df["spend"] / roas_df["ua_installs"]

        roas_df["days_since_install"] = (roas_df["clean_date"] - roas_df["install_date"]).dt.days

        # Aggregate data per install date and days since install date
        roas_df = roas_df.groupby(["install_date", "days_since_install"]) \
            .agg(cpi=pd.NamedAgg(column="cpi", aggfunc=np.mean),
                 ua_installs=pd.NamedAgg(column="ua_installs", aggfunc=np.mean),
                 organic_installs=pd.NamedAgg(column="organic_installs", aggfunc=np.mean),
                 spend=pd.NamedAgg(column="spend", aggfunc=np.mean),
                 day_ua_revenue=pd.NamedAgg(column="day_ua_revenue", aggfunc="sum"),
                 day_organic_revenue=pd.NamedAgg(column="day_organic_revenue", aggfunc="sum")).reset_index()

        # Fill in null days
        nan_filled_df = pd.DataFrame()
        for install_date in roas_df['install_date'].unique():
            same_install_age_df = roas_df[roas_df['install_date'] == install_date].copy()
            # max_age = (self.latest_np_end_date - install_date).days
            max_age = same_install_age_df["days_since_install"].max()
            same_install_age_df = same_install_age_df.set_index('days_since_install')
            # Add null rows for missing days
            same_install_age_df = same_install_age_df.reindex(range(0, max_age)).reset_index()
            # Fill null values for missing days
            same_install_age_df = same_install_age_df.fillna({"day_ua_revenue": 0,
                                                              "day_organic_revenue": 0})
            same_install_age_df = same_install_age_df.ffill(axis=0)
            nan_filled_df = nan_filled_df.append(same_install_age_df)
        roas_df = nan_filled_df

        print("min stuff:", roas_df["ua_installs"].max())

        roas_df = roas_df[(roas_df["ua_installs"] > 100) & (roas_df["spend"] > 0)]

        organic_multiplier = self.get_organic_lift(roas_df)

        # Calculate roas values
        roas_df = roas_df.sort_values(['install_date', 'days_since_install'], ascending=[True, True])
        roas_df["running_ua_revenue_total"] = roas_df.groupby('install_date')['day_ua_revenue'].cumsum()
        roas_df["roas"] = roas_df["running_ua_revenue_total"] * organic_multiplier / roas_df["spend"]
        roas_df['roas'] = roas_df.groupby(['install_date'])['roas'].ffill()

        # Assign each day to a weekly cohort, with the cohorts starting on day 1
        # offset_week_day = 7 - self.earliest_np_date.weekday()
        # roas_df["offset_install_date"] = roas_df['install_date'] + timedelta(days=offset_week_day)
        # roas_df['install_cohort'] = roas_df['offset_install_date'].dt.to_period('W').apply(lambda r: r.start_time)
        # roas_df["install_cohort"] = roas_df['install_cohort'] - timedelta(days=offset_week_day)
        # roas_df.drop("offset_install_date", axis=1, inplace=True)
        roas_df['install_cohort'] = roas_df['install_date'].dt.to_period('M').apply(lambda r: r.start_time)


        # Add install age (days since the first install date)
        roas_df["install_age"] = (roas_df['install_date'] - self.earliest_np_date).dt.days + 1

        roas_df = roas_df[roas_df["days_since_install"] < 45]

        return roas_df

        # self.raw_roas_df = roas_df
        # df = self.get_roas_mean_df()
        # return df
