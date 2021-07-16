import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date, timedelta
from scipy.optimize import curve_fit

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

    def __init__(self, config, min_num_installs=200, use_cached_data=False, comprehensive_run=False):
        self.earliest_date = config['start_date']
        self.earliest_np_date = pd.Timestamp(date(int(self.earliest_date[:4]),
                                                  int(self.earliest_date[5:7]),
                                                  int(self.earliest_date[8:])))
        self.end_date = config['end_date']
        self.min_num_installs = min_num_installs
        self.raw_roas_df = self.load_cached_roas_df() if use_cached_data else self.build_roas_df_from_redshift()
        self.data_blend = config.get('data_blend')
        self.data_function = config.get('data_function')
        self.max_train_ages = [config.get('max_train_age')] if config.get('max_train_age') else [30, 60, 90]
        self.blended_roas_df = None
        self.math_helper = MathHelper()
        self.utilities = Utilities()

        self.plot_all_raw_data()
        self.choose_blended_data()
        if self.data_blend or comprehensive_run:
            self.choose_math_function()
        if (self.data_function and len(self.max_train_ages) == 1) or comprehensive_run:
            self.run_curve()

    def build_roas_df_from_redshift(self):
        redshift_helper = RedshiftHelper(earliest_date=self.earliest_date, end_date=self.end_date)
        user_summary_df = redshift_helper.get_user_summary_df()
        spend_df = redshift_helper.get_spend_df()
        revenue_df = redshift_helper.get_revenue_df()

        revenue_df["days_since_install"] = (revenue_df["clean_date"] - revenue_df["install_date"]).dt.days

        # Create installs column
        roas_df = revenue_df.groupby("install_date").agg(
            installs=pd.NamedAgg(column="teamid", aggfunc="nunique")).reset_index()

        # Merge with spend to get CPI column
        roas_df = pd.merge(roas_df, spend_df, how="inner", on=["install_date"])
        roas_df["cpi"] = roas_df["spend"] / roas_df["installs"]

        # Merge with revenue to get revenue
        roas_df = pd.merge(roas_df, revenue_df, how="inner", on=["install_date"])

        # Assign each day to a weekly cohort, with the cohorts starting on day 1
        offset_week_day = 7 - self.earliest_np_date.weekday()
        roas_df["offset_install_date"] = roas_df['install_date'] + timedelta(days=offset_week_day)
        roas_df['install_cohort'] = roas_df['offset_install_date'].dt.to_period('W').apply(lambda r: r.start_time)
        roas_df["install_cohort"] = roas_df['install_cohort'] - timedelta(days=offset_week_day)

        # Filter out days with too few installs
        roas_df = roas_df[roas_df["installs"] > self.min_num_installs]

        # Aggregate data per install date and days since install date
        roas_df = roas_df.groupby(["install_date", "install_cohort", "days_since_install"]) \
            .agg(cpi=pd.NamedAgg(column="cpi", aggfunc=np.mean),
                 installs=pd.NamedAgg(column="installs", aggfunc=np.mean),
                 spend=pd.NamedAgg(column="spend", aggfunc=np.mean),
                 day_iap_revenue=pd.NamedAgg(column="day_iap_revenue", aggfunc="sum"),
                 day_ad_revenue=pd.NamedAgg(column="day_ad_revenue", aggfunc="sum"),
                 day_tapjoy_revenue=pd.NamedAgg(column="day_tapjoy_revenue", aggfunc="sum")).reset_index()
        roas_df["day_revenue"] = roas_df['day_iap_revenue'] + roas_df['day_ad_revenue'] + roas_df['day_tapjoy_revenue']
        roas_df = roas_df.sort_values(['install_date', 'days_since_install'], ascending=[True, True])
        roas_df["running_revenue_total"] = roas_df.groupby('install_date')['day_revenue'].cumsum()
        roas_df["roas"] = roas_df["running_revenue_total"] / roas_df["spend"]

        # Add the weighted column using months since install
        weighted_df = pd.DataFrame()
        roas_df["weighting"] = (roas_df['install_date'] - self.earliest_np_date) // np.timedelta64(1, 'M') + 1
        for player_age in roas_df['days_since_install'].unique():
            same_age_df = roas_df[roas_df['days_since_install'] == player_age]
            weights = same_age_df["weighting"].sum()
            same_age_df["weighted_unbalanced_roas"] = same_age_df["roas"] * same_age_df["weighting"]
            same_age_df["row_weight_balance"] = weights
            weighted_df = weighted_df.append(same_age_df[["install_date",
                                                "days_since_install",
                                                "weighted_unbalanced_roas",
                                                "row_weight_balance"]])
        weighted__df = roas_df.merge(weighted_df, how='outer', on=["install_date", "days_since_install"])
        roas_df = roas_df.merge(weighted_df, how='outer', on=["install_date", "days_since_install"])

        roas_df.to_feather("roas_df.feather")

        return roas_df

    @staticmethod
    def load_cached_roas_df():
        return pd.read_feather("roas_df.feather")

    def get_roas_mean_df(self):
        return self.raw_roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()

    def get_roas_median_df(self):
        return self.raw_roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.median)).reset_index()

    def get_roas_weighted_df(self):
        return self.raw_roas_df.groupby(['days_since_install']).apply(
            lambda x: x['weighted_unbalanced_roas'].sum() / x['row_weight_balance'].max()).reset_index(name='roas')

    def get_roas_weighted_median_split(self, split_date=30):
        weighted_df = self.raw_roas_df[self.raw_roas_df["days_since_install"] < 60].groupby(['days_since_install']).apply(
            lambda x: x['weighted_unbalanced_roas'].sum() / x['row_weight_balance'].max()).reset_index(
            name='weighted_roas')
        median_df = self.raw_roas_df[self.raw_roas_df["days_since_install"] >= 60].groupby("days_since_install").agg(
            median_roas=pd.NamedAgg(column="roas", aggfunc=np.median)).reset_index()
        split_df = weighted_df.append(median_df)
        split_df["roas"] = np.where(split_df["weighted_roas"].isnull(), split_df["median_roas"],
                                    split_df["weighted_roas"])
        return split_df

    def plot_all_raw_data(self):
        sns.lineplot(data=self.raw_roas_df, x='days_since_install', y='roas', hue='install_cohort')
        plt.title(f'Raw data cohorted by week')
        plt.show()

    def choose_blended_data(self):
        datasets_to_fit = [
            {"data_name": "mean", "data": self.get_roas_mean_df},
            {"data_name": "weighted-mean", "data": self.get_roas_weighted_df},
            {"data_name": "median", "data": self.get_roas_median_df},
            {"data_name": "weighted-median-age-split", "data": self.get_roas_weighted_median_split},
        ]
        assert self.data_blend in [dataset.get("data_name") for dataset in datasets_to_fit] + [None], \
            f"Unknown data_blend {self.data_blend}. Try using 'mean' or 'median'"
        for dataset in datasets_to_fit:
            df = dataset.get("data")()
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

        for function_dict in functions_to_fit:
            function, function_name = function_dict.get("function"), function_dict.get("name")
            sns.lineplot(x=X, y=y, data=self.blended_roas_df)
            for max_age in self.max_train_ages:
                # Fit the curve up to the max train age
                max_age_df = self.blended_roas_df[self.blended_roas_df["days_since_install"] < max_age]
                X_max_age = max_age_df.days_since_install.values
                Y_max_age = max_age_df.roas.values
                popt, pcov = curve_fit(function, X_max_age, Y_max_age, maxfev=20000)

                # Predict the first 30 days
                X_0_to_30 = self.blended_roas_df[self.blended_roas_df["days_since_install"] < 30].days_since_install.values
                y_0_to_30 = self.blended_roas_df[self.blended_roas_df["days_since_install"] < 30].roas.values
                y_pred_0_to_30 = function(X_0_to_30, *popt)

                # Predict the data after 30 days
                X_30_plus = self.blended_roas_df[self.blended_roas_df["days_since_install"] >= 30].days_since_install.values
                y_30_plus = self.blended_roas_df[self.blended_roas_df["days_since_install"] >= 30].roas.values
                y_pred_30_plus = function(X_30_plus, *popt)

                # Calculate the error metrics
                self.utilities.calculate_error_metrics(function_name=function_name,
                                                       dataset_tested_on=self.data_blend,
                                                       max_fit_age=max_age,
                                                       y_0_to_30=y_0_to_30,
                                                       y_pred_0_to_30=y_pred_0_to_30,
                                                       y_30_plus=y_30_plus,
                                                       y_pred_30_plus=y_pred_30_plus)

                # Calculate prediction on full data for graphing
                y_pred = function(X, *popt)
                plt.plot(X, y_pred, color=self.utilities.get_color(), linewidth=3, label=f"max training age: {max_age}")
            plt.title(f'{function_name} on data {self.data_blend}')
            plt.legend(loc="lower right")
            plt.show()

        self.utilities.print_final_error_metrics()
        if not self.data_function:
            self.utilities.print_function_message(functions_to_fit)
        if not len(self.max_train_ages) == 1:
            self.utilities.print_train_age_message(self.max_train_ages)

    def run_curve(self):
        pass
