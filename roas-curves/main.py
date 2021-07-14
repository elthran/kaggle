import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date, timedelta
from scipy.optimize import curve_fit

from redshift_utilities import RedshiftHelper
from math_functions import MathHelper

# Set pandas options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set matplotlib options
plt.interactive(False)


class RoasCurve:

    def __init__(self, min_num_installs=200, use_cached_data=False, load_plots=False):
        self.earliest_date = '2021-03-24'
        self.earliest_np_date = pd.Timestamp(date(int(self.earliest_date[:4]),
                                       int(self.earliest_date[5:7]),
                                       int(self.earliest_date[8:])))
        self.end_date = '2021-07-01'
        self.min_num_installs = min_num_installs
        self.roas_df = self.load_cached_roas_df() if use_cached_data else self.build_roas_df_from_redshift()
        self.math_helper = MathHelper()

    def build_roas_df_from_redshift(self):
        redshift_helper = RedshiftHelper()
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
        offset_week_day = 7 - start_date.weekday()
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

        roas_df["install_week"] = (roas_df['install_date'] - self.earliest_np_date) // np.timedelta64(1, 'W')

        roas_df.to_feather("roas_df.feather")
        return roas_df

    @staticmethod
    def load_cached_roas_df():
        return pd.read_feather("roas_df.feather")

    def get_roas_mean_df(self):
        return self.roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()

    def get_roas_weighted_df(self):
        df = self.roas_df.copy()
        # Use the install week as the weighting (so later installs are heavier)
        df["weight_column"] = df["install_week"] + 1
        total_weight = df["weight_column"].sum()
        df['weighted_roas'] = df['roas'] * df['weight_column'] / total_weight
        df.drop(["weight_column"], inplace=True, axis=1)

        # for days_since_install in df['days_since_install'].unique():
        #     df["weight"] = (df["install_date"] - self.earliest_np_date).dt.days
        #     max_weight = df.loc[df['install_week'] == days_since_install, 'weight'].max()
        #     df['weighted_roas'] = df['roas'] * df['install_week'] / max_weight

        print(df.head(2000))


        return df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="weighted_roas", aggfunc=np.mean)).reset_index()


    def plot_all_raw_data(self):
        sns.lineplot(data=self.roas_df, x='days_since_install', y='roas', hue='install_cohort')
        plt.title(f'Raw data cohorted by week')
        plt.show()

    def calculate_all_curves(self):

        functions_to_fit = [self.math_helper.generalized_logistic_function,
                            self.math_helper.modified_powerlaw_function,
                            self.math_helper.heavily_modified_logarithmic_function]
        datasets_to_fit = [{"data_name": "mean of all-time data", "data": self.get_roas_mean_df()},
                           {"data_name": "forward-weighted mean of all-time data", "data": self.get_roas_weighted_df()}]
        summary_of_results = []

        for function in functions_to_fit:
            for dataset in datasets_to_fit:
                df = dataset.get("data")
                X = df.days_since_install.values
                y = df.roas.values
                popt, pcov = curve_fit(function, X, y, maxfev=20000)
                y_pred = function(X, *popt)
                sns.lineplot(x=X, y=y, data=df, ci=False)
                plt.plot(X, y_pred, color='black', linewidth=1)
                plt.title(f'{function.__name__} fit on {dataset.get("data_name")}')
                error_metrics = self.get_error_metrics(popt, y, y_pred)
                summary_of_results.append({"error_metrics": error_metrics,
                                           "function_used": function.__name__,
                                           "dataset_tested_on": dataset.get("data_name")})
                print(f'{function.__name__} has RMSE of {error_metrics["root_mean_squared_error"]} '
                      f'on {dataset.get("data_name")}')
                print(error_metrics)
                print("used parameters:", popt)
                plt.show()

        best_results = min(summary_of_results, key=lambda x: x["error_metrics"]["root_mean_squared_error"])
        best_error = best_results["error_metrics"]["root_mean_squared_error"]
        best_data = best_results["dataset_tested_on"]

        print(f"""Best results is {best_results["function_used"]} with rmse of {best_error} on {best_data}""")

    def get_error_metrics(self, popt, y, y_pred):
        absolute_error = y_pred - y
        squared_error = np.square(absolute_error)
        mean_squared_error = np.mean(squared_error)
        root_mean_squared_error = np.sqrt(mean_squared_error)
        r_squared_error = 1.0 - (np.var(absolute_error) / np.var(y))
        return {"absolute_error": absolute_error,
                "squared_error": squared_error,
                "mean_squared_error": mean_squared_error,
                "root_mean_squared_error": root_mean_squared_error,
                "r_squared_error": r_squared_error}


curve = RoasCurve(use_cached_data=True)
curve.plot_all_raw_data()
curve.calculate_all_curves()
# curve.get_roas_weighted_df()

# for cohort in roas_df['install_cohort'].unique():
#     filtered_df = roas_df[roas_df['install_cohort'] == cohort].groupby("days_since_install").agg(
#         roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()
#     print(filtered_df.head(10))
#     X = filtered_df["days_since_install"].values
#     # X = X.reshape(1, -1)
#     y = filtered_df["roas"].values
#     # y = y.reshape(1, -1)
#     break
