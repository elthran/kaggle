import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date, timedelta
from scipy.optimize import curve_fit

from utilities import get_redshift_connection

# Set pandas options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set matplotlib options
plt.interactive(False)
plt.figure(figsize=(20, 12))


class RoasCurve:

    def __init__(self, min_num_installs=200, use_cached_data=False, load_plots=False):
        self.earliest_date = '2021-03-24'
        self.end_date = '2021-07-01'
        self.min_num_installs = min_num_installs
        self.roas_df = self.load_roas_data_locally() if use_cached_data else self.pull_roas_data_from_redshift()

    def pull_roas_data_from_redshift(self):
        redshift_engine = get_redshift_connection()

        user_summary_query = f"""
        SELECT DISTINCT teamid, 
                        first_contact_day
        FROM streetrace_prod_user_summary
        WHERE first_contact_day >= '{self.earliest_date}' and first_contact_day < '{self.end_date}'
        """
        user_summary_df = pd.read_sql(user_summary_query, redshift_engine)

        spend_query = f"""
        SELECT date as install_date, 
               sum(spend) as spend
        FROM streetrace_prod_ua_campaign_summary
        WHERE date >= '{self.earliest_date}' and date < '{self.end_date}'
        GROUP BY 1
        ORDER BY 1
        """
        spend_df = pd.read_sql(spend_query, redshift_engine)

        revenue_query = f"""
        SELECT us.teamid,
               first_contact_day as install_date, 
               clean_date,
               datediff(DAY,'{self.earliest_date}',first_contact_day) as max_age,
               sum(iap_revenue_day) as day_iap_revenue, 
               sum(tapjoy_revenue_day) as day_tapjoy_revenue, 
               sum(ironsource_revenue_day) as day_ad_revenue
        FROM streetrace_prod_dau d
        INNER JOIN ({user_summary_query}) us on d.teamid = us.teamid
        WHERE d.clean_date >= us.first_contact_day and (clean_date::date <= current_date - interval '1 day')
        GROUP BY 1,2,3
        """
        revenue_df = pd.read_sql(revenue_query, redshift_engine)

        start_date = pd.Timestamp(date(int(self.earliest_date[:4]),
                                       int(self.earliest_date[5:7]),
                                       int(self.earliest_date[8:])))

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

        roas_df["install_week"] = (roas_df['install_date'] - start_date) // np.timedelta64(1, 'W')

        roas_df.to_feather("roas_df.feather")
        return roas_df

    @staticmethod
    def load_roas_data_locally():
        return pd.read_feather("roas_df.feather")

    def plot_all_raw_data(self):
        sns.lineplot(data=self.roas_df, x='days_since_install', y='roas', hue='install_cohort')
        plt.show()

    @staticmethod
    def first_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    def calculate_curve(self):
        self.X = self.roas_df["days_since_install"].values
        self.y = self.roas_df["roas"].values

        sns.lineplot(x=self.X, y=self.y, data=self.roas_df, hue='install_cohort', ci=False)

        aggregated_df = self.roas_df.groupby("days_since_install").agg(
            roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()
        aggregated_X = aggregated_df["days_since_install"].values
        aggregated_y = aggregated_df["roas"].values
        popt, pcov = curve_fit(self.first_func, aggregated_X, aggregated_y)
        y_pred = self.first_func(aggregated_X, *popt)
        plt.plot(aggregated_X, y_pred, color='black', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        self.calculate_error(popt, aggregated_y, y_pred)
        plt.show()

    def calculate_error(self, popt, aggregated_y, y_pred):
        absError = y_pred - aggregated_y
        SE = np.square(absError)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(aggregated_y))

        print('Parameters:', popt)
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)


curve = RoasCurve(use_cached_data=True)
curve.plot_all_raw_data()
curve.calculate_curve()

# for cohort in roas_df['install_cohort'].unique():
#     filtered_df = roas_df[roas_df['install_cohort'] == cohort].groupby("days_since_install").agg(
#         roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()
#     print(filtered_df.head(10))
#     X = filtered_df["days_since_install"].values
#     # X = X.reshape(1, -1)
#     y = filtered_df["roas"].values
#     # y = y.reshape(1, -1)
#     break
