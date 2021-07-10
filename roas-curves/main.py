import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date, timedelta
from pull_data import load_local_data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plt.interactive(False)
plt.figure(figsize=(20, 12))
plt.close("all")

def plot_roas():

     targets_start = pd.Timestamp(date(2021, 3, 24))  # Cohorting will start from this date

     spend_df = load_local_data("spend.feather")
     spend_df.rename({'date': 'install_date'}, axis=1, inplace=True)

     revenue_df = load_local_data("revenue.feather")
     revenue_df.rename({'first_contact_day': 'install_date'}, axis=1, inplace=True)
     revenue_df["days_since_install"] = (revenue_df["clean_date"] - revenue_df["install_date"]).dt.days

     roas_df = revenue_df.groupby("install_date").agg(installs=pd.NamedAgg(column="teamid", aggfunc="nunique")).reset_index()
     roas_df = pd.merge(roas_df, spend_df, how="inner", on=["install_date"])
     roas_df["cpi"] = roas_df["spend"] / roas_df["installs"]
     roas_df = pd.merge(roas_df, revenue_df, how="inner", on=["install_date"])

     """Assign each day to a weekly cohort, with the cohorts starting on day 1"""
     offset_week_day = 7 - targets_start.weekday()
     roas_df["offset_install_date"] = roas_df['install_date'] + timedelta(days=offset_week_day)
     roas_df['install_cohort'] = roas_df['offset_install_date'].dt.to_period('W').apply(lambda r: r.start_time)
     roas_df["install_cohort"] = roas_df['install_cohort'] - timedelta(days=offset_week_day)

     roas_df = roas_df[roas_df["installs"] > 200]

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

     roas_df["install_week"] = (roas_df['install_date'] - targets_start) // np.timedelta64(1, 'W')
     roas_df["weeks_since_install"] = roas_df["days_since_install"] // 7

     print(roas_df.head(200))

     sns.lineplot(data=roas_df, x='days_since_install', y='roas', hue='install_cohort')
     plt.show()

     roas_df.to_feather("roas_df.feather")


# plot_roas()
roas_df = load_local_data("roas_df.feather")

import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

print(roas_df.head(200))


def objective(x, a, b, c, d):
     return a * np.sin(b - x) + c * x ** 2 + d


for cohort in roas_df['install_cohort'].unique():
     filtered_df = roas_df[roas_df['install_cohort'] == cohort].groupby("days_since_install").agg(roas=pd.NamedAgg(column="roas", aggfunc=np.mean)).reset_index()

     x = filtered_df["days_since_install"]
     y = filtered_df["roas"]

     popt, _ = curve_fit(objective, x, y)
     # summarize the parameter values
     a, b, c, d = popt
     print(popt)
     # plot input vs output
     plt.scatter(x, y)
     # define a sequence of inputs between the smallest and largest known inputs
     x_line = np.arange(min(x), max(x), 1)
     # calculate the output for the range
     y_line = objective(x_line, a, b, c, d)
     # create a line plot for the mapping function
     plt.plot(x_line, y_line, '--', color='red')
     plt.title(cohort)
     plt.show()
