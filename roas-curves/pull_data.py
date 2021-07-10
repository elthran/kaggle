import pandas as pd

from sqlalchemy import create_engine


def pull_data_from_redshift():
    uri = "postgresql://{username}:{password}@{hostname}:{port}/{database}".format(
        username='jbrunner',
        password='Starcraft$2',
        hostname='hothead-analytics-db-ds2.c2ukcdjmst8c.us-east-1.redshift.amazonaws.com',
        port=5439,
        database='prod'
    )
    engine = create_engine(uri, isolation_level="AUTOCOMMIT")
    redshift_engine = engine.connect()

    targets_start = '2021-03-23'
    targets_end = '2021-07-1'

    user_summary_query = f"""SELECT DISTINCT
                        teamid, 
                        first_contact_day
                    FROM streetrace_prod_user_summary
                    WHERE first_contact_day >= '{targets_start}' and first_contact_day < '{targets_end}'
                    """
    user_summary = pd.read_sql(user_summary_query, redshift_engine)
    user_summary.to_feather("user_summary.feather")

    spend_query = f"""
                SELECT 
                    date, 
                    sum(spend) as spend
                FROM streetrace_prod_ua_campaign_summary
                WHERE date >= '{targets_start}' and date < '{targets_end}'
                GROUP BY 1
                ORDER BY 1
                """
    spend = pd.read_sql(spend_query, redshift_engine)
    spend.to_feather("spend.feather")

    revenue_query = f"""
            SELECT 
                us.teamid,
                first_contact_day, 
                clean_date, 
                datediff(DAY,'{targets_start}',first_contact_day) as max_age,
                sum(iap_revenue_day) as day_iap_revenue, 
                sum(tapjoy_revenue_day) as day_tapjoy_revenue, 
                sum(ironsource_revenue_day) as day_ad_revenue
            FROM streetrace_prod_dau d
            INNER JOIN (
                {user_summary_query}
            ) us
            on d.teamid = us.teamid
            WHERE d.clean_date >= us.first_contact_day and (clean_date::date <= current_date - interval '1 day')
            GROUP BY 1,2,3 
        """
    revenue = pd.read_sql(revenue_query, redshift_engine)
    revenue.to_feather("revenue.feather")


def load_local_data(filename):
    return pd.read_feather(filename)


pull_data_from_redshift()
