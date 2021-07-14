import pandas as pd

from sqlalchemy import create_engine


class RedshiftHelper:
    def __init__(self):
        self.redshift_engine = self.get_redshift_connection()
        self.user_summary_query = f"""
                SELECT DISTINCT teamid, 
                                first_contact_day
                FROM streetrace_prod_user_summary
                WHERE first_contact_day >= '{self.earliest_date}' and first_contact_day < '{self.end_date}'
                """
        self.spend_query = f"""
                SELECT date as install_date, 
                       sum(spend) as spend
                FROM streetrace_prod_ua_campaign_summary
                WHERE date >= '{self.earliest_date}' and date < '{self.end_date}'
                GROUP BY 1
                ORDER BY 1
                """
        self.revenue_query = f"""
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

    def get_redshift_connection():
        uri = "postgresql://{username}:{password}@{hostname}:{port}/{database}".format(
            username='jbrunner',
            password='Starcraft$2',
            hostname='hothead-analytics-db-ds2.c2ukcdjmst8c.us-east-1.redshift.amazonaws.com',
            port=5439,
            database='prod'
        )
        engine = create_engine(uri, isolation_level="AUTOCOMMIT")
        redshift_engine = engine.connect()
        return redshift_engine

    def get_user_summary_df(self):
        return pd.read_sql(self.user_summary_query, self.redshift_engine)

    def get_spend_df(self):
        return pd.read_sql(self.spend_query, self.redshift_engine)

    def get_revenue_df(self):
        return pd.read_sql(self.revenue_query, self.redshift_engine)
