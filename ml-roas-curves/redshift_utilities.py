import pandas as pd

from sqlalchemy import create_engine


class RedshiftHelper:
    def __init__(self, game, start_date, end_date):
        self.redshift_engine = None

        self.game_name = game
        self.earliest_date = start_date
        self.end_date = end_date

        self.user_summary_query = f"""
SELECT DISTINCT teamid
       ,first_contact_day
       ,retention1
       ,retention7
       ,case when country = 'US' then 1 else 0 end as is_us
       ,case when country != 'US' then 1 else 0 end as is_row
       ,CASE when platform ilike 'ip%%' then 1 else 0 end as is_ios
       ,CASE when platform not ilike 'ip%%' then 1 else 0 end as is_android
       ,CASE
         WHEN campaign_summary = 'Organic' THEN 'Organic'
         ELSE 'UA'
       END AS ua_type
FROM {self.game_name}_prod_user_summary
WHERE first_contact_day BETWEEN '{self.earliest_date}' AND '{self.end_date}'
                """

        self.spend_query = f"""
SELECT DATE AS install_date
       ,SUM(spend) AS spend
FROM {self.game_name}_prod_ua_campaign_summary
WHERE DATE BETWEEN '{self.earliest_date}' AND '{self.end_date}'
GROUP BY 1
ORDER BY 1
                """

        self.revenue_query = f"""
SELECT us.teamid
       ,first_contact_day AS install_date
       ,clean_date
       ,datediff(DAY, '{self.earliest_date}', first_contact_day) AS max_age
       ,SUM(iap_revenue_day) AS day_iap_revenue
       ,SUM(tapjoy_revenue_day) AS day_tapjoy_revenue
       ,SUM(ironsource_revenue_day) AS day_ad_revenue
FROM {self.game_name}_prod_dau d
  INNER JOIN ({self.user_summary_query}) us ON d.teamid = us.teamid
WHERE clean_date::DATE BETWEEN us.first_contact_day AND '{self.end_date}'
GROUP BY 1
         ,2
         ,3
                """

    def set_redshift_connection(self):
        uri = "postgresql://{username}:{password}@{hostname}:{port}/{database}".format(
            username='jbrunner',
            password='Starcraft$2',
            hostname='hothead-analytics-db-ds2.c2ukcdjmst8c.us-east-1.redshift.amazonaws.com',
            port=5439,
            database='prod'
        )
        engine = create_engine(uri, isolation_level="AUTOCOMMIT")
        redshift_engine = engine.connect()
        self.redshift_engine = redshift_engine

    def get_user_summary_df(self):
        return pd.read_sql(self.user_summary_query, self.redshift_engine)

    def get_spend_df(self):
        return pd.read_sql(self.spend_query, self.redshift_engine)

    def get_revenue_df(self):
        return pd.read_sql(self.revenue_query, self.redshift_engine)
