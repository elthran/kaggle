import pandas as pd

from sqlalchemy import create_engine


class RedshiftHelper:
    def __init__(self, config):
        self.redshift_engine = None

        self.game_name = config['game_name']
        self.earliest_date = config['start_date']
        self.end_date = config['end_date']

        if config.get('country') and config['country'].lower() == 'us':
            country_filter = f"AND country = 'US'"
        elif config.get('country') and config['country'].lower() == 'row':
            country_filter = f"AND country != 'US'"
        else:
            country_filter = ""
        network_filter = f"AND LOWER(tracker_network) = '{config['network'].lower()}'" if config.get('network') else ""
        platform_filter = f"LOWER(CASE when platform ilike 'ip%%' then 'iOS' else 'Android' END) = LOWER('{config['platform'].lower()}')" if \
            config.get('platform') else ""
        campaign_filter = f"LOWER(tracker_campaign_type) = '{config['campaign'].lower()}'" if config.get(
            'campaign') else ""
        facebook_campaign_filter = f"LOWER(adset_name) LIKE '%%{config['facebook_campaign'].lower()}%%'" if config.get(
            'facebook_campaign') else ""

        self.user_summary_filter_clause = country_filter + network_filter + platform_filter + campaign_filter + facebook_campaign_filter

        self.user_summary_query = f"""
SELECT DISTINCT teamid
       ,first_contact_day
       ,CASE
         WHEN campaign_summary = 'Organic' THEN 'Organic'
         ELSE 'UA'
       END AS ua_type
FROM {self.game_name}_prod_user_summary
WHERE first_contact_day BETWEEN '{self.earliest_date}' AND '{self.end_date}'
{self.user_summary_filter_clause}
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

    def get_organic_lift_df(self):
        return pd.read_sql(self.organic_lift_query, self.redshift_engine)
