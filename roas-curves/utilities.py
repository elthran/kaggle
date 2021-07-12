from sqlalchemy import create_engine


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