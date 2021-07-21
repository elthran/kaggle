from roas_curves import RoasCurve

if __name__ == '__main__':
    """To run, simply set config to:
    config = {
        'game_name': game_name,
        'start_date': '2020-08-01',
        'end_date': '2021-06-18',
    }
    """
    config = {
        'game_name': 'hydrostone',
        'start_date': '2020-10-20',
        'end_date': '2021-01-19',

        # Do not set the parameters below unless instructed
        'data_blend': 'age-weighted',
        'data_function': 'generalized_logistic_function',
        'max_train_age': 14
    }
    curve = RoasCurve(use_cached_data=True, config=config)


# 5.5 minutes without cache
# 50 seconds with cache

# Graeme 3 minutes
