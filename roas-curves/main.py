from roas_curves import RoasCurve

if __name__ == '__main__':
    """To run, simply set config to:
    config = {'start_date': 
    """
    config = {
        'game_name': 'streetrace',
        'start_date': '2021-03-24',
        'end_date': '2021-07-01',

        # Do not set the parameters below unless instructed
        'data_blend': 'median',
        'data_function': None,
        'max_train_age': None
    }

    curve = RoasCurve(use_cached_data=True, config=config)
