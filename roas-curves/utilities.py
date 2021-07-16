import numpy as np

from tabulate import tabulate


class Utilities:
    def __init__(self):
        self.color_index_mapping = ["b", "g", "r", "y"]
        self.color_index = 0
        self.all_error_metrics = []

    def get_color(self, choice="random"):
        if choice == "black":
            color = "k"
        else:  # Choose a non-black color and cycle to index to next color in the list
            color = self.color_index_mapping[self.color_index]
            self.color_index = (self.color_index + 1) % len(self.color_index_mapping)
        return color

    @staticmethod
    def print_clearing_dashes():
        print(f"\n---------------------------------------------------------------------")

    def print_blend_message(self, datasets_to_fit):
        data_blend_legend = [dataset.get("data_name") for dataset in datasets_to_fit]
        self.print_clearing_dashes()
        print(f"To continue, please set parameter 'data_blend' equal to one of: \n{data_blend_legend}")

    def print_function_message(self, functions_to_fit):
        data_function_legend = [function_dict.get("name") for function_dict in functions_to_fit]
        self.print_clearing_dashes()
        print(f"\nTo continue, please set parameter 'data_function' equal to one of: \n{data_function_legend}")

    def print_train_age_message(self, max_train_ages):
        self.print_clearing_dashes()
        print(f"\nTo continue, please set parameter 'max_train_age' equal to one of: \n{max_train_ages}")

    def calculate_error_metrics(self,
                                function_name,
                                dataset_tested_on,
                                max_fit_age,
                                y_0_to_30,
                                y_pred_0_to_30,
                                y_30_plus,
                                y_pred_30_plus):

        absolute_error_0_to_30 = y_pred_0_to_30 - y_0_to_30
        squared_error_0_to_30 = np.square(absolute_error_0_to_30)
        mean_squared_error_0_to_30 = np.mean(squared_error_0_to_30)
        root_mean_squared_error_0_to_30 = np.sqrt(mean_squared_error_0_to_30)
        r_squared_error_0_to_30 = 1.0 - (np.var(absolute_error_0_to_30) / np.var(y_0_to_30))

        absolute_error_30_plus = y_pred_30_plus - y_30_plus
        squared_error_30_plus = np.square(absolute_error_30_plus)
        mean_squared_error_30_plus = np.mean(squared_error_30_plus)
        root_mean_squared_error_30_plus = np.sqrt(mean_squared_error_30_plus)
        r_squared_error_30_plus = 1.0 - (np.var(absolute_error_30_plus) / np.var(y_30_plus))

        self.append_error_metric({"function_name": function_name,
                                  "dataset_tested_on": dataset_tested_on,
                                  "max_age": max_fit_age,
                                  "absolute_error_0_to_30": absolute_error_0_to_30,
                                  "absolute_error_30_plus": absolute_error_30_plus,
                                  "squared_error_0_to_30": squared_error_0_to_30,
                                  "squared_error_30_plus": squared_error_30_plus,
                                  "mean_squared_error_0_to_30": mean_squared_error_0_to_30,
                                  "mean_squared_error_30_plus": mean_squared_error_30_plus,
                                  "root_mean_squared_error_0_to_30": root_mean_squared_error_0_to_30,
                                  "root_mean_squared_error_30_plus": root_mean_squared_error_30_plus,
                                  "r_squared_error_0_to_30": r_squared_error_0_to_30,
                                  "r_squared_error_30_plus": r_squared_error_30_plus})

    def append_error_metric(self, metric):
        self.all_error_metrics.append(metric)
        print(f"Completed {metric['function_name']} on {metric['dataset_tested_on']} for max_age {metric['max_age']}.")

    def print_final_error_metrics(self):
        sort_key = "mean_squared_error_0_to_30"
        relevant_keys = ["function_name",
                         "dataset_tested_on",
                         "max_age",
                         "mean_squared_error_0_to_30",
                         "mean_squared_error_30_plus",
                         "root_mean_squared_error_0_to_30",
                         "root_mean_squared_error_30_plus",
                         "r_squared_error_0_to_30",
                         "r_squared_error_30_plus"]
        clean_dicts = sorted([{key: dict[key] for key in relevant_keys} for dict in self.all_error_metrics],
                             key=lambda k: k[sort_key])
        header = clean_dicts[0].keys()
        rows = [x.values() for x in clean_dicts]
        print(f"Printing results sorted by {sort_key}")
        print(tabulate(rows, header, tablefmt='grid'))
