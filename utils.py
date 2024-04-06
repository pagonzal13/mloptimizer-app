from mloptimizer.hyperparams import Hyperparam, HyperparameterSpace
from mloptimizer.genoptimizer import SklearnOptimizer
from sklearn.datasets import load_iris
import streamlit as st
import pandas as pd
import time, os, sys, traceback
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
from watcher import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


class Utils:
    def __init__(self):
        self.target = ''
        self.algorithm = ''
        self.individuals = 10
        self.generations = 10
        self.x = [[]]
        self.y = []
        self.custom_hyperparams_dictionary = {}
        self.custom_fixed_hyperparams_dictionary = {}
        self.checkpoint = None
        self.custom_seed = 0
        self.inizialize_session_state_vars()

    def get_target(self):
        return self.target

    def set_target(self, target):
        self.target = target

    def format_class_name(self, class_name_option):
        partial_formatted_name = class_name_option.split("Classifier")[0]
        formatted_name = ""

        if partial_formatted_name.isupper():
            return partial_formatted_name

        prev_char_is_upper = False
        for char in partial_formatted_name:
            if char.isupper():
                # If the current character is uppercase and the previous one was too, it's considered part of the acronym
                if prev_char_is_upper:
                    formatted_name += char
                # If the current character is uppercase but the previous one wasn't, it's considered the start of a new word
                else:
                    if formatted_name:
                        formatted_name += " " + char
                    else:
                        formatted_name += char
                prev_char_is_upper = True
            else:
                formatted_name += char
                prev_char_is_upper = False

        return formatted_name

    def get_algorithm(self):
        return self.algorithm

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def get_individuals(self):
        return self.individuals

    def set_individuals(self, individuals):
        self.individuals = individuals

    def get_generations(self):
        return self.generations

    def set_generations(self, generations):
        self.generations = generations

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def get_custom_hyperparams_dictionary(self):
        return self.custom_hyperparams_dictionary

    def set_custom_hyperparams_dictionary(self, custom_hyperparams_dictionary):
        self.custom_hyperparams_dictionary = custom_hyperparams_dictionary

    def get_custom_fixed_hyperparams_dictionary(self):
        return self.custom_fixed_hyperparams_dictionary

    def set_custom_fixed_hyperparams_dictionary(self, custom_fixed_hyperparams_dictionary):
        self.custom_fixed_hyperparams_dictionary = custom_fixed_hyperparams_dictionary

    def delete_hyperparams_dictionaries(self):
        self.custom_hyperparams_dictionary = {}
        self.custom_fixed_hyperparams_dictionary = {}

    def get_checkpoint(self):
        return self.checkpoint

    def set_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint

    def get_custom_seed(self):
        return self.custom_seed

    def set_custom_seed(self, seed):
        self.custom_seed = seed

    def set_optimizer_data(self, optimizer):
        data = {
            "hyperparams_keys": optimizer.hyperparam_space.evolvable_hyperparams.keys(),
            "population_df": optimizer.population_2_df(),
            "logbook": optimizer.logbook
        }
        st.session_state.optimizer_data = data

    def get_optimizer_hyperparams_keys(self):
        return st.session_state.optimizer_data["hyperparams_keys"]

    def population_2_df(self):
        return st.session_state.optimizer_data["population_df"]

    def get_optimizer_logbook(self):
        return st.session_state.optimizer_data["logbook"]

    def get_dataframe(self):
        df = pd.DataFrame()

        hyperspace = HyperparameterSpace.get_default_hyperparameter_space(eval(self.algorithm))

        for param_name, param_obj in hyperspace.evolvable_hyperparams.items():
            scale = None
            if param_obj.hyperparam_type == "float":
                scale = param_obj.scale

            param_row = pd.DataFrame(
                {
                    'hyperparam': [param_name],
                    'hyperparam_type': [param_obj.hyperparam_type],
                    'scale': [scale],
                    'use fixed value': [False],
                    'fixed value': [None],
                    'range min': [param_obj.min_value],
                    'range max': [param_obj.max_value]
                }
            )
            df = pd.concat([df, param_row])
        return df

    def get_param_type(self, param):
        if param == "int":
            return int
        elif param == "float":
            return float
        else:
            return param

    def set_custom_hyperparams(self, fixed_rows, range_rows):
        for i in range(len(fixed_rows)):
            self.custom_fixed_hyperparams_dictionary[fixed_rows.iloc[i]["hyperparam"]] = fixed_rows.iloc[i][
                "fixed value"]

        for i in range(len(range_rows)):
            param_name = range_rows.iloc[i]["hyperparam"]
            # param_type = self.get_param_type(range_rows.iloc[i]["hyperparam_type"])
            param_type = range_rows.iloc[i]["hyperparam_type"]
            param_min = range_rows.iloc[i]["range min"]
            param_max = range_rows.iloc[i]["range max"]
            param_denominator = range_rows.iloc[i]["scale"]

            param = Hyperparam(param_name, param_min, param_max, param_type, param_denominator)

            self.custom_hyperparams_dictionary[param_name] = param

    def optimize(self, optimizer):
        try:
            optimizer.optimize_clf(self.individuals, self.generations, self.checkpoint)
        except Exception as err:
            st.error(
                'Oops...sorry, something didn\'t go as expected. Please, check your input data (read correspondent '
                'algorithm doc) and selected hyperparams)',
                icon="🚨")
            name = type(err).__name__
            st.error(name + ': ' + str(err))
        else:
            st.success('Optimization has been successfully generated!', icon="✅")
            self.set_session_state_results_vars(
                last_population_path_param=os.path.join(optimizer.tracker.results_path, "populations.csv"),
                last_logbook_path_param=os.path.join(optimizer.tracker.results_path, "logbook.csv"),
                show_results_param=True
            )

    def genetic_status_bar(self, progress_path):
        bar_gen = st.progress(0, 'Generation 0')
        bar_indi = st.progress(0, 'Individual 0')

        watch = Watcher(generations=self.generations, individuals=self.individuals)
        watch.run(watched_dir=progress_path, gen_progress_bar=bar_gen, indi_progress_bar=bar_indi)

    def execute(self):
        # optimizer = eval(
        #     self.algorithm + '(self.x, self.y, custom_hyperparams=self.custom_hyperparams_diccionary, '
        #                      'custom_fixed_hyperparams=self.custom_fixed_hyperparams_diccionary, '
        #                      'seed=self.custom_seed)')
        hyperspace_ex = HyperparameterSpace(evolvable_hyperparams=self.custom_hyperparams_dictionary,
                                            fixed_hyperparams=self.custom_fixed_hyperparams_dictionary)
        print(hyperspace_ex)
        optimizer = SklearnOptimizer(clf_class=eval(self.algorithm),
                                     hyperparam_space=hyperspace_ex,
                                     features=self.x, labels=self.y, seed=self.custom_seed
                                     )

        thread_1 = Thread(target=self.optimize, args=[optimizer])
        add_script_run_ctx(thread_1)
        thread_1.start()

        time.sleep(0.1)

        self.genetic_status_bar(os.path.join(optimizer.tracker.progress_path))

        thread_1.join()

        self.set_optimizer_data(optimizer=optimizer)

    def download_files(self, population_path='', logbook_path=''):
        if population_path != '':
            with open(population_path) as file:
                btn_p = st.download_button(
                    label="Download populations.csv",
                    data=file,
                    file_name="populations.csv",
                    mime="text/csv"
                )
        if logbook_path != '':
            with open(logbook_path) as file:
                btn_l = st.download_button(
                    label="Download logbook.csv",
                    data=file,
                    file_name="logbook.csv",
                    mime="text/csv"
                )

    def inizialize_session_state_vars(self):
        if "optimizer_data" not in st.session_state:
            st.session_state["optimizer_data"] = None

        if "input_data_frame" not in st.session_state:
            st.session_state["input_data_frame"] = None

        if "last_population_path" not in st.session_state:
            st.session_state["last_population_path"] = ''

        if "last_logbook_path" not in st.session_state:
            st.session_state["last_logbook_path"] = ''

        if "show_results" not in st.session_state:
            st.session_state["show_results"] = False

    def restart_session_state_vars(self):
        st.session_state.optimizer_data = None
        st.session_state.input_data_frame = None
        st.session_state.last_population_path = ''
        st.session_state.last_logbook_path = ''
        st.session_state.show_results = False

    def set_session_state_results_vars(self, last_population_path_param='', last_logbook_path_param='',
                                       show_results_param=False):
        st.session_state.last_population_path = last_population_path_param
        st.session_state.last_logbook_path = last_logbook_path_param
        st.session_state.show_results = show_results_param

    def set_input_data_frame(self, input_data_frame):
        st.session_state.input_data_frame = input_data_frame
