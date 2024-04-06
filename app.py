from mloptimizer.genoptimizer import SklearnOptimizer
from mloptimizer.plots import plotly_search_space, plotly_logbook
from mloptimizer.alg_wrapper import CustomXGBClassifier
from mloptimizer.hyperparams import Hyperparam, HyperparameterSpace
from sklearn.datasets import load_iris
import streamlit as st
import pandas as pd
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
from watcher import *
from utils import *
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

###########################################################################################################################
############################################## MAIN FRONT-END ELEMENTS ####################################################
###########################################################################################################################

# Config
st.set_page_config(
    page_title="MLOptimizer UI",
    page_icon="💻",
    layout="wide"
)

# Title
st.header('MLOptimizer UI')
st.subheader('Find the best hyperparameters for training your data!')
st.divider()

# Initialization (Utils is class with methods to manage optimizer and editable variables)
utils = Utils()

###########################################################################################################################

# Input file section
st.write("You can try MLOptimizer UI with a dummy example or start using it with your own input dataset")
use_custom_input = st.toggle('Try with our example')

if use_custom_input:
    utils.restart_session_state_vars()
    with open('data/iris.csv', "r") as iris_file:
        df = pd.read_csv(iris_file)
        utils.set_input_data_frame(input_data_frame=df)
        # Example file section - show data
        with st.expander("Take a look at input data"):
            st.dataframe(df, use_container_width=True)
else:
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        input_csv_file = st.file_uploader("Upload your input file", type='csv',
                                          help=':information_source: Pay attention to the quality of your input data '
                                               '(column names, types of values, consistency, etc).')
        if input_csv_file is not None:
            # Input file section - data editor
            df = pd.read_csv(input_csv_file)
            utils.set_input_data_frame(input_data_frame=df)
            with col2:
                with st.expander("Review and/or edit your data"):
                    st.data_editor(df, use_container_width=True)
        else:
            utils.restart_session_state_vars()

st.divider()

# #####################################################################################################################

if st.session_state.input_data_frame is not None:
    # Editable variables section
    target_tab, algorithm_tab, genetic_params_tab = st.tabs(["Target", "Algorithm", "Genetic params"])

    # Editable variables section - target
    with target_tab:
        col1, col2 = st.columns(2)

        with col1:
            target = st.selectbox(
                'Which column do you want to use as target?',
                df.columns)
            utils.set_target(target=target)

            utils.set_y(y=df[target])
            utils.set_x(x=df.drop(target, axis=1))

    # Editable variables section - algorithm
    with algorithm_tab:
        col1, col2 = st.columns([0.3, 0.7])

        # Get available algorithms from mloptimizer library
        with open(HyperparameterSpace.default_hyperparameter_spaces_json, 'r') as file:
            default_hyperparams = json.load(file)
        optimizer_class_list = [globals()[x] for x in default_hyperparams.keys()]

        with open('data/classifiers_external_doc.json', 'r') as file:
            classifier_links = json.load(file)

        optimizer_class_name_list = []
        optimizer_docu_list = []
        for optimizer_item in optimizer_class_list:
            class_name = optimizer_item.__name__
            optimizer_class_name_list.append(class_name)
            if class_name in classifier_links:
                link = classifier_links[class_name]
                if link:
                    optimizer_docu_list.append(f"learn more about [{utils.format_class_name(class_name_option=class_name)}]({link})")
                else:
                    optimizer_docu_list.append(f"explore other sources of information to learn more about {utils.format_class_name(class_name_option=class_name)}")
            else:
                optimizer_docu_list.append(f"explore other sources of information to learn more about {utils.format_class_name(class_name_option=class_name)}")

        # Select algorithm
        with col1:
            algorithm = st.radio(
                label="Which algorithm would you like to use?",
                options=optimizer_class_name_list,
                captions=optimizer_docu_list,
                format_func=lambda option: utils.format_class_name(class_name_option=option)
            )
            utils.set_algorithm(algorithm=algorithm)

        # Algorithm hyperparameters data editor
        with col2:
            use_custom_hyperparams = st.toggle('Use custom hyperparams')

            if use_custom_hyperparams:
                st.write("Edit the table below with the hyperparams values you want")
                st.info(
                    "By default, parameters use ranges (they are not fixed). You can mark 'fixed' column of the "
                    "parameters you want to set with a fixed value and set it in corresponding column.",
                    icon="🤓")

                edited_df = st.data_editor(
                    utils.get_dataframe(),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "fixed value": st.column_config.NumberColumn(),
                        "range min": st.column_config.NumberColumn(),
                        "range max": st.column_config.NumberColumn(),
                        "denominator": st.column_config.NumberColumn(
                            label="denominator 📎",
                            help="Denominator value to divide the hyperparameter value by. It applies only when the "
                                 "'type' column is 'float'. If 'type' is 'int', this value should be 'None' as it "
                                 "does not apply."
                        )
                    },
                    disabled=("hyperparam", "type")
                )

                fixed_rows = edited_df.loc[edited_df["use fixed"] == True]
                range_rows = edited_df.loc[edited_df["use fixed"] == False]

                utils.set_custom_hyperparams(fixed_rows=fixed_rows, range_rows=range_rows)
            else:
                utils.delete_hyperparams_dictionaries()

    # Editable variables section - genetic params
    with genetic_params_tab:
        col1, col2, col3 = st.columns([0.5, 0.3, 0.2])

        # Select amount of individuals and generation
        with col1:
            individuals = st.select_slider(
                'Select the amount of individuals',
                range(2, 101),
                value=utils.get_individuals())
            utils.set_individuals(individuals=individuals)

            generations = st.select_slider(
                'Select the amount of generations',
                range(2, 101),
                value=utils.get_generations())
            utils.set_generations(generations=generations)

        # Customizable value of random seed
        with col2:
            use_custom_seed = st.toggle('Set custom Python Random seed')

            if use_custom_seed:
                custom_seed = st.number_input(
                    label='Insert the value you want to initialize the random number generator in Python (seed):',
                    min_value=0,
                    value=1,
                    step=1,
                    format="%d")
                utils.set_custom_seed(seed=custom_seed)
            else:
                utils.set_custom_seed(seed=0)

    st.divider()

    ###########################################################################################################################

    # Restart state variables and execute
    if st.button('Start new execution'):
        utils.restart_session_state_vars()
        utils.execute()

    ###########################################################################################################################

    # Results section
    if st.session_state.show_results is not False:
        st.write("Take a look at the optimization results below")
        population_tab, evolution_tab, search_space_tab = st.tabs(["Population", "Evolution", "Search Space"])

        # Set variables needed by mloptimizer library to generate graphics
        optimizer_param_names = list(utils.get_optimizer_hyperparams_keys())
        optimizer_param_names.append("fitness")
        population_df = utils.population_2_df()

        # Results section - population: show and provide for downloading population resulting file
        with population_tab:
            with open(st.session_state.last_population_path) as file:
                utils.download_files(population_path=st.session_state.last_population_path)

                df_output = pd.read_csv(file)
                st.dataframe(data=df_output, height=350, use_container_width=True)
                file.seek(0)

        # Results section - evolution: show evolution graphic and provide logbook resulting file for downloading
        with evolution_tab:
            with open(st.session_state.last_logbook_path) as file:
                utils.download_files(logbook_path=st.session_state.last_logbook_path)

                logbook_graphic = plotly_logbook(utils.get_optimizer_logbook(), population_df)
                st.plotly_chart(logbook_graphic, use_container_width=True)

        # Results section - search space: show search space graphic
        with search_space_tab:
            dfp = population_df[optimizer_param_names]
            search_space_graphic = plotly_search_space(dfp)
            st.plotly_chart(search_space_graphic, use_container_width=True)

#######################################################################################################################
