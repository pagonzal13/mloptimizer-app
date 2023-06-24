from mloptimizer.genoptimizer import *
from sklearn.datasets import load_iris
import streamlit as st
import pandas as pd
import time, os, sys, traceback

target = ''
algorithm = 'TreeOptimizer'
individuals = 10
generations = 10
x = [[]]
y = []
custom_params_diccionary = {}
custom_fixed_params_diccionary = {}
checkpoint = None

def get_dataframe(algorithm):
    df = pd.DataFrame()

    for param_name, param_obj in eval(algorithm).get_default_params().items():
        denominator = None
        if param_obj.type.__name__ == "float":
            denominator = param_obj.denominator

        param_row = pd.DataFrame(
                {
                    'hyper-param': [param_obj.name],
                    'type': [param_obj.type.__name__],
                    'use fixed': [False],
                    'fixed value': [None],
                    'range min': [param_obj.min_value],
                    'range max': [param_obj.max_value],
                    'denominator': [denominator]
                }
            )
        df = pd.concat([df, param_row])
    return df

def get_param_type(param):
        if param == "int":
            return int
        elif param == "float":
            return float
        else:
            return param

def set_custom_params(fixed_rows, range_rows):
    for i in range(len(fixed_rows)):
        custom_fixed_params_diccionary[fixed_rows.iloc[i]["hyper-param"]] = fixed_rows.iloc[i]["fixed value"]

    for i in range(len(range_rows)):
        param_name = range_rows.iloc[i]["hyper-param"]
        param_type = get_param_type(range_rows.iloc[i]["type"])
        param_min = range_rows.iloc[i]["range min"]
        param_max = range_rows.iloc[i]["range max"]
        param_denominator = range_rows.iloc[i]["denominator"]

        param = Param(param_name, param_min, param_max, param_type, param_denominator)

        custom_params_diccionary[param_name] = param

def execute():
    latest_generation = st.empty()
    bar_gen = st.progress(0)

    latest_individual = st.empty()
    bar_indi = st.progress(0)

    try:
        #TO DO: paralellize both progress bars and optimizer execution
        optimizer = eval(algorithm+'(x, y, custom_params=custom_params_diccionary, custom_fixed_params=custom_fixed_params_diccionary)')
        optimizer.optimize_clf(individuals, generations, checkpoint)

        #TO DO: reflect csv generation progress
        for i in range(generations):
            latest_generation.text(f'Generation {i+1}')
            bar_gen.progress(int(100*(i+1)/generations))
            time.sleep(0.5)

        for i in range(individuals):
            latest_individual.text(f'Individual {i+1}')
            bar_indi.progress(int(100*(i+1)/individuals))
            time.sleep(0.1)
        
        download_files(optimizer)

    except Exception as err:
        st.error('Oops...Caparrini has to work more (but maybe you should check your input data, selected target, amount of individuals and generations...)', icon="🚨")
        name = type(err).__name__
        st.error(name + ': ' + str(err))

    else:
        st.success('Optimization has been successfully generated!', icon="✅")
    
    finally:
        st.button('Try again!', key = 'restart_btn')

    return

def download_files(optimizer):
    population_path = os.path.join(optimizer.results_path, "populations.csv")
    logbook_path = os.path.join(optimizer.results_path, "logbook.csv")
    with open(population_path) as file:
        btn_p = st.download_button(
                label="Download populations.csv",
                data=file,
                file_name="populations.csv",
                mime="text/csv"
            )
    with open(logbook_path) as file:
        btn_l = st.download_button(
                label="Download logbook.csv",
                data=file,
                file_name="logbook.csv",
                mime="text/csv"
            )

st.info('Pay attention to the quality of your input data (column names, types of values, consistency, etc).', icon="ℹ️")
input_csv_file = st.file_uploader("Upload your input file")

if input_csv_file is not None:
    df = pd.read_csv(input_csv_file)
    st.data_editor(df, use_container_width=True)
    
    target = st.selectbox(
        'Which column do you want to use as target?',
        df.columns)

    y = df[target]
    x = df.drop(target, axis=1)

    algorithm = st.radio(
        "Which algorithm would you like to use?",
        ('TreeOptimizer',
            'ForestOptimizer',
            'ExtraTreesOptimizer',
            'GradientBoostingOptimizer',
            'XGBClassifierOptimizer',
            'CatBoostClassifierOptimizer',
            'KerasClassifierOptimizer',
            'SVCOptimizer'))
    
    use_custom_params = st.checkbox('Use custom params')

    if use_custom_params:
        st.write("Edit the table below with the hyper-params values you want")
        st.info("By default, parameters use ranges (they are not fixed). You can mark 'fixed' column of the parameters you want to set with a fixed value and set it in corresponding column.", icon="🤓")

        edited_df = st.data_editor(
            get_dataframe(algorithm),
            hide_index=True,
            use_container_width=True,
            column_config={
                "fixed value": st.column_config.NumberColumn(),
                "range min": st.column_config.NumberColumn(),
                "range max": st.column_config.NumberColumn(),
                "denominator": st.column_config.NumberColumn()
            },
            disabled=("hyper-param", "type")
        )
        
        fixed_rows = edited_df.loc[edited_df["use fixed"] == True]
        range_rows = edited_df.loc[edited_df["use fixed"] == False]

        set_custom_params(fixed_rows, range_rows)

    else:
        custom_params_diccionary = {}
        custom_fixed_params_diccionary = {}
        
    individuals = st.select_slider(
        'Select the amount of individuals',
        range(2, 101),
        value = individuals)
    generations = st.select_slider(
        'Select the amount of generations',
        range(2, 101),
        value = generations)

    if st.button('Start execution', key = 'start_btn', disabled=st.session_state.get("start_disabled", False)):
        execute()

    if st.session_state.get("start_btn", False):
        st.session_state.start_disabled = False
    elif st.session_state.get("restart_btn", False):
        st.session_state.start_disabled = True