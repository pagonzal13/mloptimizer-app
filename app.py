from mloptimizer.genoptimizer import *
from sklearn.datasets import load_iris
import streamlit as st
import pandas as pd
import time, os, sys, traceback
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx

st.set_page_config(
    page_title="MLOptimizer",
    page_icon="💻",
    layout="wide"
)

st.header('MLOptimizer')
st.subheader('Find the best hyper-parameters for training your data!')
st.divider()

target = ''
algorithm = ''
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

def optimize(optimizer):
    try:
        optimizer.optimize_clf(individuals, generations, checkpoint)
    except Exception as err:
        st.error('Oops...Caparrini has to work more (but maybe you should check your input data, selected target, amount of individuals and generations...)', icon="🚨")
        name = type(err).__name__
        st.error(name + ': ' + str(err))
    else:
        st.success('Optimization has been successfully generated!', icon="✅")
        set_session_state_vars(
            last_population_path_param = os.path.join(optimizer.results_path, "populations.csv"),
            last_logbook_path_param = os.path.join(optimizer.results_path, "logbook.csv"),
            show_results_param = True
        )

def generations_status_bar():
    latest_generation = st.empty()
    bar_gen = st.progress(0)

    for i in range(generations):
        latest_generation.text(f'Generation {i+1}')
        bar_gen.progress(int(100*(i+1)/generations))
        time.sleep(0.5)

def individuals_status_bar():
    latest_individual = st.empty()
    bar_indi = st.progress(0)

    for i in range(individuals):
        latest_individual.text(f'Individual {i+1}')
        bar_indi.progress(int(100*(i+1)/individuals))
        time.sleep(0.1)

def execute():
    optimizer = eval(algorithm+'(x, y, custom_params=custom_params_diccionary, custom_fixed_params=custom_fixed_params_diccionary)')

    thread_1 = Thread(target=optimize, args=[optimizer])
    thread_2 = Thread(target=generations_status_bar)
    thread_3 = Thread(target=individuals_status_bar)

    add_script_run_ctx(thread_1)
    add_script_run_ctx(thread_2)
    add_script_run_ctx(thread_3)

    threads = [thread_1, thread_2, thread_3]

    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    #TO DO: reflect csv generation progress
    
    return optimizer

def download_files(population_path='', logbook_path=''):
    if population_path is not '':
        with open(population_path) as file:
            btn_p = st.download_button(
                    label="Download populations.csv",
                    data=file,
                    file_name="populations.csv",
                    mime="text/csv"
                )
    if logbook_path is not '':
        with open(logbook_path) as file:
            btn_l = st.download_button(
                    label="Download logbook.csv",
                    data=file,
                    file_name="logbook.csv",
                    mime="text/csv"
                )

def inizialize_session_state_vars():
    if "last_population_path" not in st.session_state:
        st.session_state["last_population_path"] = ''

    if "last_logbook_path" not in st.session_state:
        st.session_state["last_logbook_path"] = ''

    if "show_results" not in st.session_state:
        st.session_state["show_results"] = False

def restart_session_state_vars():
    st.session_state.last_population_path = ''
    st.session_state.last_logbook_path = ''
    st.session_state.show_results = False
    
def set_session_state_vars(last_population_path_param, last_logbook_path_param, show_results_param):
    st.session_state.last_population_path = last_population_path_param
    st.session_state.last_logbook_path = last_logbook_path_param
    st.session_state.show_results = show_results_param

col1, col2 = st.columns([0.3, 0.7])
with col1:
    input_csv_file = st.file_uploader("Upload your input file", type='csv', help=':information_source: Pay attention to the quality of your input data (column names, types of values, consistency, etc).')

if input_csv_file is not None:
    
    df = pd.read_csv(input_csv_file)
    with col2:
        with st.expander("Review and/or edit your data"):
            st.data_editor(df, use_container_width=True)

    st.divider()

    target, algorithm, genetic_params = st.tabs(["Target", "Algorithm", "Genetic params"])

    with target:
        col1, col2 = st.columns(2)

        with col1:
            target = st.selectbox(
            'Which column do you want to use as target?',
            df.columns)

            y = df[target]
            x = df.drop(target, axis=1)

    with algorithm:
        col1, col2 = st.columns([0.3, 0.7])

        with col1:
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
        
        with col2:
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

    with genetic_params:
        col1, col2 = st.columns(2)

        with col1:
            individuals = st.select_slider(
                'Select the amount of individuals',
                range(2, 101),
                value = individuals)
            generations = st.select_slider(
                'Select the amount of generations',
                range(2, 101),
                value = generations)
    
    inizialize_session_state_vars()

    st.divider()

    if st.button('Start new execution'):
        restart_session_state_vars()

        optimizer = execute()

    if st.session_state.show_results is not False:
        population, logbook = st.tabs(["Population", "LogBook"])

        with population:
            with open(st.session_state.last_population_path) as file:
                download_files(population_path=st.session_state.last_population_path)

                df_output = pd.read_csv(file)
                st.table(df_output)
                file.seek(0)
                
        with logbook:
            with open(st.session_state.last_logbook_path) as file:
                download_files(logbook_path=st.session_state.last_logbook_path)

                df_output = pd.read_csv(file, usecols=['avg','min','max'])
                #need to be rescaled by using altair charts
                st.line_chart(df_output)
    