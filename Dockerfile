FROM python:3.8.17

WORKDIR /mloptimizer

COPY . /mloptimizer

#RUN pip install -r /mloptimizer/requirements.txt
#RUN pip install --upgrade streamlit

#CMD [ "streamlit", "run", "app.py" ]