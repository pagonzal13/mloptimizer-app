FROM python:3.8.17

WORKDIR /mloptimizer

COPY . /mloptimizer

RUN pip install -r /mloptimizer/requirements.txt

EXPOSE 8501

CMD [ "streamlit", "run", "app.py" ]
