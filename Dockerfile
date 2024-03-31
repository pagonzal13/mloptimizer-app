FROM python:3.10.0-slim

WORKDIR /mloptimizer

COPY . /mloptimizer

RUN pip install -r /mloptimizer/requirements.txt

EXPOSE 8501

CMD [ "streamlit", "run", "app.py" ]
