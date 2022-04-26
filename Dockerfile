FROM continuumio/anaconda3


COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY train_data.csv ./train_data.csv
COPY test_data.csv ./test_data.csv
COPY movie_meta.csv ./movie_meta.csv

COPY model_and_helpers.py ./model_and_helpers.py
COPY training.py ./training.py