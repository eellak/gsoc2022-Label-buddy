FROM python:3.10-slim

RUN apt-get update
RUN apt install -y build-essential
RUN apt-get -y install python3-dev
RUN apt-get install -y libsndfile1-dev

RUN mkdir -p ./audios

COPY api.py ./api.py
COPY requirements.txt ./requirements.txt
# COPY wsgi.ini ./wsgi.ini

RUN pip3 install -r requirements.txt

CMD [ "python3", "-W ignore" ,"api.py"]
# CMD ["uwsgi", "wsgi.ini"]
