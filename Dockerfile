FROM python:3.7

COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN python3 -m pip install -U pip && pip install -r /tmp/requirements.txt

RUN mkdir /app

COPY app.py /app

WORKDIR /app

CMD streamlit run app.py  --server.port $PORT