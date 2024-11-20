FROM python:3.9-slim
# FROM --platform=linux/amd64 python:3.9.14-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl 

COPY requirements.txt /tmp/requirements.txt
# RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY ./app /app
WORKDIR /app
EXPOSE 80
# RUN mkdir ~/.streamlit
# RUN cp config.toml ~/.streamlit/config.toml
# RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]
# ENTRYPOINT ["streamlit", "run"]
# CMD ["main.py"]
# EXPOSE 8501
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]





