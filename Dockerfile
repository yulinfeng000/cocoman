FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

EXPOSE 8899

ENV MINIO_URL="localhost:9000"
ENV MINIO_ACCESS_KEY="tomcat"
ENV MINIO_SECRET_KEY="tomcatisroot"
ENV MINIO_SSL='false'
ENV MONGO_DB_URL="mongodb://root:password@localhost:27017/mycoco?authSource=admin"
ENV MONGO_DB_NAME='mycoco'
ENV APP_MODULE='cocoman.server.http_server:app'

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y python3-opencv && pip install -r ./requirements.txt && python setup.py install
