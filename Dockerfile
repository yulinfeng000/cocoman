FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11


ENV MINIO_URL="localhost:9000"
ENV MINIO_ACCESS_KEY="tomcat"
ENV MINIO_SECRET_KEY="tomcatisroot"
ENV MINIO_SSL='false'
ENV MONGO_DB_URL="mongodb://root:password@localhost:27017/mycoco?authSource=admin"
ENV MONGO_DB_NAME='mycoco'
ENV APP_MODULE='cocoman.server.http_server:app'

WORKDIR /app

# Ensure the APT sources.list exists and modify it or create a new one
RUN if [ -f /etc/apt/sources.list ]; then \
    sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn\/debian-security/g' /etc/apt/sources.list; \
    else \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt 
RUN python setup.py install