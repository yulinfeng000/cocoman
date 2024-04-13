import os

def cast_bool(obj):
    if isinstance(obj,bool):
        return obj
    if isinstance(obj,str):
        if obj.lower() == 'false':
            return False
    return bool(obj)

def config(key, default, cast=lambda x: x):
    return cast(os.getenv(key, default))


DB_URL = config("DB_URL", default="postgres:iamroot@10.8.0.17:5432/coco")
DB_POOL_SIZE = config("DB_POOL_SIZE",default=20,cast=int)
MINIO_URL = config("MINIO_URL", "10.8.0.17:9900")
MINIO_ACCESS_KEY = config("MINIO_ACCESS_KEY", "tomcat")
MINIO_SECRET_KEY = config("MINIO_SECRET_KEY", "tomcatisroot")
MINIO_SSL = config('"MINIO_SSL',False, cast_bool)
TEMP_DIR = config("TEMP_DIR",default="/tmp/cocoman/",cast=os.path.abspath)
MONGO_DB_URL = config("MONGODB_URL",cast=str,default='mongodb://root:password@10.8.0.17:27017/mycoco?authSource=admin')
MONGO_DB_NAME = config("MONGO_DB_NAME",cast=str,default='mycoco')