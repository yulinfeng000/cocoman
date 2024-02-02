from detectron2.config import CfgNode as CN
from cocoman.settings import MINIO_ACCESS_KEY,MINIO_SECRET_KEY,MINIO_SSL,MINIO_URL,DB_URL,DB_POOL_SIZE,TEMP_DIR

def add_remote_config(cfg):
    assert not hasattr(cfg,"REMOTE"),"cfg.REMOTE already exists!"
    cfg.REMOTE = CN()
    cfg.REMOTE.MINIO_URL = MINIO_URL
    cfg.REMOTE.MINIO_ACCESS_KEY = MINIO_ACCESS_KEY
    cfg.REMOTE.MINIO_SECRET_KEY = MINIO_SECRET_KEY
    cfg.REMOTE.MINIO_SSL = MINIO_SSL
    cfg.REMOTE.DB_URL = DB_URL
    cfg.REMOTE.DB_POOL_SIZE = DB_POOL_SIZE
    cfg.REMOTE.IMG_TEMP_DIR = TEMP_DIR
    return cfg