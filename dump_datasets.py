import os
import shutil
from cocoman.settings import (
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SSL,
    MINIO_URL,
    DB_POOL_SIZE,
    DB_URL,
)
from sqlalchemy import create_engine
from minio import Minio
from cocoman.mycoco.remote_coco import RemoteCOCO

engine = create_engine(DB_URL, pool_size=DB_POOL_SIZE)
minio = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SSL,
)

train_dataset_config = {
    "20230804-seg-coco": {"train": {"select-policy": {"type": "all","nums":10}}},
    "20240101-seg-coco": {"train": {"select-policy": {"type": "all","nums":10}}},
}

val_dataset_config = {
    "20230804-seg-coco": {"val": {"select-policy": {"type": "all"}}},
    "20240101-seg-coco": {"val": {"select-policy": {"type": "all"}}},
}

train_coco = RemoteCOCO(engine, minio, train_dataset_config)
val_coco = RemoteCOCO(engine, minio, val_dataset_config)

if __name__ == "__main__":
    train_coco.save(
        "/data/cam/postgres-coco/chromo_data/20240102-mixed-seg-coco/annotations/chromosome_train.json",
        "/data/cam/postgres-coco/chromo_data/20240102-mixed-seg-coco/train",
    )

    val_coco.save(
        "/data/cam/postgres-coco/chromo_data/20240102-mixed-seg-coco/annotations/chromosome_val.json",
        "/data/cam/postgres-coco/chromo_data/20240102-mixed-seg-coco/val",
    )