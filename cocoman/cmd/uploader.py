from sqlalchemy.sql import *
from cocoman.client import LocalCOCO
from cocoman.common.utils import (
    create_db_engine,
    create_minio,
    create_mongo_db,
)
from cocoman.common.settings import (
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_URL,
    MINIO_SSL,
    DB_URL,
    DB_POOL_SIZE,
    MONGO_DB_NAME,
    MONGO_DB_URL,
)
from cocoman.cmd.uploader_handles import (
    mongo_handler,
    postgres_handler,
    http_handler,
)
import argparse
import logging
from typing import Optional
logger = logging.getLogger("cocoman.cmd.uploader")


def get_args(parser:Optional[argparse.ArgumentParser]=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="上传数据集的工具")
    else:
        parser.description = "上传数据集的工具"
    parser.add_argument(
        "--backend", default="http", choices=["mongo", "postgres", "http"]
    )
    parser.add_argument(
        "--db-url",
        default=DB_URL,
        help="{username}:{password}@{host}:{port}/{db} format postgres db url or mongo url",
    )
    parser.add_argument("--db-name",default=MONGO_DB_NAME, help='mongodb dbname')
    parser.add_argument("--minio-url", default=MINIO_URL, help="minio endpoint url")
    parser.add_argument(
        "--minio-access-key",
        default=MINIO_ACCESS_KEY,
        help="minio access key alias username",
    )
    parser.add_argument(
        "--minio-secret-key",
        default=MINIO_SECRET_KEY,
        help="minio secret key alias password",
    )
    parser.add_argument(
        "--minio-ssl",
        action="store_true",
        default=MINIO_SSL,
        help="minio enable ssl protocol",
    )
    parser.add_argument(
        "--server-url", default="http://localhost:8000", help="server url"
    )
    parser.add_argument(
        "--annotation-file", required=True, help="coco annotation file path"
    )
    parser.add_argument("--img-dir", required=True, help="coco image dir path")
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="coco dataset type",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--dataset-name", required=True, help="coco dataset name")
    return parser


def cmd_entrypoint(args):

    cocoapi = LocalCOCO(
        args.dataset_name, args.dataset_type, args.annotation_file, args.img_dir
    )

    minio = create_minio(
        args.minio_url,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        ssl=args.minio_ssl,
    )

    if args.backend == "mongo":
        mongo_handler.upload(
            coco=cocoapi, db=create_mongo_db(args.db_url, args.db_name), minio=minio
        )

    elif args.backend == "postgres":
        postgres_handler.upload(
            coco=cocoapi,
            db=create_db_engine(args.db_url, pool_size=DB_POOL_SIZE),
            minio=minio,
        )

    elif args.backend == "http":
        http_handler.upload(coco=cocoapi, base_url=args.server_url)


if __name__ == "__main__":
    args = get_args()
    cmd_entrypoint(args)
