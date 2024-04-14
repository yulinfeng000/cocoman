from cocoman.client import LocalCOCO
from cocoman.common.utils import (
    create_minio,
)
import argparse
import logging
from typing import Optional
from cocoman.cmd import http_handler

logger = logging.getLogger("cocoman.cmd.uploader")


def get_args(parser: Optional[argparse.ArgumentParser] = None):
    if parser is None:
        parser = argparse.ArgumentParser(description="上传数据集的工具")
    else:
        parser.description = "上传数据集的工具"
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
    http_handler.upload(coco=cocoapi, base_url=args.server_url)


if __name__ == "__main__":
    parser = get_args()
    cmd_entrypoint(parser.parse_args())
