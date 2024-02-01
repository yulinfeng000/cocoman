from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import *
from minio import Minio
from cocoman.mycoco import LocalCOCO
from cocoman.tables import Image, Category, Annotation, DataSet
from cocoman.utils import dumpRLE, object_exists
from cocoman.settings import (
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_URL,
    MINIO_SSL,
    DB_URL,
    DB_POOL_SIZE,
)

import argparse
from tqdm import tqdm
import logging

logger = logging.getLogger("cocoman.commandline.uploader")


def upload(coco: LocalCOCO, engine: Engine, minio: Minio):
    category_mapping = {}
    image_mapping = {}
    with Session(engine) as session:
        with session.begin():
            try:
                if not minio.bucket_exists(coco.dataset_name):
                    minio.make_bucket(coco.dataset_name)

                image_paths = coco.getImgPaths(coco.getImgIds())
                logging.info("start uploading image files")
                for img_path in tqdm(image_paths, desc="upload images"):
                    if not object_exists(minio, coco.dataset_name, img_path.name):
                        suffix = img_path.suffix
                        minio.fput_object(
                            coco.dataset_name,
                            img_path.name,
                            str(img_path.absolute()),
                            content_type=f"image/{suffix[1:]}",
                        )

                logging.info("saving categories info")
                for category in tqdm(
                    coco.loadCats(coco.getCatIds()), desc="saving categories info"
                ):
                    cat = Category(
                        name=category["name"],
                        super_category=category["supercategory"],
                    )
                    if (
                        result := session.execute(
                            select(Category)
                            .where(
                                and_(
                                    Category.name == cat.name,
                                    Category.super_category == cat.name,
                                )
                            )
                            .limit(1)
                        ).one_or_none()
                    ) is None:
                        session.add(cat)
                        session.flush()
                    else:
                        cat = result[0]

                    category_mapping[category["id"]] = cat.id

                logging.info("saving images info")
                for image in tqdm(
                    coco.loadImgs(coco.getImgIds()), desc="saving images info"
                ):
                    img = Image(
                        file_name=image["file_name"],
                        width=image["width"],
                        height=image["height"],
                        bucket_name=coco.dataset_name,
                    )
                    session.add(img)
                    session.flush()
                    image_mapping[image["id"]] = img.id

                logging.info("saving annotations info")
                for ann in tqdm(
                    coco.loadAnns(coco.getAnnIds()), desc="saving annotations info"
                ):
                    annotation = Annotation(
                        image_id=image_mapping[ann["image_id"]],
                        category_id=category_mapping[ann["category_id"]],
                        iscrowd=ann["iscrowd"],
                        segmentation=dumpRLE(coco.annToRLE(ann)),
                        bbox=ann["bbox"],
                        area=ann["area"],
                    )
                    session.add(annotation)
                    session.flush()

                logging.info("creating dataset")
                dataset = DataSet(
                    dataset_name=coco.dataset_name,
                    dataset_type=coco.dataset_type,
                    image_ids=list(image_mapping.values()),
                )

                session.add(dataset)
                session.flush()
                session.commit()
                logging.info("congratulation, finish uploading !")

            except Exception as e:
                logging.error(e, exc_info=True)
                session.rollback()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="command to run", choices=["upload"])
    parser.add_argument(
        "--db-url", default=DB_URL, help="sqlalchemy format postgres db url"
    )
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
    return parser.parse_args()


def cmd_entrypoint(args):
    engine = create_engine(args.db_url, pool_size=DB_POOL_SIZE)

    minio = Minio(
        args.minio_url,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=args.minio_ssl,
    )

    upload(
        LocalCOCO(
            args.dataset_name, args.dataset_type, args.annotation_file, args.img_dir
        ),
        engine,
        minio,
    )
