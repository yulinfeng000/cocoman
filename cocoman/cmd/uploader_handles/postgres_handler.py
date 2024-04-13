from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import *
from minio import Minio
from cocoman.client import LocalCOCO
from cocoman.common.tables import Image, Category, Annotation, DataSet
from cocoman.common.utils import dumpRLE, object_exists, create_db_engine, create_minio
from cocoman.common.settings import (
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_URL,
    MINIO_SSL,
    DB_URL,
    DB_POOL_SIZE,
)

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
