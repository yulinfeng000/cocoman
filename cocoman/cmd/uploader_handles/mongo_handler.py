from pymongo.database import Database
from minio import Minio
import logging
from tqdm import tqdm
from cocoman.client import LocalCOCO
from cocoman.common.utils import object_exists

logger = logging.getLogger("cocoman.cmd.mongo_handler")


def upload(coco: LocalCOCO, db: Database, minio: Minio):
    category_mapping = {}
    image_mapping = {}

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
        cat = db["categories"].find_one(
            {"name": category["name"], "supercategory": category["supercategory"]}
        )

        if cat is None:
            result = db["categories"].insert_one(
                {"name": category["name"], "supercategory": category["supercategory"]}
            )

            category_mapping[category["id"]] = result.inserted_id
        else:
            category_mapping[category["id"]] = cat["_id"]

    logging.info("saving images info")

    for image in tqdm(coco.loadImgs(coco.getImgIds()), desc="saving images info"):
        img = dict(
            file_name=image["file_name"],
            width=image["width"],
            height=image["height"],
            bucket_name=coco.dataset_name,
        )
        result = db["images"].insert_one(img)
        image_mapping[image["id"]] = result.inserted_id

    logging.info("saving annotations info")

    annotations = [
        dict(
            image_id=image_mapping[ann["image_id"]],
            category_id=category_mapping[ann["category_id"]],
            iscrowd=True if ann["iscrowd"] else False,
            segmentation=coco.annToRLE(ann),
            bbox=ann["bbox"],
            area=ann["area"],
        )
        for ann in tqdm(coco.loadAnns(coco.getAnnIds()), desc="saving annotations info")
    ]
    db["annotations"].insert_many(annotations)

    logging.info("creating dataset")
    dataset = dict(
        dataset_name=coco.dataset_name,
        dataset_type=coco.dataset_type,
        image_ids=list(image_mapping.values()),
    )
    db["datasets"].insert_one(dataset)
    logging.info("congratulation, finish uploading !")
