import logging
from urllib.parse import urljoin
import requests
from tqdm import tqdm
from cocoman.client.local_coco import LocalCOCO
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger("cocoman.cmd.upload_handler.http_handler")


def upload(coco: LocalCOCO, base_url):
    with ThreadPoolExecutor() as executor:
        with requests.Session() as client:

            def uploadItem(api, item):
                resp = client.post(urljoin(base_url, api), json=jsonable_encoder(item))
                data = resp.json()
                return data["id"]

            category_mapping = {}
            image_mapping = {}
            image_paths = coco.getImgPaths(coco.getImgIds())
            logger.info("start uploading image files")
            worker = lambda img_path: client.post(
                urljoin(base_url, "/uploadImgPic"),
                files=dict(
                    img_file=(img_path.name, open(img_path.absolute(), "rb")),
                ),
                data={"dataset_name": coco.dataset_name},
            )
            tasks = [executor.submit(worker, img_path) for img_path in image_paths]
            for future in tqdm(
                as_completed(tasks), total=len(tasks), desc="uploading image pictures"
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.exception("has error during uploading image", e)

            for category in tqdm(
                coco.loadCats(coco.getCatIds()), desc="saving categories info"
            ):
                cat_mongo_id = uploadItem("/uploadCat", category)
                category_mapping[category["id"]] = cat_mongo_id

            def upload_img_worker(image):
                img = dict(
                    file_name=image["file_name"],
                    width=image["width"],
                    height=image["height"],
                    bucket_name=coco.dataset_name,
                )
                img_mongo_id = uploadItem("/uploadImg", img)
                image_mapping[image["id"]] = img_mongo_id

            tasks = [
                executor.submit(upload_img_worker, image)
                for image in coco.loadImgs(coco.getImgIds())
            ]

            for future in tqdm(
                as_completed(tasks), total=len(tasks), desc="saving images info"
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.exception("has error during saving image document", e)

            annotations = [
                dict(
                    image_id=image_mapping[ann["image_id"]],
                    category_id=category_mapping[ann["category_id"]],
                    iscrowd=True if ann["iscrowd"] else False,
                    segmentation=coco.annToRLE(ann),
                    bbox=ann["bbox"],
                    area=ann["area"],
                )
                for ann in coco.loadAnns(coco.getAnnIds())
            ]
            batch_size = 1000
            batches = [
                annotations[i : i + batch_size]
                for i in range(0, len(annotations), batch_size)
            ]
            tasks = [
                executor.submit(
                    lambda batch: uploadItem("/uploadAnns", dict(anns=batch)), batch
                )
                for batch in batches
            ]
            for future in tqdm(
                as_completed(tasks), total=len(tasks), desc="saving annotations info"
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.exception("has error during saving annotation document", e)

            dataset = dict(
                dataset_name=coco.dataset_name,
                dataset_type=coco.dataset_type,
                image_ids=list(image_mapping.values()),
            )
            logger.info("saving dataset info", dataset["dataset_name"])
            uploadItem("/uploadDataset", dataset)
