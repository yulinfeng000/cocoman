import sys
import json
from collections import defaultdict
import itertools
import logging
from typing import List, Dict
from pathlib import Path
from pycocotools import mask as coco_mask
from joblib import Parallel, delayed
from pymongo.database import Database
from minio import Minio
from bson import ObjectId
from tqdm import tqdm
from cocoman.common.pycococreatetools import binary_mask_to_polygon

PYTHON_VERSION = sys.version_info[0]

logger = logging.getLogger("cocoman.mycoco.mongo_coco")

{
    "20230804-seg-coco": {
        "train": {"select-policy": {"type": "random", "nums": 1000}},
        "val": {"select-policy": {"type": "all"}},
        "": {"select-policy": {"type": "index", "ids": 123123}},
    }
}


def ann_worker(ann):
    # print(ann)
    if not ann["iscrowd"]:
        ann["segmentation"] = binary_mask_to_polygon(
            coco_mask.decode(ann["segmentation"])
        )
    return ann


def _isArrayLike(obj):
    if isinstance(obj, str):
        return False
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def get_images_by_config(dataset: str, subset: str, config: dict, db: Database):
    policy = config["select-policy"]
    policy_type = policy["type"]

    if subset:
        if policy_type == "random":
            imgIds = [
                it["image_ids"]
                for it in db["datasets"].aggregate(
                    [
                        {"$match": {"dataset_type": subset, "dataset_name": dataset}},
                        {"$unwind": {"path": "$image_ids"}},
                        {"$project": {"image_ids": 1}},
                        {"$sample": {"size": policy["nums"]}},
                    ]
                )
            ]
            if not imgIds:
                raise Exception("not found")
            return imgIds

        elif policy_type == "all":
            imgIds = [
                i["image_ids"]
                for i in db["datasets"].aggregate(
                    [
                        {"$match": {"dataset_type": subset, "dataset_name": dataset}},
                        {"$unwind": {"path": "$image_ids"}},
                        {
                            "$project": {
                                "image_ids": 1,
                                "_id": 0,
                            }
                        },
                    ]
                )
            ]
            return imgIds

        elif policy_type == "index":
            return [ObjectId(i) for i in policy["ids"]]

        else:
            raise NotImplementedError(f"policy type {policy_type} not implemented")

    else:
        if policy_type == "random":
            imgIds = [
                i["_id"]
                for i in db["images"].aggregate(
                    [
                        {"$match": {"bucket_name": dataset}},
                        {"$sample": {"size": policy["nums"]}},
                        {"$project": {"_id": 1}},
                    ]
                )
            ]
            return imgIds

        elif policy_type == "all":
            imgIds = [
                i["_id"]
                for i in db["images"].aggregate(
                    [
                        {"$match": {"bucket_name": dataset}},
                        {"$project": {"_id": 1}},
                    ]
                )
            ]
            return imgIds

        elif policy_type == "index":
            return [ObjectId(i) for i in policy["ids"]]

        else:
            raise NotImplementedError(f"policy type {policy_type} not implemented")


class RemoteCOCO:

    def __init__(self, db: Database, minio: Minio, config: dict) -> None:
        self.db = db
        self.minio = minio
        self.config = config
        self.create_index()

    def create_index(self):
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.imgs, self.anns, self.cats = [], [], []

        for dataset_name, select_config in self.config.items():
            for subset, config in select_config.items():
                self.imgs.extend(
                    get_images_by_config(dataset_name, subset, config, self.db)
                )

        annotations = self.db["annotations"].aggregate(
            [{"$match": {"image_id": {"$in": self.imgs}}}]
        )

        for ann in annotations:
            self.anns.append(ann["_id"])
            self.imgToAnns[ann["image_id"]].append(ann["_id"])
            self.catToImgs[ann["category_id"]].append(ann["image_id"])

        self.cats = list(self.db["annotations"].distinct("category_id"))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None) -> List[int]:
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
            catIds  (int array)     : get anns for given cats
            areaRng (float array)   : get anns for given area range (e.g. [0 inf])
            iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """

        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            return self.anns

        if not len(imgIds) == 0:
            lists = [
                self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns
            ]
            anns = list(itertools.chain.from_iterable(lists))
            return anns

        match_conditions = []
        if len(catIds) != 0:
            match_conditions.append({"category_id": {"$in": catIds}})

        if len(areaRng) != 0:
            match_conditions.append({"area": {"$gt": areaRng[0], "$lte": areaRng[1]}})

        if iscrowd is not None:
            match_conditions.append({"iscrowd": iscrowd})

        if len(match_conditions) > 1:
            match_conditions = {"$and": match_conditions}
        print(match_conditions)
        return list(self.db["annotations"].find(match_conditions))

    def getCatIds(self, catNms=[], supNms=[], catIds=[]) -> List[int]:
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            return self.cats

        match_conditions = []

        if len(catNms) != 0:
            # Category.name.in_(catNms)
            match_conditions.append({"name": {"$in": catNms}})

        if len(supNms) != 0:
            # where_conditions.append(Category.super_category.in_(supNms))
            match_conditions.append({"supercategory": {"$in": supNms}})

        if len(catIds) != 0:
            match_conditions.append({"_id": {"$in": catIds}})

        # stmt = stmt.where(and_(*where_conditions))
        if len(match_conditions) > 1:
            match_conditions = {"$and": match_conditions}

        return list(self.db["categories"].find(match_conditions))

    def getImgIds(self, imgIds=[], catIds=[]) -> List[ObjectId]:
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs

        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]) -> List[Dict]:
        if _isArrayLike(ids):
            return list(
                self.db["annotations"].aggregate(
                    [
                        {"$match": {"_id": {"$in": ids}}},
                        {"$addFields": {"id": {"$toString": "$_id"}}},
                        {"$project": {"_id": 0}},
                        {
                            "$set": {
                                "image_id": {"$toString": "$image_id"},
                                "category_id": {"$toString": "$category_id"},
                            }
                        },
                    ]
                )
            )
        else:
            return list(
                self.db["annotations"].aggregate(
                    [
                        {"$match": {"_id": ids}},
                        {"$addFields": {"id": {"$toString": "$_id"}}},
                        {"$project": {"_id": 0}},
                        {
                            "$set": {
                                "image_id": {"$toString": "$image_id"},
                                "category_id": {"$toString": "$category_id"},
                            }
                        },
                    ]
                )
            )

    def loadImgs(self, ids=[]) -> List[Dict]:
        return self._loadByType("images", ids)

    def loadCats(self, ids=[]) -> List[Dict]:
        return self._loadByType("categories", ids)

    def _loadByType(self, collection, ids=[]):
        if _isArrayLike(ids):
            return list(
                self.db[collection].aggregate(
                    [
                        {"$match": {"_id": {"$in": ids}}},
                        {"$addFields": {"id": {"$toString": "$_id"}}},
                        {"$project": {"_id": 0}},
                    ]
                )
            )
        else:
            return list(
                self.db[collection].aggregate(
                    [
                        {"$match": {"_id": ids}},
                        {"$addFields": {"id": {"$toString": "$_id"}}},
                        {"$project": {"_id": 0}},
                    ]
                )
            )

    def annToRLE(self, ann):
        return ann["segmentation"]

    def annToMask(self, ann):
        return coco_mask.decode(ann["segmentation"])

    def serialize(self):
        images = self.loadImgs(self.getImgIds())

        annotations = []
        # singleton version
        # for ann in tqdm(self.loadAnns(self.getAnnIds()), desc="add annotations"):
        #     obj = {}
        #     obj["id"] = ann.id
        #     obj["area"] = ann.area
        #     obj["bbox"] = ann.bbox
        #     obj["category_id"] = ann.category_id
        #     obj["image_id"] = ann.image_id
        #     obj["iscrowd"] = 1 if ann.iscrowd else 0
        #     if ann.iscrowd:
        #         obj["segmentation"] = loadRLE(ann.segmentation)
        #     else:
        #         obj["segmentation"] = binary_mask_to_polygon(
        #             coco_mask.decode(loadRLE(ann.segmentation))
        #         )
        #     annotations.append(obj)

        # multiprocessing version:
        annObjs = self.loadAnns(self.getAnnIds())
        # batches = [
        #     annObjs[i : i + batch_size] for i in range(0, len(annObjs), batch_size)
        # ]
        annotations = Parallel(n_jobs=-1)(
            delayed(ann_worker)(ann)
            for ann in tqdm(annObjs, desc="Processing annotations")
        )
        # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        #     batch_results = list(
        #         tqdm(
        #             executor.map(ann_worker, batches),
        #             total=len(batches),
        #             desc="Processing annotations",
        #         )
        #     )
        # annotations = list(itertools.chain.from_iterable(batch_results))  # flatten

        categories = self.loadCats(self.getCatIds())

        coco_json = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        return coco_json

    def save(self, annFilePath, imgDir=None, saveFunc=json.dump):
        """
        save 方法

        saveFunc: 传入参数为 (json_dict, file_descriptor)
        """
        annFilePath = Path(annFilePath)
        # assert annFilePath.is_file(), "{annFilePath} is not file"
        if not annFilePath.parent.exists():
            annFilePath.parent.mkdir(parents=True)

        coco_json = self.serialize()

        with open(str(annFilePath), "w") as f:
            saveFunc(coco_json, f)

        if imgDir is not None:
            imgDir = Path(imgDir)
            if not imgDir.exists():
                imgDir.mkdir(parents=True)

            for img in coco_json["images"]:
                self.minio.fget_object(
                    img["bucket_name"],
                    img["file_name"],
                    str(imgDir.joinpath(img["file_name"])),
                )
