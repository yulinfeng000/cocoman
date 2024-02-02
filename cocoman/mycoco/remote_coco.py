import os
from typing import Any, List
import json
from contextlib import contextmanager
import itertools
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
import sqlalchemy as sq
from sqlalchemy import Engine
from minio import Minio
from sqlalchemy.orm import Session, undefer, sessionmaker, scoped_session
from sqlalchemy.sql import *

from pycocotools import mask as coco_mask
from cocoman.tables import Image, DataSet, Annotation, Category, Base
from cocoman.utils import array_sample, loadRLE
from cocoman.mycoco.pycococreatetools import binary_mask_to_polygon


logger = logging.getLogger("cocoman.mycoco.remote_coco")

{
    "20230804-seg-coco": {
        "train": {"select-policy": {"type": "random", "nums": 1000}},
        "val": {"select-policy": {"type": "all"}},
        "": {"select-policy": {"type": "index", "ids": 123123}},
    }
}


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def get_images_by_config(dataset: str, subset: str, config: dict, session: Session):
    policy = config["select-policy"]
    policy_type = policy["type"]

    if subset:
        if policy_type == "random":
            stmt = select(array_sample(DataSet.image_ids, policy["nums"])).where(
                and_(DataSet.dataset_name == dataset, DataSet.dataset_type == subset)
            )
            imgIds = session.scalar(stmt)
            if not imgIds:
                raise Exception("not found")
            return imgIds
        elif policy_type == "index":
            return policy["ids"]
        elif policy_type == "all":
            stmt = select(DataSet.image_ids).where(
                and_(DataSet.dataset_name == dataset, DataSet.dataset_type == subset)
            )
            imgIds = session.scalar(stmt)
            return imgIds
        else:
            raise NotImplementedError(f"policy type {policy_type} not implemented")

    else:
        if policy_type == "random":
            stmt = (
                select(Image.id)
                .where(Image.bucket_name == dataset)
                .order_by(sq.func.random())
                .limit(policy["nums"])
            )
            imgIds = session.scalars(stmt).all()
            return imgIds
        elif policy_type == "index":
            return policy["ids"]
        elif policy_type == "all":
            stmt = select(Image.id).where(Image.bucket_name == dataset)
            imgIds = session.scalars(stmt).all()
            return imgIds
        else:
            raise NotImplementedError(f"policy type {policy_type} not implemented")


def ann_worker(anns):
    subsets = []
    for ann in anns:
        obj = {
            "id": ann.id,
            "area": ann.area,
            "bbox": ann.bbox,
            "category_id": ann.category_id,
            "image_id": ann.image_id,
            "iscrowd": 1 if ann.iscrowd else 0,
        }
        if ann.iscrowd:
            obj["segmentation"] = loadRLE(ann.segmentation)
        else:
            obj["segmentation"] = binary_mask_to_polygon(
                coco_mask.decode(loadRLE(ann.segmentation))
            )
        subsets.append(obj)
    return subsets


class RemoteCOCO:
    def __init__(self, db: Engine, minio: Minio, config: dict):
        self.db = db
        self.sessionmaker = sessionmaker(bind=db)
        self.minio = minio
        self.config = config
        self.create_index()

    @contextmanager
    def ScopedSession(self):
        with scoped_session(self.sessionmaker)() as session:
            yield session

    @contextmanager
    def Session(self):
        with self.sessionmaker() as session:
            yield session

    def create_index(self):
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.imgs = []
        with self.ScopedSession() as session:
            for dataset_name, select_config in self.config.items():
                for subset, config in select_config.items():
                    self.imgs.extend(
                        get_images_by_config(dataset_name, subset, config, session)
                    )
            stmt = select(
                Annotation.id, Annotation.category_id, Annotation.image_id
            ).where(Annotation.image_id.in_(self.imgs))
            annotations = session.execute(stmt).all()
            self.anns = [ann[0] for ann in annotations]

            for ann in annotations:
                self.imgToAnns[ann[2]].append(ann[0])
                self.catToImgs[ann[1]].append(ann[2])

            stmt = (
                select(Annotation.category_id)
                .where(Annotation.id.in_(self.anns))
                .distinct(Annotation.category_id)
            )
            self.cats = session.scalars(stmt).all()

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

        with self.ScopedSession() as session:
            stmt = select(Annotation.id)
            where_conditions = []
            if len(catIds) != 0:
                where_conditions.append(Annotation.category_id.in_(catIds))

            if len(areaRng) != 0:
                where_conditions.append(
                    and_(Annotation.area > areaRng[0], Annotation.area <= areaRng[1])
                )

            if iscrowd:
                where_conditions.append(Annotation.iscrowd == iscrowd)
            stmt = stmt.where(and_(*where_conditions))
            return session.scalars(stmt).all()

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

        with self.ScopedSession() as session:
            stmt = select(Category.id)
            where_conditions = []

            if len(catNms) != 0:
                where_conditions.append(Category.name.in_(catNms))

            if len(supNms) != 0:
                where_conditions.append(Category.super_category.in_(supNms))

            stmt = stmt.where(and_(*where_conditions))

            return session.scalars(stmt).all()

    def getImgIds(self, imgIds=[], catIds=[]) -> List[int]:
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

    def _loadByType(self, type_cls: Base, ids=[], options=None):
        stmt: Select = select(type_cls)
        if _isArrayLike(ids):
            stmt = stmt.where(type_cls.id.in_(ids))
        else:
            stmt = stmt.where(type_cls.id == ids)

        with self.ScopedSession() as session:
            return session.scalars(stmt).all()

    def loadAnns(self, ids=[]) -> List[Annotation]:
        stmt = select(Annotation).options(undefer(Annotation.segmentation))
        if _isArrayLike(ids):
            stmt = stmt.where(Annotation.id.in_(ids))
        else:
            stmt = stmt.where(Annotation.id == ids)

        with self.ScopedSession() as session:
            return session.scalars(stmt).all()

    def loadImgs(self, ids=[]) -> List[Image]:
        return self._loadByType(Image, ids)

    def loadCats(self, ids=[]) -> List[Category]:
        return self._loadByType(Category, ids)

    def annToRLE(self, ann: Annotation):
        return loadRLE(ann.segmentation)

    def annToMask(self, ann: Annotation):
        rle = loadRLE(ann.segmentation)
        return coco_mask.decode(rle)

    def download(self, tarDir=None, imgIds=[]):
        if tarDir is None:
            print("Please specify target directory")
            return -1
        if len(imgIds) == 0:
            imgs = self.loadImgs(self.imgs)
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)

        for img in tqdm(imgs, desc="download images"):
            fpath = os.path.join(tarDir, img.bucket_name, img.file_name)
            if not os.path.exists(fpath):
                self.minio.fget_object(img.bucket_name, img.file_name, fpath)

    def save(self, annFilePath, imgDir):
        annFilePath = Path(annFilePath)
        # assert annFilePath.is_file(), "{annFilePath} is not file"
        if not annFilePath.parent.exists():
            annFilePath.parent.mkdir(parents=True)

        imgDir = Path(imgDir)
        # assert imgDir.is_dir(), f"{imgDir} is not directory"
        if not imgDir.exists():
            imgDir.mkdir(parents=True)

        images = []
        imgObjs = self.loadImgs(self.getImgIds())
        for img in tqdm(imgObjs, desc="Processing images"):
            obj = {}
            obj["id"] = img.id
            obj["bucket_name"] = img.bucket_name
            obj["file_name"] = img.file_name
            obj["height"] = img.height
            obj["width"] = img.width

            images.append(obj)

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

        # multiprocess version:
        annIds = self.getAnnIds()
        batch_size = 300
        annObjs = self.loadAnns(annIds)
        batches = [
            annObjs[i : i + batch_size] for i in range(0, len(annObjs), batch_size)
        ]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            batch_results = list(
                tqdm(
                    executor.map(ann_worker, batches),
                    total=len(batches),
                    desc="Processing annotations",
                )
            )
        annotations = list(itertools.chain.from_iterable(batch_results))  # flatten

        categories = []
        for cat in tqdm(self.loadCats(self.getCatIds()), desc="Processing categories"):
            obj = {}
            obj["id"] = cat.id
            obj["name"] = cat.name
            obj["supercategory"] = cat.super_category

            categories.append(obj)

        coco_json = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(str(annFilePath), "w") as f:
            json.dump(coco_json, f)

        for img in imgObjs:
            self.minio.fget_object(
                img.bucket_name, img.file_name, str(imgDir.joinpath(img.file_name))
            )
