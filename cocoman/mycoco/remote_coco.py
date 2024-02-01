import os
from typing import Any, List
from contextlib import contextmanager
import itertools
import sqlalchemy as sq
from sqlalchemy import Engine
from minio import Minio
from sqlalchemy.orm import Session, undefer, sessionmaker, scoped_session
from sqlalchemy.sql import *
from collections import defaultdict
from cocoman.tables import Image, DataSet, Annotation, Category, Base
from cocoman.utils import array_sample, loadRLE
from pycocotools import mask as coco_mask
import logging

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


class RemoteCOCO:
    disabled = ["showAnns", "loadRes", "loadNumpyAnnotations"]

    def __getattr__(self, __name: str) -> Any:
        if __name in self.disabled:
            raise AttributeError("no such attr")
        return super().__getattr__(__name)

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

    @contextmanager
    def loadAnns(self, ids=[]) -> List[Annotation]:
        """
        和标准使用方法不一样,由于segmentation是deferred,所以需要确保在session环境中
        """
        stmt = select(Annotation)
        if _isArrayLike(ids):
            stmt = stmt.where(Annotation.id.in_(ids))
        else:
            stmt = stmt.where(Annotation.id == ids)

        stmt.options(undefer(Annotation.segmentation))
        with self.ScopedSession() as session:
            yield session.scalars(stmt).all()

    def loadImgs(self, ids=[]) -> List[Image]:
        return self._loadByType(Image, ids)

    def loadCats(self, ids=[]) -> List[Category]:
        return self._loadByType(Category, ids)

    def annToRLE(self, ann: Annotation):
        with self.ScopedSession() as session:
            session.merge(ann)
            return loadRLE(ann.segmentation)

    def annToMask(self, ann: Annotation):
        with self.ScopedSession() as session:
            session.merge(ann)
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

        for i, img in enumerate(imgs):
            fpath = os.path.join(tarDir, img.bucket_name, img.file_name)
            if not os.path.exists(fpath):
                self.minio.fget_object(img.bucket_name, img.file_name, fpath)
                print(f"{i}/{N} {os.path.basename(fpath)} downloaded", end="\r")
