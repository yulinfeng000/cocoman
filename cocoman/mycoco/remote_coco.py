import os
import sys
from typing import List
import time
import json
from contextlib import contextmanager
import itertools
import logging
import copy
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from pathlib import Path
import numpy as np
import sqlalchemy as sq
from sqlalchemy import Engine
from minio import Minio
from sqlalchemy.orm import Session, undefer, sessionmaker, scoped_session
from sqlalchemy.sql import *
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO as MSCOCO
from cocoman.tables import Image, DataSet, Annotation, Category, Base
from cocoman.utils import array_sample, loadRLE
from cocoman.mycoco.pycococreatetools import binary_mask_to_polygon


PYTHON_VERSION = sys.version_info[0]


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


def ann_worker(ann):
    if ann["iscrowd"]:
        ann["segmentation"] = loadRLE(ann["segmentation"])
    else:
        ann["segmentation"] = binary_mask_to_polygon(
            coco_mask.decode(loadRLE(ann["segmentation"]))
        )
    return ann


def decodeAnn(ann):
    ann["segmentation"] = loadRLE(ann["segmentation"])
    return ann


class RemoteCOCO:
    def __init__(self, db: Engine, minio: Minio, config: dict, cache_dir=None):
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

            if iscrowd is not None:
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

    def _loadByType(self, type_cls: Base, ids=[]):
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
            anns = session.scalars(stmt).all()

        return [
            {
                "id": ann.id,
                "area": ann.area,
                "bbox": ann.bbox,
                "category_id": ann.category_id,
                "image_id": ann.image_id,
                "iscrowd": 1 if ann.iscrowd else 0,
                "segmentation": ann.segmentation,  # str
            }
            for ann in anns
        ]

    def loadImgs(self, ids=[]) -> List[Image]:
        images = self._loadByType(Image, ids)
        return [
            {
                "id": img.id,
                "bucket_name": img.bucket_name,
                "file_name": img.file_name,
                "height": img.height,
                "width": img.width,
            }
            for img in images
        ]

    def loadCats(self, ids=[]) -> List[Category]:
        categories = self._loadByType(Category, ids)
        return [
            {"id": cat.id, "name": cat.name, "supercategory": cat.super_category}
            for cat in categories
        ]

    def annToRLE(self, ann):
        return loadRLE(ann["segmentation"])

    def annToMask(self, ann):
        rle = self.annToRLE(ann)
        return coco_mask.decode(rle)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = MSCOCO()
        res.dataset["images"] = [img for img in self.dataset["images"]]

        print("Loading and preparing results...")
        tic = time.time()

        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            with open(resFile) as f:
                anns = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (
            set(annsImgIds) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"
        if "caption" in anns[0]:
            imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
                [ann["image_id"] for ann in anns]
            )
            res.dataset["images"] = [
                img for img in res.dataset["images"] if img["id"] in imgIds
            ]
            for id, ann in enumerate(anns):
                ann["id"] = id + 1
        elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not "segmentation" in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = coco_mask.area(ann["segmentation"])
                if not "bbox" in ann:
                    ann["bbox"] = coco_mask.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        print("DONE (t={:0.2f}s)".format(time.time() - tic))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if "segmentation" in anns[0] or "keypoints" in anns[0]:
            datasetType = "instances"
        elif "caption" in anns[0]:
            datasetType = "captions"
        else:
            raise Exception("datasetType not supported")
        if datasetType == "instances":
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon

            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if "segmentation" in ann:
                    if type(ann["segmentation"]) == list:
                        # polygon
                        for seg in ann["segmentation"]:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    elif type(ann["segmentation"]) == str:
                        m = self.annToMask(ann)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann["iscrowd"] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann["iscrowd"] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                    else:
                        # mask
                        t = self.imgs[ann["image_id"]]
                        if type(ann["segmentation"]["counts"]) == list:
                            rle = coco_mask.frPyObjects(
                                [ann["segmentation"]], t["height"], t["width"]
                            )
                        else:
                            rle = [ann["segmentation"]]
                        m = coco_mask.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann["iscrowd"] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann["iscrowd"] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))

                if "keypoints" in ann and type(ann["keypoints"]) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
                    kp = np.array(ann["keypoints"])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(
                        x[v > 0],
                        y[v > 0],
                        "o",
                        markersize=8,
                        markerfacecolor=c,
                        markeredgecolor="k",
                        markeredgewidth=2,
                    )
                    plt.plot(
                        x[v > 1],
                        y[v > 1],
                        "o",
                        markersize=8,
                        markerfacecolor=c,
                        markeredgecolor=c,
                        markeredgewidth=2,
                    )

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
                    poly = [
                        [bbox_x, bbox_y],
                        [bbox_x, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y],
                    ]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(
                polygons, facecolor="none", edgecolors=color, linewidths=2
            )
            ax.add_collection(p)
        elif datasetType == "captions":
            for ann in anns:
                print(ann["caption"])

    def download(self, tarDir=None, imgIds=[]):
        if tarDir is None:
            print("Please specify target directory")
            return -1
        if len(imgIds) == 0:
            imgs = self.loadImgs(self.imgs)
        else:
            imgs = self.loadImgs(imgIds)

        if not os.path.exists(tarDir):
            os.makedirs(tarDir)

        for img in tqdm(imgs, desc="download images"):
            fpath = os.path.join(tarDir, img["bucket_name"], img["file_name"])
            if not os.path.exists(fpath):
                self.minio.fget_object(img["bucket_name"], img["file_name"], fpath)

    def save(self, annFilePath, imgDir=None, saveFunc=json.dump):
        """
        save 方法

        saveFunc: 传入参数为 (json_dict, file_descriptor)
        """
        annFilePath = Path(annFilePath)
        # assert annFilePath.is_file(), "{annFilePath} is not file"
        if not annFilePath.parent.exists():
            annFilePath.parent.mkdir(parents=True)

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

        with open(str(annFilePath), "w") as f:
            saveFunc(coco_json, f)

        if imgDir is not None:
            imgDir = Path(imgDir)
            if not imgDir.exists():
                imgDir.mkdir(parents=True)

            for img in images:
                self.minio.fget_object(
                    img["bucket_name"],
                    img["file_name"],
                    str(imgDir.joinpath(img["file_name"])),
                )
