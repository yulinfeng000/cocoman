import json
from typing import Dict, List
import itertools
import requests
from pycocotools import mask as coco_mask
from urllib.parse import urljoin
import tempfile
from pycocotools.coco import COCO as MSCOCO


def _isArrayLike(obj):
    return (
        not isinstance(obj, str)
        and hasattr(obj, "__iter__")
        and hasattr(obj, "__len__")
    )


def cache(fn):
    def fn_warp(self, *args, **kwargs):
        if self._local_coco is not None:
            return getattr(self._local_coco, fn.__name__)(*args, **kwargs)
        else:
            return fn(self, *args, **kwargs)

    return fn_warp


class RemoteCOCO(object):
    def __init__(self, config, base_url) -> None:
        self.config = config
        self.base_url = base_url
        self.client = requests.Session()
        self._local_coco = None
        self.createIndex()

    def _call_remote(self, uri, payload):
        resp = self.client.post(urljoin(self.base_url, uri), json=payload)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise requests.exceptions.HTTPError(response=resp)

    def _call_remote_stream(self, uri, payload):
        resp = self.client.post(urljoin(self.base_url, uri), json=payload)
        if resp.status_code == 200:
            return resp.iter_content()
        else:
            raise requests.exceptions.HTTPError(response=resp)

    @cache
    def createIndex(self):
        results = self._call_remote("/createIndex", dict(config=self.config))
        self.imgToAnns = results["imgToAnns"]
        self.catToImgs = results["catToImgs"]
        self.imgs = results["imgs"]
        self.anns = results["anns"]
        self.cats = results["cats"]

    @cache
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

        else:
            return self._call_remote(
                "/getAnnIds", dict(catIds=catIds, areaRng=areaRng, iscrowd=iscrowd)
            )

    @cache
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

        else:
            return self._call_remote(
                "/getCatIds", dict(catNms=catNms, supNms=supNms, catIds=catIds)
            )

    @cache
    def getImgIds(self, imgIds=[], catIds=[]) -> List[str]:
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

    def _loadByType(self, api, ids=[]):
        if not _isArrayLike(ids):
            ids = [ids]

        with tempfile.TemporaryFile() as tmp:
            for content in self._call_remote_stream(api, dict(ids=ids)):
                tmp.write(content)
            tmp.seek(0)
            return json.load(tmp)

    @cache
    def loadAnns(self, ids=[]):
        return self._loadByType("/loadAnns", ids)

    @cache
    def loadCats(self, ids=[]):
        return self._loadByType("/loadCats", ids)

    @cache
    def loadImgs(self, ids=[]):
        return self._loadByType("/loadImgs", ids)

    @cache
    def annToRLE(self, ann):
        return ann["segmentation"]

    @cache
    def annToMask(self, ann):
        return coco_mask.decode(ann["segmentation"])

    def saveImgPic(self, img, location):
        with open(location, "wb") as f:
            with self.client.post(
                urljoin(self.base_url, "/getImgPic"), json=dict(imgId=img["id"])
            ) as resp:
                for content in resp.iter_content():
                    f.write(content)

    def localization(self):
        if self._local_coco is None:
            images = self.loadImgs(self.getImgIds())
            annotations = self.loadAnns(self.getAnnIds())
            categories = self.loadCats(self.getCatIds())
            dataset = {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
            # clone self
            local_coco = MSCOCO()
            local_coco.dataset = dataset
            local_coco.createIndex()
            self._local_coco = local_coco

    def save(self, location):
        if self._local_coco:
            with open(location, "w") as f:
                json.dump(self._local_coco.dataset, f)
        else:
            images = self.loadImgs(self.getImgIds())
            annotations = self.loadAnns(self.getAnnIds())
            categories = self.loadCats(self.getCatIds())
            dataset = {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
            with open(location, "w") as f:
                json.dump(dataset, f)
