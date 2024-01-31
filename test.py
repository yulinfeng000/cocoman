import sqlalchemy as sq
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql import *
from pycocotools.coco import COCO as MSCOCO
from minio import Minio
from pathlib import Path
from cocoman.tables import Image, Category, Annotation, DataSet, Base
from cocoman.utils import dumpRLE, object_exists, loadRLE
from pycocotools import mask as coco_mask
from collections import defaultdict
import itertools


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO(MSCOCO):
    def __init__(self, dataset_name, dataset_type, annotation_file, image_dir) -> None:
        super().__init__(annotation_file)
        self.dataset_name = dataset_name
        self.annotation_file = annotation_file
        self.image_dir = Path(image_dir)
        self.dataset_type = dataset_type

    def getImgPaths(self, imgIds: list[int] | int):
        if isinstance(imgIds, int):
            imgIds = [imgIds]
        images = self.loadImgs(imgIds)
        return [self.image_dir.joinpath(img["file_name"]) for img in images]


engine = create_engine("postgresql+psycopg2://postgres:iamroot@10.8.0.17:5432/coco")
minio = Minio(
    "10.8.0.17:9900", access_key="tomcat", secret_key="tomcatisroot", secure=False
)


def test_db():
    coco = COCO(
        "20230804-seg-coco",
        "train",
        "/data/cam/postgres-coco/chromo_data/20230804-seg-coco/annotations/chromosome_train.json",
        "/data/cam/postgres-coco/chromo_data/20230804-seg-coco/train",
    )
    category_mapping = {}
    image_mapping = {}
    with Session(engine) as session:
        with session.begin():
            try:
                if not minio.bucket_exists(coco.dataset_name):
                    minio.make_bucket(coco.dataset_name)

                image_paths = coco.getImgPaths(coco.getImgIds())
                for img_path in image_paths:
                    if not object_exists(minio, coco.dataset_name, img_path.name):
                        suffix = img_path.suffix
                        minio.fput_object(
                            coco.dataset_name,
                            img_path.name,
                            str(img_path.absolute()),
                            content_type=f"image/{suffix[1:]}",
                        )

                for category in coco.loadCats(coco.getCatIds()):
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
                        session.refresh(cat)
                    else:
                        cat = result[0]

                    category_mapping[category["id"]] = cat.id

                for image in coco.loadImgs(coco.getImgIds()):
                    img = Image(
                        file_name=image["file_name"],
                        width=image["width"],
                        height=image["height"],
                        bucket_name=coco.dataset_name,
                    )
                    session.add(img)
                    session.flush()
                    image_mapping[image["id"]] = img.id

                for ann in coco.loadAnns(coco.getAnnIds()):
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

                dataset = DataSet(
                    dataset_name=coco.dataset_name,
                    dataset_type=coco.dataset_type,
                    image_ids=list(image_mapping.values()),
                )

                session.add(dataset)
                session.flush()
                session.commit()

            except Exception as e:
                session.rollback()


def test_get_image_by_config():
    config = {
        "20230804-seg-coco": {
            "train": {"select-policy": {"type": "random", "nums": 10}},
            # "val": {"select-policy": {"type": "index", "ids": [1, 3, 4, 57, 4, 6, 9]}},
            # "": {"select-policy": {"type": "random", "nums": 10}},
        }
    }
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.sql.expression import ColumnClause
    from sqlalchemy.types import ARRAY

    class array_sample(expression.Executable, expression.ColumnElement):
        inherit_cache = True

        def __init__(self, array_column, samples):
            self.array_column = array_column
            self.num_samples = samples

    @compiles(array_sample)
    def compile_array_sample(element, compiler, **kw):
        return "array_sample( %s, %d )" % (
            compiler.process(element.array_column, asfrom=True, **kw),
            element.num_samples,
        )

    def get_images_by_config(dataset: str, subset: str, config: dict, session: Session):
        policy = config["select-policy"]
        policy_type = policy["type"]

        if subset:
            stmt = select(array_sample(DataSet.image_ids, policy["nums"])).where(
                and_(DataSet.dataset_name == dataset, DataSet.dataset_type == subset)
            )
            imgIds = session.execute(stmt).scalar()
            if not imgIds:
                raise Exception("not found")
            print(imgIds)
            return imgIds

        else:
            if policy_type == "random":
                stmt = (
                    select(Image.id)
                    .where(Image.bucket_name == dataset)
                    .order_by(sq.func.random())
                    .limit(policy["nums"])
                )

        imgIds = session.scalars(stmt).all()
        print(dataset, subset, imgIds)
        return imgIds

    class COCOMan:
        def __init__(self, engine: Engine, minio: Minio, config: dict):
            self.engine = engine
            self.minio = minio
            self.config = config
            self.create_index()

        def create_index(self):
            self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
            self.imgs = []
            with Session(self.engine) as session:
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
                    select(Category.id)
                    .where(Category.id in self.anns)
                    .distinct(Category.name, Category.super_category)
                )
                self.cats = session.scalars(stmt).all()

        def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
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

            with Session(self.engine) as session:
                stmt = select(Annotation.id)
                where_conditions = []
                if len(catIds) != 0:
                    where_conditions.append(Annotation.category_id.in_(catIds))

                if len(areaRng) != 0:
                    where_conditions.append(
                        Annotation.area > areaRng[0], Annotation.area <= areaRng[1]
                    )

                stmt = stmt.where(and_(*where_conditions))
                return session.scalars(stmt).all()

        def getCatIds(self, catNms=[], supNms=[], catIds=[]):
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

            with Session(self.engine) as session:
                stmt = select(Category.id)
                where_conditions = []

                if len(catNms) != 0:
                    where_conditions.append(Category.name in catNms)

                if len(supNms) != 0:
                    where_conditions.append(Category.super_category in supNms)

                stmt = stmt.where(and_(*where_conditions))

                return session.scalars(stmt).all()

        def getImgIds(self, imgIds=[], catIds=[]):
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
            stmt = select(type_cls)

            if _isArrayLike(ids):
                stmt.where(type_cls.id.in_(ids))
            else:
                stmt.where(type_cls.id == ids)

            with Session(self.engine) as session:
                return session.scalars(stmt).all()

        def loadAnns(self, ids=[]):
            return self._loadByType(Annotation, ids)

        def loadImgs(self, ids=[]):
            return self._loadByType(Image, ids)

        def loadCats(self, ids=[]):
            return self._loadByType(Category, ids)

        def annToRLE(self, ann: Annotation):
            with Session(self.engine) as session:
                session.refresh(ann)
                return loadRLE(ann.segmentation)

        def annToMask(self, ann: Annotation):
            with Session(self.engine) as session:
                session.refresh(ann)
                rle = loadRLE(ann.segmentation)
                return coco_mask.decode(rle)

    coolman = COCOMan(engine, minio, config)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-url", required=True, help="sqlalchemy format postgres db url"
    )
    parser.add_argument("--minio-url", required=True, help="minio endpoint url")
    parser.add_argument(
        "--minio-access-key", required=True, help="minio access key alias username"
    )
    parser.add_argument(
        "--minio-secret-key", required=True, help="minio secret key alias password"
    )
    parser.add_argument(
        "--minio-ssl",
        action="store_true",
        default=False,
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
    parser.parse_args()


def test_coco():
    global engine,minio
    config = {
        "20230804-seg-coco": {
            "train": {"select-policy": {"type": "random", "nums": 10}},
            # "val": {"select-policy": {"type": "index", "ids": [1, 3, 4, 57, 4, 6, 9]}},
            # "": {"select-policy": {"type": "random", "nums": 10}},
        }
    }
    from cocoman.mycoco import RemoteCOCO

    coco = RemoteCOCO(engine,minio,config)

    anns = coco.getAnnIds()
    ann = coco.loadAnns(anns[0])
    print(coco.annToRLE(ann[0]))


def test_integration_detectron2():
    global engine,minio
    from cocoman.mycoco import RemoteCOCO
    from cocoman.integration.detectron2.remote_coco import load_remote_coco_json
    config = {
        "20230804-seg-coco": {
            "train": {"select-policy": {"type": "random", "nums": 2}},
        },
        '3k5-seg-coco': {
            '': {'select-policy':{'type':'random','nums':1}}
        }
    }
    dataset_dict = load_remote_coco_json(
        RemoteCOCO(engine,minio,config),'new_datasets'
    )

    print(len(dataset_dict),dataset_dict[0].keys())

if __name__ == "__main__":
    test_integration_detectron2()
