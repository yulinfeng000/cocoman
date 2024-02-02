import os
import shutil
from cocoman.settings import (
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SSL,
    MINIO_URL,
    DB_POOL_SIZE,
    DB_URL,
)
from sqlalchemy import create_engine
from minio import Minio

engine = create_engine(DB_URL, pool_size=DB_POOL_SIZE)
minio = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SSL,
)


def test_coco():
    from cocoman.mycoco import RemoteCOCO

    global engine, minio, config
    select_img_nums = 10
    config = {
        "20230804-seg-coco": {
            "train": {"select-policy": {"type": "random", "nums": select_img_nums}},
            # "val": {"select-policy": {"type": "index", "ids": [1, 3, 4, 57, 4, 6, 9]}},
            # "": {"select-policy": {"type": "random", "nums": 10}},
        }
    }
    coco = RemoteCOCO(engine, minio, config)

    annIds = coco.getAnnIds()
    with coco.loadAnns(annIds[0]) as anns:
        rle = coco.annToRLE(anns[0])
        assert (
            (rle is not None)
            and isinstance(rle, dict)
            and rle.get("counts", None) is not None
        )

    assert len(coco.getImgIds()) == select_img_nums

    targetDir = "/tmp/postgres-coco"

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    imgIds = coco.getImgIds()[:3]
    files = [
        os.path.join(targetDir, img.bucket_name, img.file_name)
        for img in coco.loadImgs(imgIds)
    ]

    coco.download(targetDir, imgIds)
    for f in files:
        assert os.path.exists(f) and os.path.isfile(f)

    shutil.rmtree(targetDir)


def test_load_remote_coco():
    global engine, minio
    from cocoman.mycoco import RemoteCOCO
    from cocoman.integration.detectron2.remote_coco_dataset import load_remote_coco_json

    config = {
        "20230804-seg-coco": {
            "train": {"select-policy": {"type": "random", "nums": 2}},
        },
        "3k5-seg-coco": {"": {"select-policy": {"type": "random", "nums": 1}}},
    }
    dataset_dict = load_remote_coco_json(
        RemoteCOCO(engine, minio, config), "new_datasets"
    )

    assert len(dataset_dict), dataset_dict[0].values() != 0


def test_local_coco():
    from cocoman.mycoco import LocalCOCO

    annotation_file = "/data/cam/postgres-coco/chromo_data/20230804-seg-coco/annotations/chromosome_train.json"
    image_dir = "/data/cam/postgres-coco/chromo_data/20230804-seg-coco/train"
    coco = LocalCOCO("20230804-seg-coco", "train", annotation_file, image_dir)
    assert len(os.listdir(image_dir)) == len(coco.getImgIds())


def test_remote_coco_mapper():
    global engine, minio
    from cocoman.mycoco import RemoteCOCO
    from cocoman.integration.detectron2.remote_coco_dataset import load_remote_coco_json
    from cocoman.integration.detectron2.remote_coco_mapper import (
        RemoteCOCOInstanceChromosomeDatasetMapper,
    )
    from cocoman.integration.detectron2.config_patch import add_remote_config, CN

    config = {
        "20230804-seg-coco": {
            "train": {"select-policy": {"type": "random", "nums": 10}},
            # "val": {"select-policy": {"type": "index", "ids": [1, 3, 4, 57, 4, 6, 9]}},
            # "": {"select-policy": {"type": "random", "nums": 10}},
        }
    }

    coco = RemoteCOCO(engine, minio, config)
    dataset_dict = load_remote_coco_json(coco, "new_coco_dataset")
    assert len(dataset_dict) > 0

    cfg = CN()
    add_remote_config(cfg)
    assert hasattr(cfg, "REMOTE")

    cfg.INPUT = CN()
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.FORMAT = "RGB"
    mapper = RemoteCOCOInstanceChromosomeDatasetMapper(cfg, is_train=True)
    for item in dataset_dict[:1]:
        outputs = mapper(item)
        print(outputs)


def test_remote_coco_mapper():
    global engine, minio
    from cocoman.mycoco import RemoteCOCO
    from cocoman.integration.detectron2.remote_coco_dataset import (
        load_remote_coco_json,
        load_remote_coco_json_fast,
    )
    from cocoman.integration.detectron2.remote_coco_mapper import (
        RemoteCOCOInstanceChromosomeDatasetMapper,
    )
    from cocoman.integration.detectron2.config_patch import add_remote_config, CN
    from PIL import Image

    config_subset_random = {
        "20230804-seg-coco": {
            "train": {"select-policy": {"type": "random", "nums": 10}},
            # "val": {"select-policy": {"type":"all"}},
            # "": {"select-policy": {"type": "index", "ids": [1, 3, 4, 57, 4, 6, 9]}},
        }
    }

    subset_random_config = RemoteCOCO(engine, minio, config_subset_random)
    subset_all_config = RemoteCOCO(
        engine,
        minio,
        {
            "20230804-seg-coco": {
                "val": {"select-policy": {"type": "all"}},
            }
        },
    )
    import time

    for coco in [subset_random_config, subset_all_config]:
        tic = time.perf_counter()
        dataset_dict = load_remote_coco_json_fast(coco, "new_coco_dataset", 10)
        toc = time.perf_counter()
        print(f"Load remote coco json fast: {toc - tic:0.4f} seconds")

        tic = time.perf_counter()
        dataset_dict = load_remote_coco_json(coco, "new_coco_dataset", 10)
        toc = time.perf_counter()
        print(f"Load remote coco json: {toc - tic:0.4f} seconds")
        assert len(dataset_dict) > 0

        cfg = CN()
        add_remote_config(cfg)
        assert hasattr(cfg, "REMOTE")

        cfg.INPUT = CN()
        cfg.INPUT.IMAGE_SIZE = 1024
        cfg.INPUT.FORMAT = "RGB"
        mapper = RemoteCOCOInstanceChromosomeDatasetMapper(cfg, is_train=False)
        for item in dataset_dict[:1]:
            outputs = mapper(item)
            assert len(outputs["image"].shape) > 0
