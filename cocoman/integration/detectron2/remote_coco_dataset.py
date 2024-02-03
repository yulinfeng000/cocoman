import logging
import os
from typing import Tuple, List
import pycocotools.mask as mask_util
import functools
from tqdm import tqdm
from pathlib import Path
import itertools
import msgpack
from joblib import Parallel, delayed
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from cocoman.mycoco import RemoteCOCO, binary_mask_to_polygon
from cocoman.utils import loadRLE


logger = logging.getLogger("cocoman.integration.detectron2.remote_coco")


def record_worker(imgs_anns: Tuple[dict, List[dict]], id_map):
    imgObj, annObjs = imgs_anns
    record = {
        "bucket_name": imgObj["bucket_name"],
        "file_name": imgObj["file_name"],
        "height": imgObj["height"],
        "width": imgObj["width"],
        "image_id": imgObj["id"],
    }
    objs = []
    for anno in annObjs:
        # assert anno.image_id == image_id
        # assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

        obj = {"iscrowd": anno["iscrowd"], "category_id": anno["category_id"]}

        if "bbox" in obj and len(obj["bbox"]) == 0:
            raise ValueError(
                f"One annotation of image {anno['image_id']} contains empty 'bbox' value! "
                "This json does not have valid COCO format."
            )
        else:
            obj["bbox"] = anno['bbox']

        # segm = anno.get("segmentation", None)
        if "segmentation" in anno:
            if anno["iscrowd"]:
                segm = loadRLE(anno["segmentation"])
            else:
                segm = binary_mask_to_polygon(
                    mask_util.decode(loadRLE(anno["segmentation"]))
                )
        else:
            segm = None

        if segm is not None:  # either list[list[float]] or dict(RLE)
            if isinstance(segm, dict):
                if isinstance(segm["counts"], list):
                    # convert to compressed RLE
                    segm = mask_util.frPyObjects(segm, *segm["size"])
            else:
                # filter out invalid polygons (< 3 points)
                segm = [
                    poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                ]
                if len(segm) == 0:
                    continue  # ignore this instance
            obj["segmentation"] = segm

        # TODO:  remote_coco 暂时没有这个keypoints
        keypts = anno.get("keypoints", None)
        if keypts:  # list[int]
            for idx, v in enumerate(keypts):
                if idx % 3 != 2:
                    # COCO's segmentation coordinates are floating points in [0, H or W],
                    # but keypoint coordinates are integers in [0, H-1 or W-1]
                    # Therefore we assume the coordinates are "pixel indices" and
                    # add 0.5 to convert to floating point coordinates.
                    keypts[idx] = v + 0.5
            obj["keypoints"] = keypts

        obj["bbox_mode"] = BoxMode.XYWH_ABS
        if id_map:
            annotation_category_id = obj["category_id"]
            try:
                obj["category_id"] = id_map[annotation_category_id]
            except KeyError as e:
                raise KeyError(
                    f"Encountered category_id={annotation_category_id} "
                    "but this id does not exist in 'categories' of the json file."
                ) from e
        objs.append(obj)
    record["annotations"] = objs
    return record


def load_remote_coco_json_fast(
    remote_coco: RemoteCOCO,
    dataset_name,
    extra_annotation_keys=None,
    workers=os.cpu_count(),
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    coco_api = remote_coco

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x['id'])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    "Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you."
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs)
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.loadAnns(coco_api.imgToAnns[img_id]) for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{dataset_name} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), dataset_name)
    )
    dataset_dicts = []

    # ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (
    #     extra_annotation_keys or []
    # )

    dataset_dicts = Parallel(n_jobs=-1)(
        delayed(functools.partial(record_worker, id_map=id_map))(img_anns)
        for img_anns in tqdm(imgs_anns,desc="Processing Records")
    )

    return dataset_dicts


def load_remote_coco_json(
    remote_coco: RemoteCOCO, dataset_name, extra_annotation_keys=None
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    coco_api = remote_coco

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c['name'] for c in sorted(cats, key=lambda x: x['id'])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    "Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you."
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs)
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{dataset_name} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), dataset_name)
    )
    dataset_dicts = []

    # TODO: 暂不支持extra_annotation_keys
    # ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    num_instances_without_valid_segmentation = 0
    for i, (imgObj, annIds) in enumerate(imgs_anns, 1):
        record = {}
        record["bucket_name"] = imgObj['bucket_name']
        record["file_name"] = imgObj['file_name']
        record["height"] = imgObj['height']
        record["width"] = imgObj['width']
        image_id = record["image_id"] = imgObj['id']

        objs = []
        # 确保 annotation 被 session 接管
        annObjs = coco_api.loadAnns(annIds)
        logger.info(
            f"[{i}/{len(imgs_anns)}] {imgObj['bucket_name']}/{imgObj['file_name']}, num of annotation: {len(annObjs)}"
        )
        for anno in annObjs:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno['image_id'] == image_id
            # assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {"iscrowd": anno['iscrowd'], "category_id": anno['category_id']}
            if hasattr(anno, "bbox") and len(anno['bbox']) == 0:
                # if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )
            else:
                obj["bbox"] = anno['bbox']

            segm = anno.get("segmentation", None)

            if segm is not None:  # either list[list[float]] or dict(RLE)
                if anno['iscrowd']:
                    segm = coco_api.annToRLE(anno)
                else:
                    segm = binary_mask_to_polygon(coco_api.annToMask(anno))

                obj["segmentation"] = segm

            # TODO:  remote_coco 暂时没有这个keypoints
            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def cache_or_load_remote_coco_json_fast(remote_coco, name, cache_dir):
    if cache_dir is None:
        return load_remote_coco_json_fast(remote_coco, name)

    cache_file = Path(cache_dir).joinpath(f"{name}.msgpack")

    if not cache_file.parent.exists():
        cache_file.parent.mkdir(parents=True)

    if cache_file.exists():
        with open(str(cache_file), "rb") as f:
            return msgpack.unpack(f)
    else:
        print("cache dataset not found, will load it")
        results = load_remote_coco_json_fast(remote_coco, name)
        with open(str(cache_file), "wb") as f:
            msgpack.pack(results, f)
        return results


def register_remote_coco_instances(name, metadata, remote_coco, cache_dir="/tmp/"):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    # 1. register a function which returns dicts

    DatasetCatalog.register(
        name, lambda: cache_or_load_remote_coco_json_fast(remote_coco, name, cache_dir)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        remote_coco=remote_coco, cache_dir=cache_dir, evaluator_type="coco", **metadata
    )
