# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import os
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from pycocotools import mask as coco_mask
from minio import Minio
from PIL import Image
import io
from urllib3.response import HTTPResponse
from pathlib import Path

__all__ = ["COCOInstanceChromosomeDatasetMapper"]

logger = logging.getLogger("cocoman.integration.detectron2.remote_coco_mapper")


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    # assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    if isinstance(image_size, tuple) or isinstance(image_size, list):
        image_size = image_size[0]

    if is_train:
        augmentation = [
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomFlip(),
            T.RandomFlip(horizontal=False, vertical=True),
            T.RandomRotation(angle=[-90, 90], expand=True),
            T.ResizeScale(
                min_scale=0.5,
                max_scale=2.0,
                target_width=image_size,
                target_height=image_size,
            ),
            T.Resize((image_size, image_size)),
        ]
    else:
        augmentation = [
            T.ResizeScale(
                min_scale=0.5,
                max_scale=2.0,
                target_width=image_size,
                target_height=image_size,
            ),
            T.Resize((image_size, image_size)),
        ]

    return augmentation


# This is specifically designed for the COCO dataset.
class RemoteCOCOInstanceDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        minio,
        temp_dir,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceChromosomeDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.minio:Minio = minio
        self.temp_dir = temp_dir

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "minio":  Minio(
                cfg.REMOTE.MINIO_URL,
                access_key=cfg.REMOTE.MINIO_ACCESS_KEY,
                secret_key=cfg.REMOTE.MINIO_SECRET_KEY,
                secure=cfg.REMOTE.MINIO_SSL,
            ),
            "temp_dir": cfg.REMOTE.IMG_TEMP_DIR,
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = None
        if self.temp_dir:
            temp_dir = Path(self.temp_dir)
            if not temp_dir.exists():
                temp_dir.mkdir(parents=True)

            temp_path = temp_dir.joinpath(
                dataset_dict["bucket_name"], dataset_dict["file_name"]
            )

            if not temp_path.exists():
                self.minio.fget_object(
                    dataset_dict["bucket_name"],
                    dataset_dict["file_name"],
                    str(temp_path),
                )

            image = utils.read_image(str(temp_path.absolute()), format=self.img_format)

        else:
            try:
                resp: HTTPResponse = self.minio.get_object(
                    dataset_dict["bucket_name"], dataset_dict["file_name"]
                )
                if resp.status == 200:
                    image = utils.convert_PIL_to_numpy(
                        Image.open(io.BytesIO(resp.read())), self.img_format
                    )
            finally:
                resp.close()
                resp.release_conn()

        if image is None:
            raise RuntimeError(
                f"load image: {dataset_dict['bucket_name']}/{dataset_dict['file_name']} is None which should be transferred by web or local fs"
            )

        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["padding_mask"] = torch.as_tensor(
            np.ascontiguousarray(padding_mask)
        )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, "gt_masks"):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances

        return dataset_dict
