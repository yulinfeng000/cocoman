import re
import datetime
import numpy as np
from itertools import groupby
import cv2
from PIL import Image
from pycocotools import mask as coco_mask

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8) * 255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return rle


def binary_mask_to_polygon(binary_mask, tolerance=1):
    binary_mask = np.ascontiguousarray(binary_mask)
    binary_mask = np.uint8(binary_mask)
    contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if tolerance <= 0:
        return [cnt.flatten().tolist() for cnt in contours]
    else:
        return [
            cv2.approxPolyDP(cnt, tolerance, closed=True).flatten().tolist()
            for cnt in contours
        ]


def create_image_info(
    image_id,
    file_name,
    image_size,
    date_captured=datetime.datetime.utcnow().isoformat(" "),
    license_id=1,
    coco_url="",
    flickr_url="",
):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }

    return image_info


def create_annotation_info(
    annotation_id,
    image_id,
    category_info,
    binary_mask,
    image_size=None,
    tolerance=2,
    bounding_box=None,
):
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = coco_mask.encode(
        np.asfortranarray(binary_mask.astype(np.uint8))
    )

    area = coco_mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = coco_mask.toBbox(binary_mask_encoded)

    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if segmentation is None:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info
