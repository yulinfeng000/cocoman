"""
制备COCO数据集的工具
"""

# 为了使用项目自带的pycocotools库
# PROJECT_ROOT = "../../"
# import sys
# sys.path.append(PROJECT_ROOT)

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
import sys
import itertools
from pathlib import Path
import glob
import json
import shutil
from collections import OrderedDict
from tqdm import tqdm
import cv2
import numpy as np
import functools
from PIL import Image
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from io import TextIOWrapper
from cocoman.mycoco import (
    create_image_info,
    create_annotation_info,
    resize_binary_mask,
    binary_mask_to_polygon,
    binary_mask_to_rle,
    coco_mask as maskutils,
)
import logging

logger = logging.getLogger('cocoman.commandline.maker')

def readme_template(args,io:TextIOWrapper=None,statistic=None):
    def write_pair(io,k,v):
        io.write("{0:{1}<10}:{2}\n".format(k,chr(12288),v))
    if not io:
        io = sys.stdout
    io.write(f"Chromosome COCO Datasets\n\n\n")
    write_pair(io,"数据集名称",args.dataset_name)
    write_pair(io,"制备时间",datetime.now().strftime('%Y-%M-%d %H:%M:%S'))
    write_pair(io,"中期图路径",os.path.abspath(args.metaphase))
    write_pair(io,"掩码图路径",os.path.abspath(args.mask))
    write_pair(io,"染色体是否多类别标签",args.multi_cls
    )
    write_pair(io,"是否为旋转矩形框",args.rotated_box)
    write_pair(io,"数据集输出路径",os.path.abspath(args.output))
    if statistic:
        io.write("\n\n")
        write_pair(io,"训练集样本数",statistic['num_train_sample'])
        write_pair(io,"验证集样本数",statistic['num_val_sample'])
    # f"""
    # Chromosome COCO Datasets


    # 数据集名称:             {args.dataset_name}
    # 制备时间:               {datetime.now().strftime("%Y-%M-%d %H:%M:%S")}
    

    # 中期图路径:             {os.path.abspath(args.metaphase)}
    # 掩码图路径:             {os.path.abspath(args.mask)}
    # 染色体是否多类别标签:     {args.multi_cls}
    # 是否为旋转矩形框:        {args.rotated_box}


    # 数据集输出路径:          {os.path.abspath(args.output)}
    # """
    
    # if statistic:
    #     temp += f"""
        
    #     # 训练集样本数:           {statistic['num_train_sample']}
    #     # 验证集样本数:           {statistic['num_val_sample']}
    #     # """
    # return temp

def get_args():
    parser = ArgumentParser(
        description="""
    制作chromosome—coco数据集命令行工具
    """
    )
    parser.add_argument("command",help='command',choices=['make'])
    parser.add_argument("--dataset-name",help="要生成的数据集名字")
    parser.add_argument("--metaphase", required=True, help="中期图存放的位置")
    parser.add_argument("--mask", required=True, help="mask存放的位置")
    parser.add_argument("--output", required=True, help="生成的coco数据集存放的位置")
    parser.add_argument(
        "--rotated-box", action="store_true", default=False, help="是否生成旋转矩形框"
    )
    parser.add_argument("--multi-cls", action="store_true", default=False, help="是否生成为多分类")
    parser.add_argument("--pic-suffix",choices=["jpg","png",],default="png",help="中期图的后缀")
    parser.add_argument("--mask-fname-format",choices=["huaxi","taiwan",],default="huaxi",help="mask文件名的格式")
    return parser.parse_args()


def search_pic(root_pics_dir, suffix="png"):
    # 搜索所有suffix图片，返回不含文件后缀的文件名 和 文件路径
    data = [
        (Path(pic_path).stem, pic_path)
        for pic_path in glob.glob(os.path.join(root_pics_dir, f"*.{suffix}"))
    ]
    logging.debug(root_pics_dir)
    return data


def mask_info(mask_fname:str,format:str):

    if format == "huaxi":
        case_id, serial_id, stage, chromo_info, suffix = mask_fname.split(".")
            # 几号染色体,在一张图中的编号
        chromo, cid = chromo_info.split("_")
        return case_id, serial_id, stage, chromo, cid, suffix

    elif format == "taiwan":
        case_id, stage, chromo_info, suffix = mask_fname.split(".")
        chromo, cid = chromo_info.split("_")
        return case_id, 0, stage, chromo, cid, suffix

def find_masks(root_masks_dir, pic_id, chromo_start_from=0,format="huaxi"):
    """
    format: [huaxi,taiwan]
    """

    result = OrderedDict(**{i: [] for i in range(1, 25)})

    # 要制作的染色体的分类的数据集编号是1-24号,但染色体id从0开始编号
    # 兼容代码,染色体原始数据是从0开始标号的
    if chromo_start_from == 0:

        def c_idx(chromo):
            return int(chromo) + 1

    else:

        def c_idx(chromo):
            return chromo

    case_mask_dir = os.path.join(root_masks_dir, pic_id)
    for mask_path in glob.glob(os.path.join(case_mask_dir, f"{pic_id}*")):
        mask_fname = os.path.basename(mask_path)
        # 病例id,图片的序列id,阶段,mask所代表的染色体信息,图片后缀
        case_id, serial_id, stage, chromo, cid, suffix = mask_info(mask_fname,format=format)
        # TODO 后续增加对mar类型的处理
        if chromo == "mar":
            continue
        result[c_idx(chromo)].append(mask_path)
    return result

def create_rotated_annotation_info(
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

    binary_mask_encoded = maskutils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8))
    )

    area = maskutils.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box,area = mask2rotated_box(binary_mask) # maskutils.toBbox(binary_mask_encoded)

    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area,
        "bbox": bounding_box,
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info


def generate_annotation_json(root_pics_dir, root_mask_dir, suffix="png",maskfname_format="huaxi"):

    annotation_idgen = itertools.count(start=1)
    # 分类信息生成
    categories = [
        {
            "supercategory": "chromosome",
            "id": int(cid),
            "name": f"chromosome_{str(cid).zfill(2)}",  # chromosome_01 chromosome_21
        }
        for cid in range(1, 25)
    ]

    images = []
    annotations = []

    for img_idx, (pic_id, pic_path) in enumerate(
        tqdm(search_pic(root_pics_dir, suffix=suffix)), start=1
    ):
        img = Image.open(pic_path)

        img_info = create_image_info(
            image_id=int(img_idx),
            file_name=os.path.basename(pic_path),
            image_size=img.size,
        )
        images.append(img_info)

        masks = find_masks(root_mask_dir, pic_id,format=maskfname_format)

        for chromo, masks_paths in masks.items():
            for mask_path in masks_paths:
                mask = cv2.imread(mask_path)
                bin_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                # print(img.shape)
                mask_info = create_annotation_info(
                    annotation_id=next(annotation_idgen),
                    image_id=int(img_idx),
                    category_info=dict(id=int(chromo), is_crowd=False),
                    binary_mask=bin_mask,
                    image_size=img.size,
                )
                annotations.append(mask_info)

    return dict(images=images, categories=categories, annotations=annotations)


def mask2rotated_box(mask: np.ndarray):
    """
    Args:
        mask: [numpy.ndarry] binary mask

    Returns:
        rotated box : [center_x,center_y,w,h,theta]
        area        : box area
    """
    if mask.dtype != np.uint8:
        mask = np.uint8(mask)
    # RETR_EXTERNAL 只要最外层轮廓
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # 做一点 contour format 的转换
    contours = np.vstack(contours).squeeze()
    rect = cv2.minAreaRect(contours)
    (cx, cy), (w, h), theta = rect
    return [cx, cy, w, h, theta], w * h


def generate_annotation_json_single_cls(root_pics_dir, root_mask_dir, suffix="png",mask_fname_format="huaxi"):
    annotation_idgen = itertools.count(start=1)
    # 分类信息生成
    categories = [
        {
            "supercategory": "chromosome",
            "id": 1,
            "name": f"chromosome",  # chromosome_01 chromosome_21
        }
    ]

    images = []
    annotations = []

    for img_idx, (pic_id, pic_path) in enumerate(
        tqdm(search_pic(root_pics_dir, suffix=suffix)), start=1
    ):
        img = Image.open(pic_path)

        img_info = create_image_info(
            image_id=int(img_idx),
            file_name=os.path.basename(pic_path),
            image_size=img.size,
        )
        images.append(img_info)

        masks = find_masks(root_mask_dir, pic_id,format=mask_fname_format)

        for chromo, masks_paths in masks.items():
            for mask_path in masks_paths:
                mask = cv2.imread(mask_path)
                bin_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                # print(img.shape)
                mask_info = create_annotation_info(
                    annotation_id=next(annotation_idgen),
                    image_id=int(img_idx),
                    category_info=dict(id=1, is_crowd=False),
                    binary_mask=bin_mask,
                    image_size=img.size,
                )
                annotations.append(mask_info)

    return dict(images=images, categories=categories, annotations=annotations)


# multiprocess task worker
def task_worker(
    annotation_id,
    img_idx,
    img_size,
    chromo,
    mask_path,
    gen_mask_fn
):
    mask = cv2.imread(mask_path)
    bin_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_info = gen_mask_fn(
        annotation_id=annotation_id,
        image_id=int(img_idx),
        category_info=dict(id=int(chromo), is_crowd=False),
        binary_mask=bin_mask,
        image_size=img_size,
    )
    return mask_info


def multiprocess_generate_annotation_json(root_pics_dir, root_mask_dir, suffix="png",gen_mask_fn=create_annotation_info,mask_fname_format="huaxi"):
    annotation_idgen = itertools.count(start=1)
    # 分类信息生成
    categories = [
        {
            "supercategory": "chromosome",
            "id": int(cid),
            "name": f"chromosome_{str(cid).zfill(2)}",  # chromosome_01 chromosome_21
        }
        for cid in range(1, 25)
    ]

    images = []
    annotations = []
    tasks = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for img_idx, (pic_id, pic_path) in enumerate(
            tqdm(search_pic(root_pics_dir, suffix=suffix), desc="提交并行任务中"), start=1
        ):
            img = Image.open(pic_path)
            img_info = create_image_info(
                image_id=int(img_idx),
                file_name=os.path.basename(pic_path),
                image_size=img.size,
            )
            images.append(img_info)

            masks = find_masks(root_mask_dir, pic_id,format=mask_fname_format)
            for chromo, masks_paths in masks.items():
                for mask_path in masks_paths:
                    tasks.append(
                        # 提交多进程任务
                        executor.submit(
                            task_worker,
                            annotation_id=next(annotation_idgen),
                            img_idx=img_idx,
                            img_size=img.size,
                            chromo=chromo,
                            mask_path=mask_path,
                            gen_mask_fn=gen_mask_fn
                        )
                    )

        with tqdm(total=len(tasks)) as pbar:
            for future in as_completed(tasks):        
                try:        
                    ann_info = future.result()
                except Exception as e:
                    print(e)
                    return 
                if ann_info:
                    annotations.append(ann_info)
                pbar.update(1)

    return dict(images=images, categories=categories, annotations=annotations)


def multiprocess_generate_annotation_json_single_cls(
    root_pics_dir, root_mask_dir, suffix="png",gen_mask_fn=create_annotation_info,mask_fname_format="huaxi"
):  
    annotation_idgen = itertools.count(start=1)
    # 分类信息生成
    categories = [
        {
            "supercategory": "chromosome",
            "id": 1,
            "name": f"chromosome",
        }
    ]

    images = []
    annotations = []
    tasks = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        #
        for img_idx, (pic_id, pic_path) in enumerate(
            tqdm(search_pic(root_pics_dir, suffix=suffix), desc="提交并行任务中"), start=1
        ):
            img = Image.open(pic_path)
            img_info = create_image_info(
                image_id=int(img_idx),
                file_name=os.path.basename(pic_path),
                image_size=img.size,
            )
            images.append(img_info)

            masks = find_masks(root_mask_dir, pic_id,format=mask_fname_format)
            for chromo, masks_paths in masks.items():
                for mask_path in masks_paths:
                    tasks.append(
                        # 提交多进程任务
                        executor.submit(
                            task_worker,
                            annotation_id=next(annotation_idgen),
                            img_idx=img_idx,
                            img_size=img.size,
                            chromo=1,  # 单分类染色体类别id为1
                            mask_path=mask_path,
                            gen_mask_fn=gen_mask_fn
                        )
                    )

        with tqdm(total=len(tasks), desc="并行任务执行中") as pbar:
            for future in as_completed(tasks):
                ann_info = future.result()
                if ann_info:
                    annotations.append(ann_info)
                pbar.update(1)

    return dict(images=images, categories=categories, annotations=annotations)


def split_and_move_pics(root_pic_dir, train_dir=None, val_dir=None,do_copy_inplace=True,pic_suffix="png"):
    # 2/3 train  1/3 val
    pics = search_pic(root_pic_dir,suffix=pic_suffix)
    print("num of pics:", len(pics))
    train, val = train_test_split(
        pics,
        test_size=0.3,
    )
    if do_copy_inplace:
        for _, t in train:
            shutil.copy(t, os.path.join(train_dir, os.path.basename(t)))
        for _, v in val:
            shutil.copy(v, os.path.join(val_dir, os.path.basename(v)))
    
    return train,val

def dump_annotation(annotation_dir, fname, data):
    os.makedirs(annotation_dir, exist_ok=True)
    with open(os.path.join(annotation_dir, fname), "w") as f:
        json.dump(data, f)


def cmd_entrypoint(args):
    """
    cli 命令行入口
    """
    DATASET_NAME = args.dataset_name
    PIC_ROOT_DIR = args.metaphase
    MASK_ROOT_DIR = args.mask
    OUTPUT_DIR = args.output
    TRAIN_DIR = os.path.join(OUTPUT_DIR, DATASET_NAME,"train")
    VAL_DIR = os.path.join(OUTPUT_DIR,DATASET_NAME, "val")
    ANNOTATION_DIR = os.path.join(OUTPUT_DIR,DATASET_NAME, "annotations")
    PIC_SUFFIX = args.pic_suffix
    MASKFNAME_FORMART = args.mask_fname_format
    readme_template(args)
    # 创建文件夹
    for d in [TRAIN_DIR, VAL_DIR, ANNOTATION_DIR]:
        os.makedirs(d, exist_ok=True)

    train,val = split_and_move_pics(PIC_ROOT_DIR, TRAIN_DIR, VAL_DIR,pic_suffix=PIC_SUFFIX)
    # 记录训练集样本数量
    s = {'num_train_sample':len(train),'num_val_sample':len(val)}
    logging.info(s)
    
    generate_json_func = multiprocess_generate_annotation_json_single_cls
    if args.multi_cls:
        generate_json_func = multiprocess_generate_annotation_json

    partial_params={}
    if args.rotated_box:
        partial_params["gen_mask_fn"]=create_rotated_annotation_info
    partial_params.update(dict(suffix=PIC_SUFFIX,mask_fname_format=MASKFNAME_FORMART))
    generate_json_func = functools.partial(generate_json_func,**partial_params)

    train_json = generate_json_func(TRAIN_DIR, MASK_ROOT_DIR)
    dump_annotation(ANNOTATION_DIR, "chromosome_train.json", train_json) 
    val_json = generate_json_func(VAL_DIR, MASK_ROOT_DIR)
    dump_annotation(ANNOTATION_DIR, "chromosome_val.json", val_json)

    with open(os.path.join(OUTPUT_DIR,DATASET_NAME,"README"),"w") as f:
        readme_template(args,f,s)
    
    print("Done")


if __name__ == "__main__":
    cmd_entrypoint()
