from pycocotools.coco import COCO as MSCOCO
from pathlib import Path


class LocalCOCO(MSCOCO):
    def __init__(self, dataset_name, dataset_type, annotation_file, image_dir) -> None:
        if isinstance(annotation_file, Path):
            annotation_file = str(annotation_file.absolute())
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