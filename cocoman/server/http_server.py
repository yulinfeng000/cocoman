from collections import defaultdict
import json
from typing import Optional, TypedDict, Dict, Literal, Any, List, Union
from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import StreamingResponse, Response, ORJSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from cocoman.common.utils import (
    create_mongodb_db_async,
    create_minio,
    async_dump_big_json_list_stream,
)
from cocoman.common.settings import (
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SSL,
    MINIO_URL,
    MONGO_DB_NAME,
    MONGO_DB_URL,
)
from contextvars import ContextVar
from motor.motor_asyncio import AsyncIOMotorDatabase
from concurrent.futures import ThreadPoolExecutor
from minio import Minio
import asyncio
from fastapi.middleware.gzip import GZipMiddleware
from cocoman.common.utils import AdvJSONEncoder
from bson import ObjectId
from pathlib import Path

MONGO_DB: ContextVar[AsyncIOMotorDatabase] = ContextVar("MONGO_DB", default=None)
MINIO: ContextVar[Minio] = ContextVar("MINIO", default=None)
EXECUTOR: ContextVar[ThreadPoolExecutor] = ContextVar("EXECUTOR", default=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.thread_pool = ThreadPoolExecutor(max_workers=10)
    app.state.minio = create_minio(
        minio_url=MINIO_URL,
        minio_access_key=MINIO_ACCESS_KEY,
        minio_secret_key=MINIO_SECRET_KEY,
        ssl=MINIO_SSL,
    )
    app.state.mongo = create_mongodb_db_async(url=MONGO_DB_URL, db_name=MONGO_DB_NAME)
    yield
    app.state.thread_pool.shutdown()
    print("shutdown process pool")


app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)


class CreateRemoteIndexReq(BaseModel):
    config: Dict


class GetAnnIdsReq(BaseModel):
    # imgIds: Optional[List[str]] = []
    catIds: Optional[List[str]] = []
    areaRng: Optional[List[float]] = []
    iscrowd: Optional[bool] = None


class GetCatIdsReq(BaseModel):
    catNms: Optional[List[str]] = []
    supNms: Optional[List[str]] = []
    catIds: Optional[List[str]] = []


class LoadCommonItemReq(BaseModel):
    ids: Optional[List[str]] = []


class GetImgPicReq(BaseModel):
    imgId: str


class UploadImgPicReq(BaseModel):
    pass


class UploadImgReq(BaseModel):
    file_name: str
    width: int
    height: int
    bucket_name: str


class UploadCatReq(BaseModel):
    name: str
    supercategory: str


class UploadAnnReq(BaseModel):
    image_id: str
    category_id: str
    segmentation: Dict[str, Any]
    bbox: List[Union[int, float]]
    area: Union[int, float]
    iscrowd: bool


class UploadAnnsReq(BaseModel):
    anns: List[UploadAnnReq]


class UploadDatasetReq(BaseModel):
    dataset_name: str
    dataset_type: str
    image_ids: List[str]


async def get_images_by_config(
    dataset: str, subset: str, config: dict, db: AsyncIOMotorDatabase
):
    policy = config["select-policy"]
    policy_type = policy["type"]

    if subset:
        if policy_type == "random":
            imgIds = [
                it["image_ids"]
                async for it in db["datasets"].aggregate(
                    [
                        {"$match": {"dataset_type": subset, "dataset_name": dataset}},
                        {"$unwind": {"path": "$image_ids"}},
                        {"$project": {"image_ids": 1}},
                        {"$sample": {"size": policy["nums"]}},
                    ]
                )
            ]
            if not imgIds:
                raise Exception("not found")
            return imgIds

        elif policy_type == "all":
            imgIds = [
                i["image_ids"]
                async for i in db["datasets"].aggregate(
                    [
                        {"$match": {"dataset_type": subset, "dataset_name": dataset}},
                        {"$unwind": {"path": "$image_ids"}},
                        {
                            "$project": {
                                "image_ids": 1,
                                "_id": 0,
                            }
                        },
                    ]
                )
            ]
            return imgIds

        elif policy_type == "index":
            return [ObjectId(i) for i in policy["ids"]]

        else:
            raise NotImplementedError(f"policy type {policy_type} not implemented")

    else:
        if policy_type == "random":
            imgIds = [
                i["_id"]
                async for i in db["images"].aggregate(
                    [
                        {"$match": {"bucket_name": dataset}},
                        {"$sample": {"size": policy["nums"]}},
                        {"$project": {"_id": 1}},
                    ]
                )
            ]
            return imgIds

        elif policy_type == "all":
            imgIds = [
                i["_id"]
                async for i in db["images"].aggregate(
                    [
                        {"$match": {"bucket_name": dataset}},
                        {"$project": {"_id": 1}},
                    ]
                )
            ]
            return imgIds

        elif policy_type == "index":
            return [ObjectId(i) for i in policy["ids"]]

        else:
            raise NotImplementedError(f"policy type {policy_type} not implemented")


@app.middleware("http")
async def middleware(req: Request, call_next):
    # Load the ML model
    MINIO.set(app.state.minio)
    MONGO_DB.set(app.state.mongo)
    EXECUTOR.set(app.state.thread_pool)
    return await call_next(req)


app.add_middleware(GZipMiddleware, minimum_size=1000)


async def _loadByType(
    db: AsyncIOMotorDatabase, collection: str, ids: List[ObjectId] = []
) -> List[Dict]:
    return (
        await db[collection]
        .aggregate(
            [
                {"$match": {"_id": {"$in": ids}}},
                {"$addFields": {"id": {"$toString": "$_id"}}},
                {"$project": {"_id": 0}},
            ]
        )
        .to_list(None)
    )


async def _loadByTypeStream(
    db: AsyncIOMotorDatabase, collection: str, ids: List[ObjectId] = []
):
    async for item in db[collection].aggregate(
        [
            {"$match": {"_id": {"$in": ids}}},
            {"$addFields": {"id": {"$toString": "$_id"}}},
            {"$project": {"_id": 0}},
        ]
    ):
        yield item


@app.post("/createIndex")
async def createIndex(req: CreateRemoteIndexReq):
    db = MONGO_DB.get()

    config = req.config
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    imgs, anns, cats = [], [], []

    for dataset_name, select_config in config.items():
        for subset, config in select_config.items():
            imgs.extend(await get_images_by_config(dataset_name, subset, config, db))

    annotations = db["annotations"].aggregate([{"$match": {"image_id": {"$in": imgs}}}])

    async for ann in annotations:
        anns.append(ann["_id"])
        imgToAnns[str(ann["image_id"])].append(str(ann["_id"]))
        catToImgs[str(ann["category_id"])].append(str(ann["image_id"]))

    cats = await db["annotations"].distinct("category_id")

    results = {
        "imgToAnns": imgToAnns,
        "catToImgs": catToImgs,
        "imgs": imgs,
        "anns": anns,
        "cats": cats,
    }

    return Response(
        json.dumps(results, cls=AdvJSONEncoder),
        headers={"Content-Type": "application/json"},
    )


@app.post("/getAnnIds")
async def getAnnIds(req: GetAnnIdsReq):
    catIds = req.catIds
    areaRng = req.areaRng
    iscrowd = req.iscrowd
    db = MONGO_DB.get()

    match_conditions = []
    if len(catIds) != 0:
        match_conditions.append({"category_id": {"$in": catIds}})

    if len(areaRng) != 0:
        match_conditions.append({"area": {"$gt": areaRng[0], "$lte": areaRng[1]}})

    if iscrowd is not None:
        match_conditions.append({"iscrowd": iscrowd})

    if len(match_conditions) > 1:
        match_conditions = {"$and": match_conditions}
    elif len(match_conditions) == 1:
        match_conditions = match_conditions[0]
    else:
        match_conditions = None

    pipline = [
        {"$addFields": {"id": {"$toString": "$_id"}}},
        {"$project": {"id": 1, "_id": 0}},
    ]
    if match_conditions:
        pipline.insert(0, {"$match": match_conditions})

    results = await db["annotations"].aggregate(pipline).to_list(None)
    return [i["id"] for i in results]


@app.post("/getCatIds")
async def getCatIds(req: GetCatIdsReq):
    catNms = req.catNms
    supNms = req.supNms
    catIds = req.catIds
    db = MONGO_DB.get()

    match_conditions = []

    if len(catNms) != 0:
        # Category.name.in_(catNms)
        match_conditions.append({"name": {"$in": catNms}})

    if len(supNms) != 0:
        # where_conditions.append(Category.super_category.in_(supNms))
        match_conditions.append({"supercategory": {"$in": supNms}})

    if len(catIds) != 0:
        match_conditions.append({"_id": {"$in": catIds}})

    if len(match_conditions) > 1:
        match_conditions = {"$and": match_conditions}
    elif len(match_conditions) == 1:
        match_conditions = match_conditions[0]

    pipline = [
        {"$addFields": {"id": {"$toString": "$_id"}}},
        {"$project": {"id": 1, "_id": 0}},
    ]
    if len(match_conditions) >= 1:
        pipline.insert(0, {"$match": match_conditions})
    results = await db["categories"].aggregate(pipline).to_list(None)

    return [r["id"] for r in results]


@app.post("/loadAnns")
async def loadAnns(req: LoadCommonItemReq) -> List[Dict]:
    ids = [ObjectId(i) for i in req.ids]
    db = MONGO_DB.get()

    async def generator():
        async for item in db["annotations"].aggregate(
            [
                {"$match": {"_id": {"$in": ids}}},
                {"$addFields": {"id": {"$toString": "$_id"}}},
                {"$project": {"_id": 0}},
                {
                    "$set": {
                        "image_id": {"$toString": "$image_id"},
                        "category_id": {"$toString": "$category_id"},
                    }
                },
            ]
        ):
            yield item

    return StreamingResponse(
        async_dump_big_json_list_stream(generator()),
        headers={"Content-Type": "application/json"},
    )


# @app.post("/loadAnns")
# async def loadAnns(req: LoadCommonItemReq) -> List[Dict]:
#     ids = [ObjectId(i) for i in req.ids]
#     db = MONGO_DB.get()
#     return list(
#         db["annotations"].aggregate(
#             [
#                 {"$match": {"_id": {"$in": ids}}},
#                 {"$addFields": {"id": {"$toString": "$_id"}}},
#                 {"$project": {"_id": 0}},
#                 {
#                     "$set": {
#                         "image_id": {"$toString": "$image_id"},
#                         "category_id": {"$toString": "$category_id"},
#                     }
#                 },
#             ]
#         )
#     )


@app.post("/loadCats")
async def loadCats(req: LoadCommonItemReq) -> List[Dict]:
    ids = [ObjectId(i) for i in req.ids]
    generator = _loadByTypeStream(MONGO_DB.get(), "categories", ids)
    return StreamingResponse(
        async_dump_big_json_list_stream(generator),
        headers={"Content-Type": "application/json"},
    )


# @app.post("/loadCats")
# async def loadCats(req: LoadCommonItemReq) -> List[Dict]:
#     ids = [ObjectId(i) for i in req.ids]
#     return _loadByType(MONGO_DB.get(), "categories", ids)


@app.post("/loadImgs")
async def loadImgs(req: LoadCommonItemReq) -> List[Dict]:
    ids = [ObjectId(i) for i in req.ids]
    generator = _loadByTypeStream(MONGO_DB.get(), "images", ids)
    return StreamingResponse(
        async_dump_big_json_list_stream(generator),
        headers={"Content-Type": "application/json"},
    )


@app.post("/getImgPic")
async def getImgPic(req: GetImgPicReq):
    imgId = req.imgId
    db = MONGO_DB.get()
    target = (await db["images"].find({"_id": ObjectId(imgId)}).to_list(None))[0]
    bucket_name, file_name = target["bucket_name"], target["file_name"]
    resp = MINIO.get().get_object(bucket_name, file_name)
    return StreamingResponse(
        resp.stream(), headers={"Content-Type": f"image/{Path(file_name).suffix[1:]}"}
    )


@app.post("/uploadImgPic")
async def uploadImgPic(
    img_file: UploadFile = File(...),
    dataset_name: str = Form(...),
) -> None:
    minio = MINIO.get()
    if not minio.bucket_exists(dataset_name):
        minio.make_bucket(dataset_name)

    loop = asyncio.get_event_loop()
    # print(f"image/{Path(img_file.filename).suffix[1:]}")
    await loop.run_in_executor(
        EXECUTOR.get(),
        lambda: minio.put_object(
            bucket_name=dataset_name,
            object_name=img_file.filename,
            data=img_file.file,
            length=img_file.size,
        ),
    )
    return {}, 200


@app.post("/uploadImg")
async def uploadImg(req: UploadImgReq):
    db = MONGO_DB.get()
    results = await db["images"].insert_one(req.model_dump())
    return {"id": str(results.inserted_id)}


@app.post("/uploadCat")
async def uploadCat(req: UploadCatReq):
    db = MONGO_DB.get()
    item = {"name": req.name, "supercategory": req.supercategory}
    cat = await db["categories"].find_one(item)
    if cat is None:
        result = await db["categories"].insert_one(item)
        return {"id": str(result.inserted_id)}
    else:
        return {"id": str(cat["_id"])}


@app.post("/uploadAnn")
async def uploadAnn(req: UploadAnnReq):
    db = MONGO_DB.get()
    results = await db["annotations"].insert_one(req.model_dump())
    return {"id": str(results.inserted_id)}


@app.post("/uploadAnns")
async def uploadAnns(req: UploadAnnsReq):
    db = MONGO_DB.get()
    anns = [ann.model_dump() for ann in req.anns]
    results = await db["annotations"].insert_many(anns)
    return {"id": [str(i) for i in results.inserted_ids]}


@app.post("/uploadDataset")
async def uploadDataset(req: UploadDatasetReq):
    db = MONGO_DB.get()
    results = await db["dataset"].insert_one(req.model_dump())
    return {"id": str(results.inserted_id)}
