import json
import base64
from typing import Any
from fastapi.responses import ORJSONResponse
from fastapi.encoders import jsonable_encoder
from minio import Minio
from minio.error import S3Error
import bson
import orjson
from motor.motor_asyncio import AsyncIOMotorClient


def object_exists(cli: Minio, bucket_name, object_name):
    # follow solution with https://github.com/minio/minio-go/issues/1082#issuecomment-468215014
    try:
        cli.stat_object(bucket_name, object_name)
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        else:
            raise e
    return True


def create_minio(
    minio_url: str, minio_access_key: str, minio_secret_key: str, ssl: str = False
):
    return Minio(
        minio_url,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=ssl,
    )

def create_mongodb_db_async(url: str, db_name: str):
    return AsyncIOMotorClient(url).get_database(db_name)


class AdvJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bson.ObjectId):
            return str(o)
        else:
            return super().default(o)


def default_json(obj):
    if isinstance(obj, bson.ObjectId):
        return str(obj)
    raise TypeError


class AdvJSONResponse(ORJSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(
            content, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
        )


def dump_big_json_list_stream(generator):
    yield "["
    first = True
    for item in generator:
        if not first:
            yield ","
        else:
            first = False
        yield orjson.dumps(
            jsonable_encoder(item),
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        ).decode("utf-8")
    yield "]"


async def async_dump_big_json_list_stream(generator):
    yield "["
    first = True
    async for item in generator:
        if not first:
            yield ","
        else:
            first = False
        yield orjson.dumps(
            jsonable_encoder(item),
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        ).decode("utf-8")
    yield "]"
