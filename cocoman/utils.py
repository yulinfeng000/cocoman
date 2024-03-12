import json
import base64
from minio import Minio
from minio.error import S3Error
from sqlalchemy import create_engine as _create_engine
from sqlalchemy.sql import expression
from sqlalchemy.ext.compiler import compiles


def dumpRLE(rle) -> str:
    rle["counts"] = base64.b64encode(rle["counts"]).decode("utf-8")
    return json.dumps(rle)


def loadRLE(json_rle) -> dict:
    rle = json.loads(json_rle)
    rle["counts"] = base64.b64decode(rle["counts"].encode("utf-8"))
    return rle


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


class array_sample(expression.Executable, expression.ColumnElement):
    inherit_cache = True

    def __init__(self, array_column, samples: int):
        self.array_column = array_column
        self.num_samples = samples


@compiles(array_sample, "postgresql")
def compile_array_sample(element, compiler, **kw):
    return "array_sample( %s , %d )" % (
        compiler.process(element.array_column, asfrom=True, **kw),
        element.num_samples,
    )


def create_db_engine(
    db_url: str = None,
    db_host: str = None,
    db_port: int = None,
    db_name: str = None,
    db_username: str = None,
    db_password: str = None,
    **kwargs,
):
    if db_url is None:
        assert all(
            [db_host, db_port, db_name, db_username, db_password]
        ), "db_host,db_port,db_name,db_username,db_password must all none empty!"
        
        return _create_engine(
            f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}",
            **kwargs,
        )
    else:
        return _create_engine(f"postgresql+psycopg2://{db_url}", **kwargs)


def create_minio(
    minio_url: str, minio_access_key: str, minio_secret_key: str, ssl: str = False
):
    return Minio(
        minio_url,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=ssl,
    )
