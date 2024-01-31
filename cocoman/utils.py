import json
import base64
from minio import Minio
from minio.error import S3Error
from sqlalchemy.sql import expression
from sqlalchemy.ext.compiler import compiles


def dumpRLE(rle)->str:
    rle["counts"] = base64.b64encode(rle["counts"]).decode("utf-8")
    return json.dumps(rle)


def loadRLE(json_rle)->dict:
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

class array_sample(expression.Executable,expression.ColumnElement):
    inherit_cache = True

    def __init__(self,array_column, samples:int):
        self.array_column = array_column
        self.num_samples = samples


@compiles(array_sample,'postgresql')
def compile_array_sample(element, compiler, **kw):
    return 'array_sample( %s , %d )' % ( compiler.process(element.array_column, asfrom=True, **kw),  element.num_samples )
