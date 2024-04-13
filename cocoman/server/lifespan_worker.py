from uvicorn.workers import UvicornWorker

CONFIG_KWARGS = UvicornWorker.CONFIG_KWARGS
CONFIG_KWARGS.update({"lifespan": "on"})


class LifespanUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = CONFIG_KWARGS
