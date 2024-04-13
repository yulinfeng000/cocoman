from setuptools import setup, find_packages


setup(
    name="cocoman",
    version="1.2",
    description="make coco cool again",
    author="cam",
    author_email="yulinfeng000@gmail.com",
    packages=find_packages(exclude=["server"]),
    entry_points={"console_scripts": ["cocoman=cocoman.cmd:main"]},
    install_requires=[
        "numpy",
        "SQLAlchemy",
        "minio",
        "psycopg2-binary",
        "pycocotools",
        "pymongo",
        "motor",
        "aiofiles",
        "msgpack",
        "joblib",
        "tqdm",
        "opencv-python",
        "scikit-learn",
        "requests",
    ],
)
