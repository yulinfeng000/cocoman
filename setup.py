from setuptools import setup,find_packages,Extension


setup(
    name='cocoman',
    version='1.0',
    description='make coco cool again',
    author='cam',
    author_email='yulinfeng000@gmail.com',
    packages=find_packages(exclude=['minio-py','cocoapi','psycopg2','sqlalchemy']),
    entry_points={
        'console_scripts':[
            'cocoman=cocoman.commandline:main'
        ]
    },
    install_requires=[
        'SQLAlchemy>=2.0',
        'minio',
        'psycopg2-binary',
        'pycocotools',
        'msgpack'
    ]
)