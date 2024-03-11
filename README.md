# COCOMAN is the Better COCO


##  安装

-  直接安装

    `pip install git+http://{username}:{password}@voyager.orientsoft.cn/yulinfeng/cocoman.git`

- 克隆仓库安装

    `git clone http://voyager.orientsoft.cn/yulinfeng/cocoman.git  && pip install -e cocoman`

## 命令行

- 制作数据集

    命令行执行 `cocoman make -h` 来查看数据集制备工具命令行参数

- 上传数据集

    命令行执行 `cocoman upload -h` 来查看数据集制备工具命令行参数


## 通过环境变量配置全局配置

- DB_URL

    需要 sqlalchemy 的 url 格式, eg: "postgresql+psycopg2://postgres:iamroot@10.8.0.17:5432/coco"

- DB_POOL_SIZE

    数据集的连接池大小

- MINIO_URL
    
    minio 的 url 格式, eg: "10.8.0.17:9900"

- MINIO_ACCESS_KEY

    minio 的 access key (username)

- MINIO_SECRET_KEY

    minio 的 secret key (username)

- MINIO_SSL

    minio 是否使用ssl协议 (true or false)


## 数据集生成配置格式

一个简单的配置示例

```yml
{
    "20240101-seg-coco": {
        "train": {
            "select-policy": { "type": "random", "nums": 100 }
        },
        "val": {
            "select-policy": { "type": "all" }
        }
    },
    "20240814-seg-coco": {
        "": {
            "select-policy": { "type": "all" }
        }
    }
}
```

尝试对配置进行解释:

```
"20240101-seg-coco":            要使用的数据集
    
    "train":                    表示使用夫数据集的训练子集。
    
        "select-policy":        选择策略的配置。
            "type": "random"    选择策略的类型，这里是"random"，表示随机选择。
            "nums": 100         随机选择的数量，这里是100个。

    "val":                      表示使用夫数据集的验证子集。
        "select-policy": 
            "type":" "all"      选择策略的类型，这里是"all"，表示选择所有样本。


"20240814-seg-coco":
    "":                         表示选择该数据集中的所有图片。
        "select-policy":        选择策略的配置。
            "type": "all"       选择策略的类型，这里是"all"，表示选择所有样本。
```


## RemoteCOCO

> cocoman.mycoco.remote_coco.RemoteCOCO

- `def __init__(self, db, minio, config)`

    db:         数据库sqlalchemy engine 实例
    minio:      Minio 实例
    config:     数据集生成配置

- `def save(self, annFilePath, imgDir)`

    annFilePath: 要将生成的coco数据集json文件保存路径
    imgDir: 图片要保存的文件夹路径

- 其余方法和COCO基本保持一致


## 与 detectron2 整合

提供dataset_mapper,dataset,config_path三种组件

- **cocoman.integration.detectron2::RemoteCOCOInstanceDatasetMapper**

    继承该类可以重写from_config方法来达到自定义transform的效果
    ```python
    from cocoman.integration.detectron2.remote_coco_mapper import RemoteCOCOInstanceDatasetMapper

    def build_transform_gen(cfg,is_train):
        # TODO: 在这里构建数据增广...
        pass

    class RemoteChromosomeCOCOInstanceDatasetMapper(RemoteCOCOInstanceDatasetMapper):

        @classmethod
        def from_config(cls, cfg, is_train=True):
            # Build augmentation
            tfm_gens = build_transform_gen(cfg, is_train)
            ret = super().from_config(cfg, is_train)
            ret["tfm_gens"]=tfm_gens
            return ret
    ```

- **cocoman.integration.detectron2::register_remote_coco_instances**

     使用该方法来注册数据集

    `def register_remote_coco_instances(name, metadata, remote_coco, cache_dir="/tmp/")`
    
        name:                   要注册在detectron2框架中的数据集的名字
        metadata:               detectron2 metadata
        remote_coco:            RemoteCOCO实例
        cache_dir:              数据集的缓存目录


- **cocoman.integration.detectron2::add_remote_config**

     使用该方法来注册数据集,一般在train_net::setup方法中使用

    ```python

    def setup(args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        #cfg.MODEL.BACKBONE.__LAZY__ = "" # update for mask2former
        # for poly lr schedule
        #add_deeplab_config(cfg)
        #add_maskformer2_config(cfg)
        #override_mask_head_config(cfg) # must be after add_maskformer2_config
        #add_sam_config(cfg)
        #add_vit_config(cfg)
        add_remote_config(cfg)  
        #cfg.merge_from_file(args.config_file)
        #cfg.merge_from_list(args.opts)
        cfg.freeze() # 一定要在cfg.freeze()之前patch
        #default_setup(cfg, args)
        # Setup logger for "mask_former" module
        #setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
        return cfg
    ```