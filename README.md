# COCOMAN is the Better COCO

## 编译容器

在项目根目录下执行`docker build --tag cam/cocoman-server:0.1 .`进行容器镜像编译

## 部署

在项目docker文件夹下执行`docker compose up -d` 进行容器部署

## 命令行
- 上传数据集

    命令行执行 `cocoman upload -h` 来查看数据集制备工具命令行参数

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