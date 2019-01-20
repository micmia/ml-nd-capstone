# ml-nd-capstone
## 描述
Rossmann 于 2015 年 9 月在 Kaggle 开启一[项目竞赛](https://www.kaggle.com/c/rossmann-store-sales)，即开发一个稳健模型以预测其在德国境内 1115 家商店未来 6 个月的日销售情况。该竞赛已提供 1115 家商店历史销售情况的数据集，该数据集包含特征（例如：商店类型、最近竞争者距离）和标签（即销售额）。本项目即该竞赛解决方案的一种具体实现。

## 软件需求
- Python > 3.0

## 安装
安装 Python 依赖包：

```sh
$ pip install -r requirements.txt
```
## 运行
启动 Jupyter Notebook 环境：

```sh
$ python jupyter-notebook --no-browser --ip 127.0.0.1 --port 8888 --port-retries=0
```

打开 `http://127.0.0.1:8888/notebooks/report.ipynb` 并运行所有单元格。

模型训练过程耗时大约 2.5 小时。

## 文档结构

    .
    ├── data
    │   ├── model.joblib.dat.zip # 最终模型持久化数据（压缩包
    │   ├── store.csv            # 商店补充信息数据集
    │   ├── submission.csv       # 测试集预测结果
    │   ├── test.csv             # 测试集
    │   └── train.csv            # 训练集
    ├── utils
    │   ├── cv.py                # 验证工具模块
    │   ├── model.py             # 模型构建工具模块
    │   └── preprocessing.py     # 预处理工具模块
    ├── report.ipynb             # 项目报告 Jupyter Notebook
    └── report.pdf               # 项目报告
