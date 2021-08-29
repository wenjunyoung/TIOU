# TIoU  on Python Multiprocessing

此仓库为TIOU评价方式的python多进程代码库。多进程模式包含两种模式：1. 将数据切分为若干份并分别用一个进程处理，最后汇总结算结果。2. 生产者-消费者模式+队列模式。默认为第一种模式。

# 快速开始

Install prerequisite packages:
```shell
pip install shapely Polygon3
```

Test on IC15:
```shell
python ic15/script.py -g=./ic15/gt.zip -s=./ic15/pixellinkch4.zip
```

Test on CTW1500:
```shell
python curved-tiou/script.py -g=ctw1500-gt.zip -s=det_ctw1500.zip
```

Test on Total-Text:
```shell
python curved-tiou/script.py -g=total-text-gt.zip -s=total-text_baseline.zip
```

