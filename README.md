# 2023-spring-CV-PJ2-ex2

## ①训练步骤：
·打开文件夹中的train\_frcnn.py或train\_retinane.py文件

·将需要识别的类别名称存储为txt文件形式，并将地址传入“classes_path”变量

·提前下载pytorch中提供的模型预训练权重，并将路径传入“model_path”

·分别将含有训练集和验证集图像annotation的的txt文件所在地址传入“train\_annotation_path”变量和“val\_annotation_path”

·设置和调整GPU参数、训练参数、保存路径等

·运行

## ②测试步骤
·打开predict.py文件

·选择model_name为“frcnn”或者“retinanet”

·依次修改需要生成建议框的图像名称或修改图像存储路径，最好以1~n的形式命名图像名称。

·运行
