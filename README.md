 
# 基于车载视频图像分析动态路况
# Dynamic Road Condition Analysis Based on Vehicle Video Images


## Mission explanation:
A number of images of dynamic traffic condition captured by vehicle cameras are provided. Based on the pictures, the model should distinguish the current road condition. To simplify it, traffic conditions are divided into three types: smooth, slow and congested.

## 任务描述：
根据所提供的车载视频图像分析并判断当前交通状况。交通状况共分为三种：畅通，缓行和拥堵


## Data explanation: 
There are 1500 sequences of vehicle video images in total for training. In each sequence, 3 ~ 5 images are included. The images in one sequence are captured in the same short period. Each sequence has a label. Labels are: "0","1","2", representing smooth, slow and congested separately.

## 数据描述：
训练集共1500个序列。每个序列包含3到5张车载视频图片。同一序列的图片是在同一段时间内，同一车辆，同一视角拍摄完成。每个序列都带有标签。标签有三种：“0” “1” “2”，分别代表畅通,缓行和拥堵。

## Main idea of solving the problem:
This project solves the problem by using CNN+RNN. Specifically, resnet50 and LSTM are used. First, use resnet50 to extract the feature vector of every image and flatten them into one dimension vector. Notice that, to simplify the training, I change the number of images of every sequence into four.(If there are 5 images, delete the last one;if three, copy the last one). After obtaining the feature vectors, pack them into tensors. One sequence (four feature vectors) is packed into one tensor. Then,send these tensors into LSTM and add Linear Layer and softmax to do the classification.

## 整体思路：
该项目采用方法为CNN+RNN的思路。特别的，选取了Resnet50和LSTM的网络架构。先用resnet50提取每一张图片的特征图并处理成一维向量。之后将同一sequence的特征向量打包进一张tensor并将tensor传入LSTM网络中进行训练。LSTM网络后还加了一些Linear层和Softmax以便直接进行分类。

## 文件：
共四个文件， CNN.py以及LSTM.py <br>
CNN.py用以提取特征向量，输出为1*32720的一维向量，保存为pkl文件形式
RNN.py用以训练模型，需先读取CNN输出的pkl文件
data.txt:数据集下载地址
status.txt:标签文件




