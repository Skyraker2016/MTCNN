# MTCNN state-of-the-art face detection method

## Update 2017.09.25

* 增加了跨平台编译的能力

* 适配至最新版caffe

* 增加了轻量级版本，便于移植到android平台

## 概述

[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)是[Kaipeng Zhang](https://kpzhang93.github.io/)等人提出的多任务级联卷积神经网络进行人脸检测的方法，是迄今为止开放源码的效果最好的人脸检测器之一，[在fddb上有100个误报时的检出率高达90%以上](https://github.com/imistyrain/fddb-windows)，作者提供的版本为[matlab版](https://github.com/kpzhang93/MTCNN_face_detection_alignment),它最终的效果如图所示：

![](https://i.imgur.com/FbglxoX.jpg)


## 运行方法

1.按照[MRHead](https://github.com/imistyrain/MRHead)描述的方法配置好opencv跨平台编译环境

2.编译最新版[caffe](https://github.com/BVLC/caffe)，这个网上已有很多[教程](http://blog.csdn.net/akashaicrecorder/article/details/71016942),恕不赘述
```
git clone https://github.com/BVLC/caffe
cd caffe
git checkout windows
script\build_win.cmd
```

3.打开MTCNN.sln，把MTCNN设为启动项。

4.设置所需的环境变量

打开菜单里的视图->其他窗口里面的属性管理器，依次展开MTCNN、Debug\x64子节点，然后在Microsoft.Cpp.x64.user项上右键，选择属性窗口，找到VC++目录，包含目录，将以下路径添加到包含目录项里

C:\Users\lenovo\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\include
D:\CNN\caffe\include
D:\CNN\caffe\build
D:\CNN\caffe\build\include

其中lenovo是我的电脑用户名，请换成你自己的名，D:\CNN\caffe是我本机caffe包所在路径

将以下路径加入到库路径：

C:\Users\lenovo\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\lib
C:\Users\lenovo\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib
D:\CNN\caffe\build\lib

拷贝以下文件夹下的所有dll文件至系统路径文件夹下(比如C:\Windows\Systems32)
D:\CNN\caffe\build\install\bin

5.编译运行

程序默认会读取imgs文件下的文件，把检测结果输出到results文件夹下，如果想测试摄像头的效果，在main.cpp的main函数里将testcamera();解注释即可

### mxnet版

编译mxnet的windows版，参考[mxnet VS2015编译
](https://github.com/imistyrain/mxnet-oneclick/blob/master/mxnet%20VS2015%E7%BC%96%E8%AF%91.pdf)，然后打开MTCNN.sln,把MTCNNPy设为启动项.加载此工程需要安装VS python的插件[PTVS 2.2.6 VS 2015](https://github.com/Microsoft/PTVS/releases/v2.2.6)

本机测试环境为VS2015,Cuda8.0,CuDNN5.1,python2.7

## 参考

*  [Win10+VS2015 caffe环境搭建](http://blog.csdn.net/akashaicrecorder/article/details/71016942)

* [fddb-windows](https://github.com/imistyrain/fddb-windows)