# SSDT：A single-shot detector for PCB  defects
 pytorch, SSDT, PCB defects detection

## Prerequisites
1. Python 3.6
2. Numpy
3. Pytorch 0.3
4. torchvision
5. visdom
6. tqdm

## Train dataset
support VOC, COCO and [PCB dataset](http://robotics.pkusz.edu.cn/resources/dataset/) **(proposed by PKU-HRI)**
## Train the model
> python train_test.py

## Use the visdom
> python -m visdom.server -port ****

## Result examples
![](examples/result1.png)
![](examples/result4.png)