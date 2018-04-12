![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


## Train VOC dataset with YOLOv3.

### Make the directory to hold all the training data.

```
mkdir -p voc_training/backup
cd voc_training
```

### Download and extract the dataset

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

### Generate the YOLO consumable labels

```
python ../scripts/voc_label.py
```

### Download the pre-trained conv weights

```
wget https://pjreddie.com/media/files/darknet53.conv.74
```


### Update the configuration files

```
cd ..
cp cfg/yolov3-voc.cfg cfg/yolov3-voc-test.cfg
```

Make sure the bath size and subdivisions are set properly in `cfg/yolov3-voc.cfg`

```
Training
batch=64
subdivisions=16
```

Make sure the path are correctly set in the `cfg/voc.data`

```
classes= 20
train  = voc_training/train.txt
valid  = voc_training/2007_test.txt
names = data/voc.names
backup = voc_training/backup
```

### Train the model

```
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg voc_training/darknet53.conv.74
```
