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
wget https://pjreddie.com/media/files/yolov2.weights
../darknet partial ../cfg/yolov2-voc.cfg yolov2.weights yolov2.conv.20 20
```

### Generate the anchors before training

```
python ../scripts/gen_anchors.py -filelist train.txt -output_dir generated_anchors/voc -num_clusters 5
```

### Update the configuration files

```
cd ..
cp cfg/yolov2-voc.cfg cfg/yolov2-voc-test.cfg
```

Make sure the bath size and subdivisions are set properly in `cfg/yolov2-voc.cfg`

```
Training
batch=64
subdivisions=8
```

Copy the anchors generated in the last steps and replace the anchors in `cfg/yolov2-voc.cfg` and `cfg/yolov2-voc-test.cfg`

```
anchors =  copy from "generated_anchors/voc/anchors5.txt"
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
mkdir voc_training/log
./darknet detector train cfg/voc.data cfg/yolov2-voc.cfg voc_training/yolov2.conv.20 >> voc_training/log/voc.log
```
