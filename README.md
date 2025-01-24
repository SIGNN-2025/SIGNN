# SAM-injected Graph Neural Network for One-Shot Object Detection
![SIGNN](images/SIGNN.png)
## Installation

1. Create a conda virtual environment and activate it

```shell
conda create -n SIGNN python=3.8 -y
conda activate SIGNN
```

2. Install PyTorch and torchvision 

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

3. Install mmcv and geometric

```shell
pip install mmcv-full 1.7.0
pip install torch_geometric 1.7.0
```

4. Install build requirements and then install MMDetection.

```shell
pip install -r requirements/build.txt
pip install -v -e . 
```

## Datasets Preparation

Download coco dataset and voc dataset from the official websites. ```

4. Install build requirements and then install MMDetection.

```shell
pip install -r requirements/build.txt
pip install -v -e . 
```

## Datasets Preparation

Download coco dataset and voc dataset from the official

Download voc_annotation from this [link](https://drive.google.com/drive/folders/1czLhPw65ILmiGU8z95qHGkVTi0EdTGiJ?usp=sharing).

Download ref_ann_file from this [link](https://drive.google.com/drive/folders/1GztcOl8ltCVv9YJdhuvFZq15LTwxWJ7M?usp=sharing).

We expect the directory structure to be the following:
```
SIGNN
├── data
│   ├──coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   ├──VOCdevkit
│   │   ├── voc_annotation
│   │   ├── VOC2007
│   │   ├── VOC2012
├── ref_ann_file
...
```

## Backbone Weight Preparation

Download the ResNet50 model for training from this [link](https://drive.google.com/file/d/1tcRtU-CBu1q00cnnZ6jiF2vvQCzY0a4P/view?usp=sharing).

```
SIGNN
├── resnet_model
│   ├──res50_loadfrom.pth
```

## Inference with a pretrained model
```shell
./tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS} --out ${RESULTS} --eval bbox --average ${EVALUATION_NUMBER}

# e.g.,
# test unseen classes
bash ./tools/dist_test.sh configs/coco/split1/SIGNN.py /work_dirs/split1.pth 2 --out results.pkl --eval bbox --average 5

# test seen classes
bash ./tools/dist_test.sh configs/coco/split1/SIGNN.py /work_dirs/split1.pth 2 --out results.pkl --eval bbox --average 5 --test_seen_classes
```

Note: We haven't released the training code yet, but we will release it after the paper is accepted.
