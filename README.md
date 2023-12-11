# HopeNet (in MLX)

**Hopenet** is an accurate and easy to use head pose estimation network made by Nataniel Ruiz, which applies ResNet-50 as the backbone with multi-bin classification and regression heads. The original paper is available in [CVPR workshop from the author](https://arxiv.org/abs/1710.00925).

After Apple published MLX libraries to be available on GitHub's repo, I tried applying this library with rewriting and modifying the code to predict head pose with almost similar to the original HopeNet work except for BatchNormalization which I replace to be GroupNorm due to no BatchNormalization available in MLX. Moreover, I changed the data augmentation proceses to be the two-step augmentation similar to [my GitHub gist](https://gist.github.com/nickuntitled/2e4bb2c57633a9a3ca8bdb1450cf72d6).

The initial head pose result trained by 300W_LP with 5 epochs (just an example), and with the resolution 128x128 pixels is shown the below table. The reason of using this resolution is the limitation on my Mac Mini with 8GB RAM cannot run the model fast enough, and does not have capacity to run this resolution with a large batch size (like 32).

| Technique   | Yaw   | Pitch | Roll  | Mean  | Pretrained |
|-------------|-------|-------|-------|-------|------------|
| My training | 3.684 | 5.754 | 4.550 | 4.663 | [Google Drive](https://drive.google.com/file/d/1_bIg1TobBM8m_DoyDO0MLGBDHHp4dsNr/view?usp=share_link) |

## Dataset

The dataset used for training is the same as the original HopeNet work, which is 300W_LP. You can download the dataset from [the author](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), and the filelist from [Google Drive](https://drive.google.com/file/d/137ZW1213DcNmaXjY7FXQxgWOLxLuYLTA/view?usp=sharing).

For evaluation, the dataset is AFLW2000. You can download the dataset from [the author](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), and the filelist from [Google Drive](https://drive.google.com/file/d/1G9k5WF1yX5pM_GOuumSsQXK9H0fr7ulP/view?usp=sharing).

After downloading both dataset, it is better to create the folder named datasets, and places 300W_LP and AFLW2000 inside. The folder structure is like below.

```
datasets
- 300W_LP
-- AFW, HELEN, LFPW, IBUG
-- AFLW_Flip, HELEN_Flip, LFPW_Flip, IBUG_Flip
-- 300wlp_files.txt
- AFLW2000
-- <image files>
-- aflw2000_list.txt
```

## Pretrained Dataset

You have to download the pretrained ImageNet model by using the wget command like below.

```
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

After finished downloading, you have to place the model inside the weights folder. And run the convert.py.

```
python convert.py --weight <The download weight path>
```

## Training

You can train the dataset by running the command below (you may run 25 epochs with the resolution 224x224 pixels to be similar to the author).

```
python train.py --dataset 300W_LP --data_dir datasets/300W_LP --filename_list datasets/300wlp_files.txt --num_epochs 25 --batch_size 32 --input_size 224 --output_string resnet50 --lr 0.00001
```

## Evaluation

You can evaluate the trained model by running the command below.

```
python test.py --dataset AFLW2000 --data_dir datasets/AFLW2000 --filename_list datasets/aflw2000_list.txt --snapshot < path of the saved snapshots > --input_size 224
```

## Reference

You may refer to [Ruiz's original work on HopeNet](https://github.com/natanielruiz/deep-head-pose).

```
@InProceedings{Ruiz_2018_CVPR_Workshops,
author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
title = {Fine-Grained Head Pose Estimation Without Keypoints},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}
```