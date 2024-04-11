## Total3DUnderstanding [[Project Page]](https://yinyunie.github.io/Total3D/)[[Oral Paper]](https://arxiv.org/abs/2002.12212)[[Talk]](https://www.youtube.com/watch?v=tq7jBhfdszI)

**Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image** <br>
Yinyu Nie, Xiaoguang Han, Shihui Guo, Yujian Zheng, Jian Chang, Jian Jun Zhang <br>
In [CVPR, 2020](http://cvpr2020.thecvf.com/).

<img src="demo/inputs/1/img.jpg" alt="img.jpg" width="20%" /> <img src="demo/outputs/1/3dbbox.png" alt="3dbbox.png" width="20%" /> <img src="demo/outputs/1/recon.png" alt="recon.png" width="20%" /> <br>
<img src="demo/inputs/2/img.jpg" alt="img.jpg" width="20%" /> <img src="demo/outputs/2/3dbbox.png" alt="3dbbox.png" width="20%" /> <img src="demo/outputs/2/recon.png" alt="recon.png" width="20%" />

---

### Install
This implementation uses Python 3.6, [Pytorch1.1.0](http://pytorch.org/), cudatoolkit 9.0. We recommend to use [conda](https://docs.conda.io/en/latest/miniconda.html) to deploy the environment.

* Install with conda:
```
conda env create -f environment.yml
conda activate Total3D
```

* Install with pip:
```
pip install -r requirements.txt
```

---

### Demo
The pretrained model can be download [here](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/EfU0ElYyJ9JFjLL8qolLLLkBNXx6DxkyA4lVAhuMfmSnIw?e=dSScnc). We also provide the pretrained Mesh Generation Net [here](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/EeC4Fqo_W2xJiG2-S6y_7F8BDtJ65eGGNprMvfq6nikRGw?e=qzTyfX). Put the pretrained models under
```
out/pretrained_models
```

A demo is illustrated below to see how the method works. [vtk](https://vtk.org/) is used here to visualize the 3D scenes. The outputs will be saved under 'demo/outputs'. You can also play with your toy with this script.
```
cd Total3DUnderstanding
python main.py configs/total3d.yaml --mode demo --demo_path demo/inputs/1
```

---
### Data preparation
In our paper, we use [SUN-RGBD](https://rgbd.cs.princeton.edu/) to train our Layout Estimation Net (LEN) and Object Detection Net (ODN), and use [Pix3D](http://pix3d.csail.mit.edu/) to train our Mesh Generation Net (MGN).

##### Preprocess SUN-RGBD data

You can either directly download the processed training/testing data [[link](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/EdcyVWe5CLNKlsCnSYbkIy8BfMYmxL9iKqJHWAt0DkF1yw?e=4Eat3A)] to (recommended)
```
data/sunrgbd/sunrgbd_train_test_data
```

or <br>
<br>
1. Download the raw [SUN-RGBD data](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip) to
```
data/sunrgbd/Dataset/SUNRGBD
```
2. Download the 37 class labels of objects in SUN RGB-D images [[link](https://github.com/ankurhanda/sunrgbd-meta-data/blob/master/sunrgbd_train_test_labels.tar.gz)] to 
```
data/sunrgbd/Dataset/SUNRGBD/train_test_labels
```
3. Follow [this work](https://github.com/thusiyuan/cooperative_scene_parsing) to download the preprocessed clean data of SUN RGB-D [[link](https://drive.google.com/open?id=1XeCE87yACXxGisMTPPFb41u_AmQHetBE)] to
```
'data/sunrgbd/Dataset/data_clean'
```
4. Follow [this work](https://github.com/thusiyuan/cooperative_scene_parsing) to download the preprocessed ground-truth of SUN RGB-D [[link](https://drive.google.com/open?id=1QUbq7fRtJtBPkSJbIsZOTwYR5MwtZuiV)], and put the '3dlayout' and 'updated_rtilt' folders respectively to
```
data/sunrgbd/Dataset/3dlayout
data/sunrgbd/Dataset/updated_rtilt
```
5. Run below to generate training and testing data in 'data/sunrgbd/sunrgbd_train_test_data'.
```
python utils/generate_data.py
```
&nbsp;&nbsp; If everything goes smooth, a ground-truth scene will be visualized like

<img src="demo/gt_scene.png" alt="gt_scene.png" width="40%" align="center" />


##### Preprocess Pix3D data
You can either directly download the preprocessed ground-truth data [[link](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/ESM1-BvjC-5Np1xxj8QUNQ8Bv58zJyGqWHuRwgm5Pnf1AA?e=RYzy8L)] to (recommended)
```
data/pix3d/train_test_data
```
Each sample contains the object class, 3D points (sampled on meshes), sample id and object image (w.o. mask). Samples in the training set are flipped for augmentation.

or <br>
<br>

1. Download the [Pix3D dataset](http://pix3d.csail.mit.edu/) to 
```
data/pix3d/metadata
```
2. Run below to generate the train/test data into 'data/pix3d/train_test_data'
```
python utils/preprocess_pix3d.py
```

---
### Training and Testing
We use the configuration file (see 'configs/****.yaml') to fully control the training/testing process. There are three subtasks in Total3D (layout estimation, object detection and mesh reconstruction). We first pretrain each task individually followed with joint training.


##### Pretraining
1. Switch the keyword in 'configs/total3d.yaml' between ('layout_estimation', 'object_detection') as below to pretrain the two tasks individually.
```
train:
  phase: 'layout_estimation' # or 'object_detection'

python main.py configs/total3d.yaml --mode train
```
The two pretrained models can be correspondingly found at 
```
out/total3d/a_folder_named_with_script_time/model_best.pth
```

2. Train the Mesh Generation Net by:
```
python main.py configs/mgnet.yaml --mode train
```
The pretrained model can be found at
```
out/mesh_gen/a_folder_named_with_script_time/model_best.pth
```

##### Joint training

List the addresses of the three pretrained models in 'configs/total3d.yaml', and modify the phase name to 'joint' as
```
weight: ['folder_to_layout_estimation/model_best.pth', 'folder_to_object_detection/model_best.pth', 'folder_to_mesh_recon/model_best.pth']

train:
  phase: 'joint'
```
Then run below for joint training.
```
python main.py configs/total3d.yaml --mode train
```
The trained model can be found at
```
out/total3d/a_folder_named_with_script_time/model_best.pth
```

##### Testing
Please make sure the weight path is renewed as 
```
weight: ['folder_to_fully_trained_model/model_best.pth']
```
and run
```
python main.py configs/total3d.yaml --mode test
```

This script generates all 3D scenes on the test set of SUN-RGBD under
```
out/total3d/a_folder_named_with_script_time/visualization
```

You can also visualize a 3D scene given the sample id as
```
python utils/visualize.py --result_path out/total3d/a_folder_named_with_script_time/visualization --sequence_id 274
```

##### Differences to the paper
1. We retrained the model with the learning rate decreases to half if there is no gain within five steps, which is much more efficient.
2. We do not provide the Faster RCNN code. Users can train their 2D detector with [[link](https://github.com/facebookresearch/maskrcnn-benchmark)].

---

### Citation
If you find our work is helpful, please cite
```
@InProceedings{Nie_2020_CVPR,
author = {Nie, Yinyu and Han, Xiaoguang and Guo, Shihui and Zheng, Yujian and Chang, Jian and Zhang, Jian Jun},
title = {Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes From a Single Image},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
Our method partially follows the data processing steps in [this work](https://github.com/thusiyuan/cooperative_scene_parsing). If it is also helpful to you, please cite
```
@inproceedings{huang2018cooperative,
  title={Cooperative Holistic Scene Understanding: Unifying 3D Object, Layout, and Camera Pose Estimation},
  author={Huang, Siyuan and Qi, Siyuan and Xiao, Yinxue and Zhu, Yixin and Wu, Ying Nian and Zhu, Song-Chun},
  booktitle={Advances in Neural Information Processing Systems},
  pages={206--217},
  year={2018}
}	
```



