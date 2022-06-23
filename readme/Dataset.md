# IOD-Video Dataset

## 1.Dataset download
You can preview the IOD-Video Dataset with [NJU Box](https://box.nju.edu.cn/d/654f74926166482fa569/), please contact Kailai Zhou (**calayzhou@smail.nju.edu.cn** or **DG21230090@smail.nju.edu.cn**) for the IOD-Video dataset. We will email back
to you as soon as possible (no more than 24 hours for most cases). The download link will include NJU Box, Baidu Cloud and Google Drive.

For better academic communication, your email format is encouraged to contain the following information:

```bash
Name: (who you are.)
Affiliation: (the name of your institution or university.)
Job Title: ( B.S., M, Ph.D, Professor, etc.)
How to use: (Only for non-commercial use.)
```

## 2.Dataset information
```bash
IOD-Video
|------Annotation
|         |------PKL_Annotations.zip
|         |------XML_Annotations.zip
|------IOD-Video_mp4
|         |------001_wild_static_vague.mp4
|         |------002_experiment_static_clear.mp4
|         |------...
|------IOD-Video_withbox_mp4
|         |------001_wild_static_vague.mp4
|         |------002_experiment_static_clear.mp4
|         |------...
|------Split_testlist
|         |------Split1_testlist.txt
|         |------Split2_testlist.txt
|         |------Split3_testlist.txt
|------Frames.zip
|------IOD-Video_avi.zip
|------Dataset.md
```
2.1 Annotation

The annotation includes PKL and XML format. The PKL format is used for training, the XML format shows the
frame-level boxes of each video samples, which are easy to view.

2.2 IOD-Video_mp4

The IOD-Video_mp4 folder contains 600 video samples to have a quick look.
The video samples are named by 
```
[number]_[scene]_[dynamic/static]_[clear/vague].mp4
```
2.3 IOD-Video_withbox_mp4

The IOD-Video_withbox_mp4 folder contains 600 video samples with boxes to have a quick look.

2.4 Split_testlist

Since IOD-Video dataset is randomly divided into three split (train/test at a ratio of 2:1), and the final 
results are averaged over three splits. The test samples of split 1,2,3 can be found in Split1_testlist.txt, Split2_testlist.txt and Split3_testlist.txt. 

2.5 Frames.zip and IOD-Video_avi.zip

The Frames.zip contains the frame level images which are used for training. IOD-Video_avi.zip contains the original avi format video samples.

## 3.Dataset Usage
Download `Frames.zip` and `PKL_Annotations.zip` to the `data` folder as
```bash
IOD-Video
|------data
|         |------TLGDM
|                   |------TrueLeakedGas.pkl
|                   |------TrueLeakedGas_ACT1.pkl
|                   |------TrueLeakedGas_c1_290.pkl
|                   |------TrueLeakedGas_v1_310.pkl
|                   |------Frames
|                              |------TrueLeakedGas
|                                         |------001_wild_dynamic_vague
|                                         |------...
```


## 4.Term of Use and License

This IOD-Video Dataset and Code are made freely available for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation, which are licensed under the CC BY-NC 4.0 [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

