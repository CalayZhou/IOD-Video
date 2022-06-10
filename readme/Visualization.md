# Visualization
The visualization codes are located in [${IOD_ROOT}/src_IOD/vis](../src_IOD/vis). Following [MOC](https://github.com/MCG-NJU/MOC-Detector), a part of codes from `opts.py` and `datasets/dataset` are separated in `vis` for convenience.

##1.Prepare the demo sample and model
1.1 demo samples

Put the video sample in the path ``./src_IOD/vis/demo_video/xxxx.avi``. The 001_wild_static_vague.avi is provided by default.

1.2 models

Since the 001_wild_static_vague.avi is in the test list of split 3 (Split3_testlist.txt), download [TEA_STA_K8S3_model_last.pth](https://drive.google.com/file/d/1MZBYKeoOr6OCAJkLNyRpcuqWvx_PdOyP/view?usp=sharing) to `./experiment/result_model/TEA_STA_K8S3/TEA_STA_K8S3_model_last.pth`.

##2.Demo 

For short, run this script in `${IOD_ROOT}/src_IOD/vis`

```python
cd src_IOD/vis
python3 vis_det.py  --vname 001_wild_static_vague.avi 
```

The following args can be modified in [tiny_opt.py](../src_IOD/vis/tiny_opt.py):

```python
# important args:
#
# --DATA_ROOT          path to the test video, by default is ${IOD_ROOT}/src_IOD/vis/demo_video
# --inference_dir      path to generate result video, by default is ${MOC_ROOT}/src_IOD/vis/demo_video
# --rgb_model          path to the IOD-Video trained model, by default is ${MOC_ROOT}/experiment/result_model/TEA_STA_K8S3/TEA_STA_K8S3_model_last.pth
#--vname               the video to be processed, by default is 001_wild_static_vague.avi 
#--SimpleFrameProcess  0: simply show the box which has the max score; 1: link the boxes tubelet as  [MOC](https://github.com/MCG-NJU/MOC-Detector) did

# the following args should be same with the training args of model 

#--arch                TEAresnet_50 by default
#--loss_option         STAloss by default
#--K                   8 by default
```

If you set `--SimpleFrameProcess` to 0, it will link the boxes tubelet as  [MOC](https://github.com/MCG-NJU/MOC-Detector) did, you can modify these two thresholds to control visualization performance:
```python
# visualization threshold:
#
# --tube_vis_th      the lowest score for retaining a tubelet, by default is 0.12 (tubelet score)
# --frame_vis_th     the lowest score for retaining a individual frame in the tubelet, by default is 0.015 (frame score)
```

>Do not set a ver large `--tube_vis_th` due to the property of the focal loss, otherwise it will eliminate most of detection tubelets.

>`--frame_vis_th` will eliminate the lower score detection frames from a tubelet. On the one hand it can handle the action boundary but on the other hand, it may lead to the **discontinuity**.
 

 
##3.Demo Results
The demo result can be found in `--inference_dir`(./src_IOD/vis/demo_video/xxxx_simpleframe_result.avi or xxxx_videotubes_result.avi).

If you want to save the demo result as gif, you can try
```python
cd src_IOD/vis
python3 vis_det.py  --vname 001_wild_static_vague.avi  --save_gif
```
The result will be found in /src_IOD/vis/demo_video