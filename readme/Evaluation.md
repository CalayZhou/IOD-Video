# Evaluation

## 1.Results and Models
Methods | Split |AP@0.5 | AP@0.75|  AP@clear|AP@vague|  0.5:0.95 |Download
---|--- |---| --- |--- |--- |---  |---
TEA |1 |  41.97% | 8.10% |  20.86% | 11.21% | 15.43% |[TEA_MOV_K8S1_model_last.pth](https://drive.google.com/file/d/1dvJZ6IQ8P8wyjf4QnVinbr9R7gT-LRe8/view?usp=sharing)
TEA |2 |  41.36% | 7.80% | 22.86% | 9.12% | 14.96% | [TEA_MOV_K8S2_model_last.pth](https://drive.google.com/file/d/1gEkKrCskuBozy02cNVwu1uo8h6Po9zmM/view?usp=sharing)
TEA |3 |  43.24% | 10.08% |  25.18% | 9.51% |16.68% | [TEA_MOV_K8S3_model_last.pth](https://drive.google.com/file/d/1L-faMe_huC-NrdKblA7IbchuFJiEQi-7/view?usp=sharing)
TEA |average |  42.19% | 8.66% |  22.97% | 9.95% | 15.69% |
TEA+STloss| 1 | 45.35% | 8.73% | 22.70% | 12.02% | 16.60%| [TEA_STA_K8S1_model_last.pth](https://drive.google.com/file/d/1NECG8ML63tPEaH8D1U8diuyMSM6yncIc/view?usp=sharing)
TEA+STloss| 2 | 41.33% | 7.40% | 23.27% | 9.01% | 14.94%| [TEA_STA_K8S2_model_last.pth](https://drive.google.com/file/d/1CriQ-bQNucpwCFnPlExqNrN_iiQe4hUI/view?usp=sharing)
TEA+STloss| 3 | 48.90% | 11.93% | 27.47% | 11.75% | 19.21%| [TEA_STA_K8S3_model_last.pth](https://drive.google.com/file/d/1MZBYKeoOr6OCAJkLNyRpcuqWvx_PdOyP/view?usp=sharing)
TEA+STloss |average |  45.19% | 9.35% | 24.48% | 10.93% | 16.92% |

*Model name:  `methods\_(loss)\_K?_S?\_model_last.pth`*

We set opt.offset_h_ratio and opt.offset_w_ratio to 1 for stable convergence when K is small, so TEA+STAloss is slightly different from the original paper.
All these models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1n1VG_nWj5e57iKeJlVOJgs74lQe7q7hn?usp=sharing),
[Baidu Cloud,(code:`buac`)](https://pan.baidu.com/s/1ddV-u5RXnAsKtM8f19W7DA ) and [NJU Box](https://box.nju.edu.cn/d/7d89bd4796ab486b9886/).
The final AP is averaged over three splits.


## 2.Inference K=8 with TEA+STAloss and TEA
Please train or download the above model to the ${PATH_TO_SAVE_MODEL}, for example, download [TEA_STA_K8S1_model_last.pth](https://drive.google.com/file/d/1dvJZ6IQ8P8wyjf4QnVinbr9R7gT-LRe8/view?usp=sharing) to `../experiment/result_model/TEA_STA_K8S2/TEA_STA_K8S2_model_last.pth`
and run

**TEA+STAloss**:
~~~bash
python3 det.py --task normal --K 8  --gpus 0,1  --batch_size 20 --master_batch 10  --num_workers 2 --rgb_model ../experiment/result_model/TEA_STA_K8S1/TEA_STA_K8S1_model_last.pth  --inference_dir ../result/inference_TLGDM_pkl1   --dataset IODVideo   --split  1  --arch TEAresnet_50
~~~
**TEA**:
~~~bash
python3 det.py --task normal --K 8  --gpus 0,1  --batch_size 20 --master_batch 10  --num_workers 2 --rgb_model ../experiment/result_model/TEA_MOV_K8S1/TEA_MOV_K8S1_model_last.pth  --inference_dir ../result/inference_TLGDM_pkl1   --dataset IODVideo   --split  1  --arch TEAresnet_50 --loss_option MOV 
# ==============Args==============
# 
# --task           "normal" by default
# --K              input frame numbers, 8 by default
# --gpus           gpu list, in our experiments, we use 2 NVIDIA GTX 3090
# --batch_size     total batch size 
# --master_batch   batch size in the first gpu
# --num_workers    total workers
# --rgb_model      ${PATH_TO_SAVE_MODEL}
# --inference_dir  "../result/inference_TLGDM_pkl1" by default, path to save inference results, will be used in mAP step
# --dataset        "IODVideo" by default   
# --split 1        1 or 2 or 3; the final results are averaged on three splits
# --arch           resnet_50, I3Dresnet_50, S3Dresnet_50, MSresnet_50, TDNresnet_50, TINresnet_50, TAMresnet_50, TSMresnet_50, TEAresnet_50
# --loss_option    MOV or STAloss； STAloss by default, MOV is original loss
~~~

## 3.Evaluate mAP

The evaluation code is followed from [ACT](https://github.com/vkalogeiton/caffe/tree/act-detector).
<br/>
The evalution time will depend on CPU and take a long time, so `--pkl_ACT` is added to train and ACT at the same time.

1. For frame mAP @0.5, please run:
```bash
python3 ACT.py --pkl_ACT 1 --task frameAP --K 8   --th 0.5 --inference_dir ../result/inference_TLGDM_pkl1 --dataset IODVideo --split 1
```

2. For all frame mAP (@0.5, @0.75, @vague, @clear, 0.5:0.95), you can run:
```bash
bash ACT_total1.sh 8 1
# ==============ACT_total1.sh==============
# 8: K
# 1: split
```

The `--inference_dir` of ACT_total1.sh is set as `../result/inference_TLGDM_pkl1` by default.
The ACT process will read the `TrueLeakedGas_v1_310.pkl`, `TrueLeakedGas_c1_290.pkl` and `TrueLeakedGas_ACT1.pkl`, it will not conflict with the training process.


