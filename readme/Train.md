# Train

## 1.Train K=8 with TEA+STAloss
please run
```bash
python3 train.py --K 8 --exp_id Train_K8_Imagenet_TLGDM_STA_S1 --rgb_model ../experiment/result_model/TEA_STA_K8S1/ --batch_size 16  --master_batch 8  --lr 5e-4 --gpus 0,1 --num_workers 4  --num_epochs 12 --lr_step 6,8 --dataset IODVideo --split 1  --arch TEAresnet_50   --pretrain_model imagenet
# ==============Args==============
#
# --K              input frame numbers, 8 by default
# --exp_id         your experiment ID
# --rgb_model      ${PATH_TO_SAVE_MODEL}
# --batch_size     total batch size 
# --master_batch   batch size in the first gpu
# --lr             initial learning rate, 5e-4 by default (Adam)
# --gpus           gpu list, in our experiments, we use 2 NVIDIA GTX 3090
# --num_workers    total workers
# --num_epoch      max epoch
# --lr_step        epoch for x0.1 learning rate
# --dataset        IODVideo by default 
# --split 1        1 or 2 or 3; the final results are averaged on three splits
# --arch           resnet_50, I3Dresnet_50, S3Dresnet_50, MSresnet_50, TDNresnet_50, TINresnet_50, TAMresnet_50, TSMresnet_50, TEAresnet_50 

# [Optional]
# --loss_option    MOV or STAloss； STAloss by default, MOV is original loss  
# --save_all       save each epoch's training model
# --val_epoch      compute loss on validation set after each epoch
```

If you use `--save_all`, it will save every epoch's training model like `model_[12]_2020-02-12-15-25.pth`. Otherwise it will save last epoch model like `model_last.pth`.

## 2.Recovery from the specific epoch

If the code encounters some running error, we can use following step to recovery training from specific epoch:

```bash
python3 train.py --K 8 --exp_id Train_K8_Imagenet_TLGDM_STA_S1 --rgb_model ../experiment/result_model/TEA_STA_K8S1/ --batch_size 16  --master_batch 8  --lr 5e-4 --gpus 0,1 --num_workers 4  --num_epochs 12 --lr_step 6,8 --dataset IODVideo --split 1  --arch TEAresnet_50 --load_model ? --start_epoch ?
```

<br/>
For example, if we want to recovery from epoch 4, then we add `--load_model model_[4]_2020-01-20-03-25.pth --start_epoch 4`.
<br/>

## 3.Logs
During training,  go to `${PATH_TO_SAVE_MODEL}` and run:

```powershell
tensorboard --logdir=logs_tensorboardX --port=6006
```
The loss curve will show in `localhost:6006`.


## 4.Train with .sh script 

```bash
bash run.sh

# ==============run.sh==============
# python3 train.py --K 8 --exp_id Train_K8_Imagenet_TLGDM_STA_S1 --rgb_model ../experiment/result_model/TEA_STA_K8S1/ --batch_size 16  --master_batch 8  --lr 5e-4 --gpus 0,1 --num_workers 4  --num_epochs 12 --lr_step 6,8 --dataset IODVideo --split 1  --arch TEAresnet_50   --pretrain_model imagenet
# rm -r  ../result/inference_TLGDM_pkl1
# python3 det.py --task normal --K 8  --gpus 0,1  --batch_size 20 --master_batch 10  --num_workers 2 --rgb_model .../experiment/result_model/TEA_STA_K8S1/model_last.pth  --inference_dir ../result/inference_TLGDM_pkl1   --dataset IODVideo   --split  1  --arch TEAresnet_50
# bash ACT_total1.sh 8  1
```
The run.sh integrates `train` and  `evaluation`.