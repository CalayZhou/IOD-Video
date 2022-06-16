python3 train.py --K 8 --exp_id Train_K8_Imagenet_TLGDM_STA_S1 --rgb_model ../experiment/result_model/TEA_STA_K8S1/ --batch_size 16  --master_batch 8  --lr 5e-4 --gpus 0,1 --num_workers 4  --num_epochs 12 --lr_step 6,8 --dataset IODVideo --split 1  --arch TEAresnet_50   --pretrain_model imagenet
rm -r  ../result/inference_TLGDM_pkl1
python3 det.py --task normal --K 8  --gpus 0,1  --batch_size 20 --master_batch 10  --num_workers 2 --rgb_model ../experiment/result_model/TEA_STA_K8S1/model_last.pth  --inference_dir ../result/inference_TLGDM_pkl1   --dataset IODVideo  --split 1  --arch TEAresnet_50
bash ACT_total1.sh 8  1
