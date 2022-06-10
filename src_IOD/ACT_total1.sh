echo "--K $1  --split $2"
rm -r ../result/inference_TLGDM_pkl1/frame_detections.pkl
echo "ACT 0.5"
python3 ACT.py --pkl_ACT 1 --task frameAP --K $1   --th 0.5 --inference_dir ../result/inference_TLGDM_pkl1 --dataset IODVideo --split $2
echo "ACT 0.75"
python3 ACT.py --pkl_ACT 1 --task frameAP --K $1   --th 0.75 --inference_dir ../result/inference_TLGDM_pkl1 --dataset IODVideo --split $2
echo "ACT all 0.5-0.95"
python3 ACT.py --pkl_ACT 1 --task frameAP_all --K $1  --inference_dir ../result/inference_TLGDM_pkl1 --dataset IODVideo --split $2


echo "backup TrueLeakedGas_ACT1.pkl"
mv ../data/TLGDM/TrueLeakedGas_ACT1.pkl  ../data/TLGDM/TrueLeakedGas_backup1.pkl

echo "AP clear"
rm -r ../result/inference_TLGDM_pkl1/frame_detections.pkl
mv ../data/TLGDM/TrueLeakedGas_c1_290.pkl ../data/TLGDM/TrueLeakedGas_ACT1.pkl
python3 ACT.py --pkl_ACT 1 --task frameAP_all --K $1   --inference_dir ../result/inference_TLGDM_pkl1 --dataset IODVideo --split $2
mv  ../data/TLGDM/TrueLeakedGas_ACT1.pkl  ../data/TLGDM/TrueLeakedGas_c1_290.pkl

echo "AP vague"
rm -r ../result/inference_TLGDM_pkl1/frame_detections.pkl
mv ../data/TLGDM/TrueLeakedGas_v1_310.pkl ../data/TLGDM/TrueLeakedGas_ACT1.pkl
python3 ACT.py --pkl_ACT 1  --task frameAP_all --K $1   --inference_dir ../result/inference_TLGDM_pkl1 --dataset IODVideo --split $2
mv  ../data/TLGDM/TrueLeakedGas_ACT1.pkl  ../data/TLGDM/TrueLeakedGas_v1_310.pkl

echo "restore TrueLeakedGas.pkl"
mv ../data/TLGDM/TrueLeakedGas_backup1.pkl  ../data/TLGDM/TrueLeakedGas_ACT1.pkl
