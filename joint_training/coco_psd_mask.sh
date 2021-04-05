# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 22308 tools/test_for_coco.py --cfg /home/xinggang/hb/HRNet-Semantic-Segmentation/experiments/CD/cd_test.yaml
CUDA_VISIBLE_DEVICES=0,1 python tools/test_for_coco.py --cfg ./experiments/CD/cd_test.yaml \
--coco_val_json ./data/coco/annotations/instances_train2017.json \
--coco_val_image_folder ./data/coco/train2017/ \
--fcn_ratio 0.6 \
--model_pth "" \
--mode psd \
# --coco_val_json /data/xinggang/coco/coco/annotations/instances_train2017.json \
# --coco_val_image_folder /data/xinggang/coco/coco/train2017/
# --coco_val_json ./data/coco/annotations/instances_val2017.json \
# --coco_val_image_folder ./data/coco/val2017/ \
# --model_pth "/home/hubin/vivo_1/output/Cdrop0.75D4701_ratio0.6875/cd_trans_fcn/checkpoint_epoch_55.pth.tar"
