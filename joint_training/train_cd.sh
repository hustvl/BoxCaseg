CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 19445 tools/cd_train.py --cfg ./experiments/CD/cd_trans_fcn.yaml \
--weakly_ratio 0.6875 \
--coco_train_root ./data/coco/train2017/ \
--coco_train_json ./data/coco/annotations/instances_train2017.json \
--coco_overlap_file ./tools/overlap_anno_id.json \
--dut_image_folder ./data/DUTS/DUTS-TR-Image-single/ \
--dut_label_folder ./data/DUTS/DUTS-TR-Mask-single/ \
--fcn_ratio 0.6
# --dut_image_folder ./data/DUTS/DUTS-TR-Image-single/ \
# --dut_label_folder ./data/DUTS/DUTS-TR-Mask-single/ \
# --dut_image_folder ./data/DUTS/DUTS-TR-Image-single-new/ \
# --dut_label_folder ./data/DUTS/DUTS-TR-Mask-single-new/ \