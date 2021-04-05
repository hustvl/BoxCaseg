CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 18001 tools/pd_train.py --cfg ./experiments/PD/pd_trans_fcn.yaml \
--voc_json_file ./data/VOCSBD/voc_2012_train_aug_cocostyle.json \
--voc_image_folder ./data/VOCSBD/VOC2012/JPEGImages/ \
--dut_image_folder ./data/DUTS/DUTS-TR-Image-single/ \
--dut_label_folder ./data/DUTS/DUTS-TR-Mask-single/ \
--fcn_ratio 0.7 \

