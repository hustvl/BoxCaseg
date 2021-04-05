CUDA_VISIBLE_DEVICES=0,1 python tools/test_for_pascal.py --cfg ./experiments/PD/pd_test.yaml \
--voc_image_folder ./data/VOCSBD/VOC2012/JPEGImages/ \
--voc_val_json_file ./data/VOCSBD/voc_2012_train_aug_cocostyle.json \
--voc_val_mask_root ./data/VOCSBD/VOC2012/SegmentationObject/ \
--fcn_ratio 0.7 \
--mode psd \
--model_pth /workspace/hb/vivo_1/output/PD_new_40ep/pd_trans_fcn_new/checkpoint_epoch35.pth.tar \
# --voc_val_json_file /home/hubin/WSIS/data/VOCSBD/voc_2012_val_cocostyle.json \
# --voc_val_json_file /home/hubin/WSIS/data/VOCSBD/voc_2012_train_aug_cocostyle.json \