import json
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import cv2
import pycocotools.mask as mask_util
import argparse

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    inters = iw*ih

    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    iou = inters / uni
    return iou

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get Proxy Mask")
    parser.add_argument("--gt-path", type=str, default='/data/coco/annotations/instances_train2017.json',
                        help="Path to the json file of coco.")
    parser.add_argument("--seg-pred", type=str, default='./seg_result_psd.json',
                        help="Path to the proxy mask seg_result.json.")
    parser.add_argument("--index-path", type=str, default='./indexes.json',
                        help="Indexes json.")
    return parser.parse_args()
args = get_arguments()

with open(args.gt_path, 'r') as fo:
    json_dict = json.load(fo)
with open(args.seg_pred, 'r') as fo:
    seg_json_dict = json.load(fo)
with open(args.index_path, 'r') as fo:
    indexes = json.load(fo)

coco = COCO(args.gt_path)
annotations = []
for i in range(len(json_dict['images'])):
    if i%1000 ==0:
        print(i," processed...")
    image_id = json_dict['images'][i]['id']
    img_name = json_dict['images'][i]['file_name']
    # img = np.array(Image.open("/data/xinggang/coco/coco/train2017/"+img_name))
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    num_anns = len(anns)
    areas = []
    for ann_i in range(num_anns):
        bbox = anns[ann_i]['bbox']
        areas.append(bbox[2]*bbox[3])
    height = json_dict['images'][i]['height']
    width = json_dict['images'][i]['width']
    mask = np.zeros((height,width))
    area_sort_ids = np.argsort(areas)[::-1]
    for area_sort_id in area_sort_ids:
        ann = anns[area_sort_id]
        # ann_id = (json_dict['annotations']).index(ann)
        ann_id = indexes[str(ann['id'])]
        mask = np.where(mask_util.decode(seg_json_dict[ann_id]['segmentation'])==1,area_sort_id+1,mask)
    for ann_i in range(num_anns):
        bbox = anns[ann_i]['bbox']
        binary_mask = np.where(mask==(ann_i+1),1,0).astype(np.uint8)
        x,y = np.where(binary_mask)
        if binary_mask.sum() !=0:
            y_min, y_max, x_min, x_max = x.min(), x.max(), y.min(), y.max()
        else:
            y_min = 0 
            y_max = 0
            x_min = 0
            x_max = 0
        # img_bbox = cv2.rectangle(img.copy(), (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
        # img_bbox[:,:,2] = np.where(binary_mask>0,255,img_bbox[:,:,2])
        # cv2.imwrite("./train_proxygt_visual/"+str(i)+"_"+str(ann_i)+".png",img_bbox[:,:,[2,1,0]])
        mask_box = [x_min, y_min, x_max, y_max]
        gt_box = [int(bbox[0]), int(bbox[1]), int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])]
        box_iou = get_iou(mask_box, gt_box)
        encode_mask = mask_util.encode(np.array(binary_mask[:, :, np.newaxis], order="F"))[0]
        anns[ann_i]['segmentation'] = encode_mask
        anns[ann_i]['segmentation']['counts'] = bytes.decode(encode_mask['counts'])#.encode("utf-8")
        anns[ann_i]['confidence'] = box_iou
        annotations.append(anns[ann_i])
json_dict['annotations'] = annotations
with open('coco_train2017_proxymask.json', 'w') as f:
    json.dump(json_dict, f)
