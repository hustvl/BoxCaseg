import json
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def cal_iou_map(bboxes1, bboxes2):
    matrices = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            matrices[i, j] = cal_iou(bbox1, bbox2)

    return matrices

def cal_iou(bbox1, bbox2):
    l = max(bbox1[0], bbox2[0])
    u = max(bbox1[1], bbox2[1])
    r = min(bbox1[2], bbox2[2])
    d = min(bbox1[3], bbox2[3])
    w = max(r-l, 0)
    h = max(d-u, 0)
    inter = w*h
    uni = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1]) - inter
    return inter/(uni + 1e-7)

json_root = "/data/xinggang/coco/coco/annotations/instances_train2017.json"
with open(json_root, 'r') as fo:
    json_file = json.load(fo)
print('Successfully load {}!'.format(json_root))

image_infos = json_file['images']
anno_infos = json_file['annotations']

overlap_map = np.zeros((90, 90))
large_overlap_anno_id_ls = []
large_overlap_idx_ls = []

for t, image_info in tqdm(enumerate(image_infos)):
    image_id = int(image_info['file_name'].split('.')[0].strip('0'))
    bbox_per_image = np.zeros((0,4))
    cls_ls = []
    anno_id_ls = []
    idx_ls = []
    for idx, anno_info in enumerate(anno_infos):
        if anno_info['image_id'] == image_id:
            # bbox
            bbox = np.array(anno_info['bbox']).astype(np.int)
            bbox[2:4] = bbox[0:2] + bbox[2:4]
            bbox_per_image = np.concatenate((bbox_per_image,bbox.reshape(1,4)), axis=0)
            # class
            cls = anno_info['category_id'] - 1  # 0 - 89
            cls_ls.append(cls)
            # anno_id
            anno_id = anno_info['id']
            anno_id_ls.append(anno_id)
            idx_ls.append(idx)


    iou_map = cal_iou_map(bbox_per_image, bbox_per_image)
    for i in range(iou_map.shape[0]):
        iou_map[i, i] = 0.0
    indexs = np.where(iou_map>0.5)

    # print(image_id)
    # print(cls_ls, anno_id_ls)
    # print(iou_map)
    large_overlap_anno_id_ls_tmp = []
    large_overlap_idx_ls_tmp = []
    for i, j in zip(indexs[0], indexs[1]):
        cls1, cls2 = cls_ls[i], cls_ls[j]
        overlap_map[cls1, cls2] += 1
        large_overlap_anno_id_ls_tmp.append(anno_id_ls[i])
        large_overlap_anno_id_ls_tmp.append(anno_id_ls[j])
        large_overlap_idx_ls_tmp.append(idx_ls[i])
        large_overlap_idx_ls_tmp.append(idx_ls[j])
    large_overlap_anno_id_ls_tmp = list(set(large_overlap_anno_id_ls_tmp))
    large_overlap_anno_id_ls.extend(large_overlap_anno_id_ls_tmp)
    large_overlap_idx_ls_tmp = list(set(large_overlap_idx_ls_tmp))
    large_overlap_idx_ls.extend(large_overlap_idx_ls_tmp)

    # if t==100:
    #     break


    # print(len(large_overlap_anno_id_ls))

    # import pdb; pdb.set_trace()
    # break

print(len(large_overlap_anno_id_ls))
with open('overlap_box_id.json', 'w') as fo:
    json.dump(large_overlap_anno_id_ls, fo)
with open('overlap_anno_id.json', 'w') as fo:
    json.dump(large_overlap_idx_ls, fo)

for i in range(overlap_map.shape[0]):
    overlap_map[i, i] /= 2
with open('overlap_map.json', 'w') as fo:
    json.dump(overlap_map.tolist(), fo)
overlap_map = overlap_map/(np.sum(overlap_map, axis=1, keepdims=True) + 1e-10)

heatmap = sns.heatmap(overlap_map, cmap='YlGnBu')#, annot=True)
plt.show()
plt.savefig('1.jpg')