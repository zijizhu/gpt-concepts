import os
import re
import json
import random
import argparse
from pycocotools.coco import COCO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='data/coco/cat2img_samples.json')
    parser.add_argument('--coco_train_dir', type=str, default='data/coco/annotations/instances_train2017.json')

    parser.add_argument('--exclude', nargs='+', type=int, default=[1], help='Category ids to be excluded.')
    parser.add_argument('--area_thresh', type=float, default=5000, help='Include objects with area larger than this threshold.')
    parser.add_argument('--num_per_category', type=int, default=10, help='Number of images to sample per class.')

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    coco_train = COCO(args.coco_train_dir)
    cat2img_samples = {cat_id: [] for cat_id in coco_train.cats}

    exclude_ids = args.exclude
    for cat_id in exclude_ids:
        del cat2img_samples[cat_id] # Delete 'person' class

    for cat_id, img_ids in cat2img_samples.items():
        img_ids_with_cat = coco_train.catToImgs[cat_id]
        img_ids_with_cat_filtered = []
        for img_id in img_ids_with_cat:
            img_annotations = coco_train.imgToAnns[img_id]
            # Do not select imgs with 'person' class (GPT4-V does not generate texts for imgs with people)
            if any(ann['category_id'] in exclude_ids for ann in img_annotations):
                continue
            # Do not select imgs with only small objects of target class
            ann_with_cat_id = [ann for ann in img_annotations if ann['category_id'] == cat_id]
            if all(ann['area'] < args.area_thresh for ann in ann_with_cat_id):
                continue
            img_ids_with_cat_filtered.append(img_id)
        sampled_img_ids = random.sample(img_ids_with_cat_filtered, args.num_per_category)
        img_ids += sampled_img_ids
    
    with open(args.output_dir, 'w+') as fp:
        json.dump(cat2img_samples, fp=fp)
