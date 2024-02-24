import os
import json
import argparse
import pickle as pkl
from tqdm import tqdm
from openai import OpenAI
from pycocotools.coco import COCO


def create_gpt4vision_prompt(cat_name, img_url):
  return [
    {
      "role": "user",
      "content": [
        {"type": "text",
         "text": ("List the visual attributes (e.g. features, shape, color) of a vase: "
                  "cylindrical; tall and slim; reflective surface. "
                  f"Now list the visual attributes (e.g. features, shape, color) of the {cat_name} in this image. List the visual attributes without using full sentences:")
         },
        {"type": "image_url", "image_url": { "url": img_url }},
      ],
    }
  ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/coco/cat2img_samples.json')
    parser.add_argument('--output_dir', type=str, default='data/coco/cat2gpt_responses.json')
    parser.add_argument('--coco_train_dir', type=str, default='data/coco/annotations/instances_train2017.json')

    args = parser.parse_args()
    print(args)

    coco_train = COCO(args.coco_dir)
    with open(args.input_dir, 'r') as fp:
        cat2img_ids = json.load(fp=fp)

    client = OpenAI()
    cat2gpt_responses = {cat_id: [] for cat_id in cat2img_ids}

    for cat_id, img_samples in cat2img_ids.items():
        cat_id, img_samples = int(cat_id), [int(img_id) for img_id in img_samples]
        cat_name = coco_train.cats[cat_id]['name']
        print(f'Generating Concepts for category {cat_name}...')
        for img_id in tqdm(img_samples):
            img_url = coco_train.imgs[img_id]['coco_url']
            prompts = create_gpt4vision_prompt(cat_name, img_url)

            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=prompts,
                max_tokens=300,
            )

            cat2gpt_responses[cat_id].append(response)
    
    with open(args.output_dir, 'wb') as fp:
        pkl.dump(cat2gpt_responses, file=fp)