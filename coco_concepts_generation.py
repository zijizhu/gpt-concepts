import os
import json
import argparse
import pickle as pkl
from tqdm import tqdm
from pycocotools.coco import COCO
from openai import (OpenAI, RateLimitError,
                    InternalServerError, BadRequestError)


def create_gpt4vision_prompt(cat_name, img_url):
  return [
    {
      "role": "user",
      "content": [
        {"type": "text",
         "text": ("List the visual attributes (e.g. features, shape, color) of a vase: "
                  "cylindrical; tall and slim; reflective surface. "
                  f"Now list the visual attributes (e.g. features, shape, color) of the {cat_name} in this image."
                  f"If there exists multiple instances of {cat_name}, list the visual attributes of any of them."
                  "List the visual attributes without using full sentences:")
         },
        {"type": "image_url", "image_url": { "url": img_url }},
      ],
    }
  ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/coco/cat2img_samples.json')
    parser.add_argument('--output_dir', type=str, default='data/coco/cat2gpt_responses.pkl')
    parser.add_argument('--coco_train_dir', type=str, default='data/coco/annotations/instances_train2017.json')

    args = parser.parse_args()
    print(args)

    coco_train = COCO(args.coco_train_dir)
    with open(args.input_dir, 'r') as fp:
        cat2img_ids = json.load(fp=fp)

    client = OpenAI()

    # Resume from saved data
    if os.path.exists(args.output_dir):
        with open(args.output_dir, 'rb') as fp:
            cat2gpt_responses = pkl.load(file=fp)
    else:
        cat2gpt_responses = {int(cat_id): [] for cat_id in cat2img_ids}

    for cat_id, img_samples in cat2img_ids.items():
        cat_id, img_samples = int(cat_id), [int(img_id) for img_id in img_samples]
        cat_name = coco_train.cats[cat_id]['name']

        if len(cat2gpt_responses[cat_id]) != 0:
            print(f'Skip category {cat_name}, gpt responses for this category is already generated!')
            continue

        print(f'Generating Concepts for category {cat_name}...')
        for img_id in tqdm(img_samples):
            img_url = coco_train.imgs[img_id]['coco_url']
            prompts = create_gpt4vision_prompt(cat_name, img_url)
            try:
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=prompts,
                    max_tokens=300,
                )
            except (InternalServerError, BadRequestError) as e:
                tqdm.write(f'GPT4 Vision encountered an error when generating for category {cat_name} in image {img_url}')
                response = None
            except RateLimitError as e:
                print('Rate limit or quota exceeded!')
                exit(1)

            cat2gpt_responses[cat_id].append(response)

        # Save the gpt responses after generation is compeleted for each category
        with open(args.output_dir, 'wb') as fp:
            pkl.dump(cat2gpt_responses, file=fp)
