import os
import json
import pickle as pkl
from tqdm import tqdm
from openai import OpenAI
from pycocotools.coco import COCO


if __name__ == '__main__':
    coco = COCO('data/coco/annotations/instances_val2017.json')
    categories = {k: v['name'] for k, v in coco.cats.items()}

    client = OpenAI()
    gpt_responses = {}
    for idx, cat in tqdm(categories.items()):
        other_cats_str = str([c for c in categories.values() if c != cat]).replace("'", '').replace('"', '')
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ("You are a computer scientist who is developing a computer vision system "
                             "that can detect and classify different real world objects.")},
                {"role": "user",
                 "content": (f"List 5 visual attributes belong to a {cat} that can help a system tell a {cat} apart from "
                             f"{other_cats_str}. List visual attributes only:")}
            ]
        )
        print(idx, cat)
        print(completion.choices[0].message.content)
        print()
        gpt_responses[idx] = {'category': cat, 'gpt': completion.choices[0].message.content}
    with open('data/coco/gpt_responses.pkl', 'wb') as fp:
        pkl.dump(gpt_responses, file=fp)
    with open('data/coco/gpt_responses.json', 'w+') as fp:
        json.dump(gpt_responses, fp=fp)
