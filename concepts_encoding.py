import os
import re
import json
import clip
import torch
import argparse
import itertools
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backbone', type=str, default='RN50')
    parser.add_argument('--input_path', type=str, default='data/coco/cat2concepts.json')
    parser.add_argument('--output_dir', type=str, default='data/coco')

    args = parser.parse_args()
    print(args)

    # Load unique concepts
    with open('data/coco/cat2concepts.json', 'r') as fp:
        cat2concepts = json.load(fp=fp)
    all_concepts_set = set(itertools.chain.from_iterable(cat2concepts.values()))
    all_concepts_list = sorted(list(all_concepts_set))

    # Maps concept idx to relevant category ids
    concept2cats = []
    for concept in all_concepts_list:
        cat_ids = tuple(k for k, v in cat2concepts.items() if concept in v)
        concept2cats.append(cat_ids)

    # Encode concepts using clip
    clip_model, _ = clip.load(args.backbone, device=args.device)

    concept_emb_list = []
    for i in tqdm(range(0, len(all_concepts_list), args.batch_size)):
        batch = all_concepts_list[i:i+args.batch_size]
        batch_emb = clip_model.encode_text(clip.tokenize(batch))
        concept_emb_list.append(batch_emb)
    concept_emb = torch.cat(concept_emb_list)

    # Save to file
    torch.save(concept_emb, os.path.join(args.output_dir, f'concept_emb_{args.backbone}.pth'))
    with open(os.path.join(args.output_dir, f'unique_concept2cat.pth'), 'r') as fp:
        json.dump({'concepts': all_concepts_list, 'cat_ids': concept2cats})
