import os
import re
import json
import argparse
import pickle as pkl
from nltk.metrics import edit_distance


def parse_response_to_concepts(message):
    '''Clean LLM response and extract concept entities.'''
    phrases_to_remove = ['some', 'appears to be', 'appears to have']
    drop_responses_with_str = ['sorry', 'does not', "doesn't", 'there is no', 'rather than']
    drop_concepts_with_str = ['size', 'length', '"', '\'']
    drop_concepts = ['small', 'large', 'medium']

    message = message.lower()
    # Drop responses such as "I'm sorry, ..."
    if any(word in message for word in drop_responses_with_str):
        return []
    brackets = r'\(.*?\)' # Remove round brackets
    message = re.sub(brackets, '', message)
    for phrase in phrases_to_remove:    # Remove redundant phrases (e.g. some)
        message = message.replace(phrase, '')
    splitted = re.split(r'\n- |- |; |, | , |;|\n|\\', message)

    concepts = []
    for s in splitted:
        s = s.strip(' ._+!@#$?')
        if not s.isascii():
            continue
        if any(word in s for word in drop_concepts_with_str):
            continue
        if s in drop_concepts:
            continue
        if len(s) == 0:
            continue
        concepts.append(s)
    return concepts


def extract_unique_concepts(concepts: list[str], edit_dist_thresh=3):
    '''Extact unique concept strings by removing similar strings based on edit distance.'''
    unique_concepts = []
    for c in concepts:
        if any(edit_distance(c, uc) < edit_dist_thresh for uc in unique_concepts):
            continue
        unique_concepts.append(c)
    return sorted(unique_concepts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/coco/cat2gpt_responses.pkl')
    parser.add_argument('--output_path', type=str, default='data/coco/cat2concepts.json')
    parser.add_argument('--edit_dist_thresh', type=int, default=3,
                        help='Edit distance threshold to remove similar concept strings.')
    args = parser.parse_args()

    with open(args.input_path, 'rb') as fp:
        cat2gpt_responses = pkl.load(file=fp)
    
    cat2cleaned_concepts = {cat: [] for cat in cat2gpt_responses}
    for cat_id, gpt_responses in cat2gpt_responses.items():
        cat_concepts = []
        for res in gpt_responses:
            cat_concepts += parse_response_to_concepts(res.choices[0].message.content)
        cat_concepts = extract_unique_concepts(cat_concepts, 3)
        cat2cleaned_concepts[cat_id] = cat_concepts
    
    with open(args.output_path, 'w+') as fp:
        json.dump(cat2cleaned_concepts, fp=fp)
