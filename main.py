import os
import json
import re
import collections
import csv


def find_writing_sessions(dataset_dir):
    return [
        os.path.join(dataset_dir, path)
        for path in os.listdir(dataset_dir)
        if path.endswith('jsonl')
    ]


def read_writing_session(path):
    with open(path, 'r') as f:
        return [json.loads(event) for event in f]


def apply_ops(doc, mask, ops, source):
    original_doc = doc
    original_mask = mask
    new_doc, new_mask = '', ''

    for op in ops:
        if 'retain' in op:
            num_char = op['retain']
            new_doc += original_doc[:num_char]
            new_mask += original_mask[:num_char]
            original_doc = original_doc[num_char:]
            original_mask = original_mask[num_char:]
        elif 'insert' in op:
            insert_doc = op['insert']
            if isinstance(insert_doc, dict):
                continue
            insert_mask = 'A' * len(insert_doc) if source == 'api' else 'U' * len(insert_doc)
            new_doc += insert_doc
            new_mask += insert_mask
        elif 'delete' in op:
            num_char = op['delete']
            if original_doc:
                original_doc = original_doc[num_char:]
                original_mask = original_mask[num_char:]
            else:
                new_doc = new_doc[:-num_char]
                new_mask = new_mask[:-num_char]
    return new_doc + original_doc, new_mask + original_mask


def get_text_and_mask(events, event_id, remove_prompt=True):
    prompt = events[0]['currentDoc'].strip()
    text = prompt
    mask = 'P' * len(prompt)

    for event in events[:event_id]:
        if 'ops' not in event['textDelta']:
            continue
        ops = event['textDelta']['ops']
        source = event['eventSource']
        text, mask = apply_ops(text, mask, ops, source)

    if remove_prompt and 'P' in mask:
        end_index = mask.rindex('P')
        text = text[end_index + 1:]
        mask = mask[end_index + 1:]

    return text, mask


def simple_sent_tokenize(text):
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    return [s.strip() for s in sentence_endings.split(text.strip()) if s.strip()]


def identify_author(mask):
    if 'U' in mask and 'A' in mask:
        return 'user_and_api'
    elif 'U' in mask:
        return 'user'
    elif 'A' in mask:
        return 'api'
    else:
        return 'unknown'


def classify_sentences_by_author(text, mask):
    sentences = []
    pointer = 0
    for sentence_id, sentence in enumerate(simple_sent_tokenize(text)):
        sentence_len = len(sentence)
        while pointer < len(text) and text[pointer].isspace():
            pointer += 1
        if text[pointer:pointer + sentence_len] != sentence:
            continue
        sentence_mask = mask[pointer:pointer + sentence_len]
        author = identify_author(sentence_mask)
        sentences.append({
            'sentence_id': sentence_id,
            'author': author,
            'sentence': sentence
        })
        pointer += sentence_len
    return sentences


def process_file(path, output_dir):
    filename = os.path.basename(path).replace('.jsonl', '')
    try:
        events = read_writing_session(path)
        text, mask = get_text_and_mask(events, len(events), remove_prompt=True)
        sentences = classify_sentences_by_author(text, mask)

        output_csv = os.path.join(output_dir, f"{filename}.csv")
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['sentence_id', 'author', 'sentence'])
            writer.writeheader()
            writer.writerows(sentences)

        print(f"Saved: {output_csv}")
    except Exception as e:
        print(f"Error processing {path}: {e}")


def process_all_files(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for path in find_writing_sessions(dataset_dir):
        process_file(path, output_dir)


if __name__ == '__main__':
    dataset_directory = './files'
    output_directory = './output_csvs'
    process_all_files(dataset_directory, output_directory)
