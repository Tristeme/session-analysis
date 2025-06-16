import os
import json
import csv

def process_file_for_collab_metrics(filepath):
    H = 0  # Human inserts
    M = 0  # Machine suggestions accepted
    I = 0  # Interaction events
    A = 0  # Alone writing events

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                name = event.get('eventName', '')
                source = event.get('eventSource', '')

                # Human writing: insert by user
                if name == 'text-insert' and source == 'user':
                    H += 1

                # Machine suggestion used (choose): suggestion-select by user
                if name == 'suggestion-select' and source == 'user':
                    M += 1

                # Human-GPT interaction events
                if (name in ['suggestion-select', 'suggestion-reopen', 'suggestion-up',
                             'suggestion-down'] and source == 'user'):
                    I += 1

                if name == 'text-insert' and source == 'api':
                    for op in event.get('textDelta', {}).get('ops', []):
                        if 'insert' in op and isinstance(op['insert'], str):
                            I += len(op['insert'])

                # Solo events
                if name in ['text-delete', 'text-insert', 'suggestion-close'] and source == 'user':
                    A += 1

            except json.JSONDecodeError:
                continue

    equality = 1 - abs(H - M) / (H + M) if (H + M) > 0 else 0
    mutuality = I / (I + A) if (I + A) > 0 else 0
    return H, M, I, A, round(equality, 3), round(mutuality, 3)


def process_all_sessions(dataset_dir, output_file='collab_scores.csv'):
    results = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json') or filename.endswith('.jsonl'):
            filepath = os.path.join(dataset_dir, filename)
            H, M, I, A, equality, mutuality = process_file_for_collab_metrics(filepath)
            results.append({
                'filename': filename,
                'H_user_insert': H,
                'M_model_accepted': M,
                'I_interaction': I,
                'A_alone': A,
                'equality': equality,
                'mutuality': mutuality
            })

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Collaboration metrics saved to {output_file}")


if __name__ == '__main__':
    dataset_dir = 'files'  # your folder with .json/.jsonl logs
    process_all_sessions(dataset_dir)
