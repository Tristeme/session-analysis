import os
import json
import csv


def process_file(filepath):
    """处理单个 JSON 文件，返回统计结果"""
    api_queries = 0
    suggestions_accepted = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                name = event.get('eventName', '')
                source = event.get('eventSource', '')
                if name == 'suggestion-open' and source == 'api':
                    api_queries += 1
                elif name == 'suggestion-select':
                    suggestions_accepted += 1
            except json.JSONDecodeError:
                continue

    acceptance_rate = (suggestions_accepted / api_queries) if api_queries else 0.0
    return api_queries, suggestions_accepted, acceptance_rate


def process_all_sessions(dataset_dir, output_dir, summary_file):
    os.makedirs(output_dir, exist_ok=True)
    summary_data = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(dataset_dir, filename)
            api_queries, accepted, rate = process_file(filepath)

            # 写入每个文件的单独统计 CSV
            output_csv_path = os.path.join(output_dir, filename.replace('.json', '.csv'))
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'api_queries', 'suggestions_accepted', 'acceptance_rate'])
                writer.writerow([filename, api_queries, accepted, f'{rate:.2f}'])

            summary_data.append([filename, api_queries, accepted, f'{rate:.2f}'])

    # 写入 summary 文件
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'api_queries', 'suggestions_accepted', 'acceptance_rate'])
        writer.writerows(summary_data)

    print(f"Processed {len(summary_data)} files. Summary saved to {summary_file}")


if __name__ == '__main__':
    dataset_directory = './files'
    output_directory = 'output_csvs'
    summary_csv = 'summary.csv'

    process_all_sessions(dataset_directory, output_directory, summary_csv)
