import os
import csv
import json
from typing import List, Dict, Tuple
import re
import openai
from openai import OpenAI
import time


def rouge_l_score(reference: str, candidate: str) -> float:
    """
    Calculate ROUGE-L score between reference and candidate texts
    """
    if not reference or not candidate:
        return 0.0

    # Tokenize (simple word-based tokenization)
    ref_tokens = re.findall(r'\b\w+\b', reference.lower())
    cand_tokens = re.findall(r'\b\w+\b', candidate.lower())

    if not ref_tokens or not cand_tokens:
        return 0.0

    # Find LCS (Longest Common Subsequence)
    lcs_length = find_lcs_length(ref_tokens, cand_tokens)

    if lcs_length == 0:
        return 0.0

    # Calculate precision and recall
    precision = lcs_length / len(cand_tokens)
    recall = lcs_length / len(ref_tokens)

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0

    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def find_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Find the length of the Longest Common Subsequence using dynamic programming
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def load_csv_data(csv_dir: str) -> List[Dict]:
    """
    Load all CSV files from the specified directory
    """
    all_sentences = []

    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(csv_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sentence_data = {
                            'sentence_id': row.get('sentence_id', ''),
                            'author': row.get('author', '').strip(),
                            'sentence': row.get('sentence', '').strip(),
                            'filename': filename
                        }
                        # Only add if sentence is not empty
                        if sentence_data['sentence']:
                            all_sentences.append(sentence_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    return all_sentences


def analyze_keypoints(sentences: List[Dict], keypoints: List[str]) -> List[Dict]:
    """
    Analyze keypoints and find most similar sentences using ROUGE-L
    """
    results = []

    for i, keypoint in enumerate(keypoints):
        print(f"Analyzing keypoint {i + 1}/{len(keypoints)}: {keypoint[:50]}...")

        # Calculate similarity scores for all sentences
        similarities = []
        for sentence_data in sentences:
            score = rouge_l_score(keypoint, sentence_data['sentence'])
            similarities.append({
                **sentence_data,
                'similarity_score': score
            })

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Get top 3 most similar sentences
        top_similar = similarities[:3]

        # Determine author classification
        if top_similar:
            best_match = top_similar[0]
            author = best_match['author']

            # Classify author
            if author == 'api':
                author_type = 'API'
            elif author == 'user':
                author_type = 'USER'
            elif author == 'user_and_api':
                author_type = 'USER_AND_API'
            else:
                author_type = author.upper()

            result = {
                'keypoint': keypoint,
                'keypoint_index': i + 1,
                'best_match_sentence': best_match['sentence'],
                'best_match_score': round(best_match['similarity_score'], 4),
                'best_match_author': author_type,
                'best_match_sentence_id': best_match['sentence_id'],
                'best_match_filename': best_match['filename'],
                'top_3_matches': [
                    {
                        'sentence': match['sentence'][:100] + '...' if len(match['sentence']) > 100 else match[
                            'sentence'],
                        'score': round(match['similarity_score'], 4),
                        'author': match['author'],
                        'sentence_id': match['sentence_id'],
                        'filename': match['filename']
                    }
                    for match in top_similar
                ]
            }
            results.append(result)
        else:
            results.append({
                'keypoint': keypoint,
                'keypoint_index': i + 1,
                'best_match_sentence': 'No match found',
                'best_match_score': 0.0,
                'best_match_author': 'NONE',
                'best_match_sentence_id': '',
                'best_match_filename': '',
                'top_3_matches': []
            })

    return results


def save_results(results: List[Dict], output_file: str = 'keypoint_analysis_results.csv'):
    """
    Save analysis results to CSV file
    """
    if not results:
        print("No results to save")
        return

    fieldnames = [
        'keypoint_index',
        'keypoint',
        'best_match_sentence',
        'best_match_score',
        'best_match_author',
        'best_match_sentence_id',
        'best_match_filename'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'keypoint_index': result['keypoint_index'],
                'keypoint': result['keypoint'],
                'best_match_sentence': result['best_match_sentence'],
                'best_match_score': result['best_match_score'],
                'best_match_author': result['best_match_author'],
                'best_match_sentence_id': result['best_match_sentence_id'],
                'best_match_filename': result['best_match_filename']
            })

    print(f"Results saved to {output_file}")


def save_detailed_results(results: List[Dict], output_file: str = 'keypoint_analysis_detailed.json'):
    """
    Save detailed analysis results to JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to {output_file}")


def load_api_key_from_csv(csv_file: str = 'api_keys.csv', host: str = 'openai', domain: str = 'default') -> str:
    """
    Load API key from CSV file
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('host', '').strip().lower() == host.lower() and row.get('domain',
                                                                                   '').strip().lower() == domain.lower():
                    return row.get('key', '').strip()

        print(f"API key not found for host='{host}', domain='{domain}' in {csv_file}")
        return None
    except FileNotFoundError:
        print(f"API keys file '{csv_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading API keys file: {e}")
        return None


def extract_keypoints_with_gpt4o(text: str, api_key: str, max_keypoints: int = 15) -> List[str]:
    """
    Extract key points from text using ChatGPT-4o API
    """
    client = OpenAI(api_key=api_key)

    prompt = f"""
    Please analyze the following text and extract the most important key points or main ideas. 
    Return only the key points as a JSON list of strings, with each key point being a concise sentence or phrase.
    Extract up to {max_keypoints} key points, focusing on the most significant themes, arguments, or ideas.

    Text to analyze:
    {text}

    Return format: ["key point 1", "key point 2", "key point 3", ...]
    """

    try:
        print("Calling ChatGPT-4o API to extract key points...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an expert at identifying key points and main ideas from text. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        response_text = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            keypoints = json.loads(response_text)
            if isinstance(keypoints, list):
                print(f"Successfully extracted {len(keypoints)} key points from GPT-4o")
                return keypoints
            else:
                print("GPT-4o response is not a list, trying to extract manually...")
                return extract_keypoints_from_text(response_text)
        except json.JSONDecodeError:
            print("Failed to parse JSON, trying to extract key points manually...")
            return extract_keypoints_from_text(response_text)

    except Exception as e:
        print(f"Error calling GPT-4o API: {e}")
        print("Falling back to manual key point extraction...")
        return extract_keypoints_from_text(text)


def extract_keypoints_from_text(text: str) -> List[str]:
    """
    Fallback method to extract key points from text manually
    """
    # Simple extraction based on sentence structure
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # Filter for sentences that might be key points
    keypoints = []
    for sentence in sentences:
        # Look for sentences with key indicators
        if any(indicator in sentence.lower() for indicator in [
            'i think', 'i believe', 'main point', 'important', 'key', 'significant',
            'the problem', 'the solution', 'we should', 'need to', 'have to'
        ]):
            keypoints.append(sentence)

    # If we don't have enough, take the first few substantial sentences
    if len(keypoints) < 5:
        keypoints.extend([s for s in sentences[:10] if s not in keypoints])

    return keypoints[:15]  # Limit to 15 key points


def get_all_text_from_sentences(sentences: List[Dict]) -> str:
    """
    Combine all sentences into a single text for key point extraction
    """
    all_text = ""
    for sentence_data in sentences:
        all_text += sentence_data['sentence'] + " "
    return all_text.strip()


def print_summary(results: List[Dict]):
    """
    Print a summary of the analysis results
    """
    print("\n" + "=" * 80)
    print("KEYPOINT ANALYSIS SUMMARY")
    print("=" * 80)

    author_counts = {'API': 0, 'USER': 0, 'USER_AND_API': 0, 'NONE': 0}

    for result in results:
        author = result['best_match_author']
        if author in author_counts:
            author_counts[author] += 1

        print(f"\nKeypoint {result['keypoint_index']}: {result['keypoint'][:60]}...")
        print(f"  Best Match Score: {result['best_match_score']}")
        print(f"  Author Type: {result['best_match_author']}")
        print(f"  Sentence: {result['best_match_sentence'][:100]}...")
        print(f"  File: {result['best_match_filename']}")

    print(f"\n" + "-" * 50)
    print("AUTHOR DISTRIBUTION:")
    for author, count in author_counts.items():
        percentage = (count / len(results)) * 100 if results else 0
        print(f"  {author}: {count} ({percentage:.1f}%)")

    avg_score = sum(r['best_match_score'] for r in results) / len(results) if results else 0
    print(f"  Average Similarity Score: {avg_score:.4f}")


def main():
    # Configuration
    csv_dir = 'output_csvs'  # Directory containing CSV files

    # Load API key from CSV file
    print("Loading API key from api_keys.csv...")
    api_key = load_api_key_from_csv('api_keys.csv', host='openai', domain='default')

    if not api_key:
        # Fallback to environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print("Using API key from environment variable")
        else:
            print("Error: No API key found. Please:")
            print("1. Add your OpenAI API key to api_keys.csv with format: openai,default,your-key")
            print("2. Or set OPENAI_API_KEY environment variable")
            return
    else:
        print("Successfully loaded API key from api_keys.csv")

    print(f"Loading CSV files from {csv_dir}...")
    sentences = load_csv_data(csv_dir)
    print(f"Loaded {len(sentences)} sentences from CSV files")

    if not sentences:
        print("No sentences found. Please check your CSV files.")
        return

    # Extract key points using GPT-4o
    print("\nExtracting key points using ChatGPT-4o...")
    all_text = get_all_text_from_sentences(sentences)
    print(f"Total text length: {len(all_text)} characters")

    # If text is too long, truncate it (GPT-4o has token limits)
    if len(all_text) > 12000:  # Roughly 3000 tokens
        print("Text is too long, truncating to first 12000 characters...")
        all_text = all_text[:12000]

    keypoints = extract_keypoints_with_gpt4o(all_text, api_key, max_keypoints=15)

    if not keypoints:
        print("Failed to extract key points. Exiting.")
        return

    print(f"\nExtracted {len(keypoints)} key points:")
    for i, kp in enumerate(keypoints, 1):
        print(f"  {i}. {kp}")

    print(f"\nAnalyzing {len(keypoints)} keypoints...")
    results = analyze_keypoints(sentences, keypoints)

    # Save results
    save_results(results)
    save_detailed_results(results)

    # Save the extracted keypoints as well
    with open('extracted_keypoints.json', 'w', encoding='utf-8') as f:
        json.dump(keypoints, f, indent=2, ensure_ascii=False)
    print("Extracted keypoints saved to extracted_keypoints.json")

    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()