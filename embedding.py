import os
import csv
import json
from typing import List, Dict, Tuple
import re
import openai
from openai import OpenAI
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Check if CUDA is available and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the sentence transformer model
print("Loading Sentence-BERT model...")
try:
    # Using a multilingual model that works well for various languages
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Successfully loaded Sentence-BERT model")
except Exception as e:
    print(f"Error loading Sentence-BERT model: {e}")
    print("Please install sentence-transformers: pip install sentence-transformers")
    exit(1)


def embedding_similarity_score(reference: str, candidate: str) -> float:
    """
    Calculate semantic similarity between reference and candidate texts using Sentence-BERT embeddings
    """
    if not reference or not candidate:
        return 0.0

    try:
        # Get embeddings for both texts
        embeddings = model.encode([reference, candidate])

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        similarity_score = similarity_matrix[0][0]

        # Ensure the score is between 0 and 1
        return max(0.0, min(1.0, similarity_score))

    except Exception as e:
        print(f"Error calculating embedding similarity: {e}")
        return 0.0


def batch_embedding_similarity_scores(reference: str, candidates: List[str]) -> List[float]:
    """
    Calculate semantic similarity between one reference and multiple candidates using batch processing
    This is more efficient than calculating similarities one by one
    """
    if not reference or not candidates:
        return [0.0] * len(candidates)

    try:
        # Filter out empty candidates
        valid_candidates = [c for c in candidates if c]
        if not valid_candidates:
            return [0.0] * len(candidates)

        # Prepare all texts for encoding
        all_texts = [reference] + valid_candidates

        # Get embeddings for all texts at once (more efficient)
        embeddings = model.encode(all_texts)

        # Calculate cosine similarities between reference and all candidates
        reference_embedding = embeddings[0:1]  # Keep as 2D array
        candidate_embeddings = embeddings[1:]

        similarity_scores = cosine_similarity(reference_embedding, candidate_embeddings)[0]

        # Ensure scores are between 0 and 1
        similarity_scores = np.clip(similarity_scores, 0.0, 1.0)

        return similarity_scores.tolist()

    except Exception as e:
        print(f"Error calculating batch embedding similarities: {e}")
        return [0.0] * len(candidates)


def rouge_l_score(reference: str, candidate: str) -> float:
    """
    Calculate ROUGE-L score between reference and candidate texts
    (Kept as fallback option)
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


def load_single_csv_data(filepath: str) -> List[Dict]:
    """
    Load data from a single CSV file
    """
    sentences = []
    filename = os.path.basename(filepath)

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
                    sentences.append(sentence_data)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

    return sentences


def get_csv_files(csv_dir: str) -> List[str]:
    """
    Get all CSV files from the specified directory
    """
    csv_files = []
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(csv_dir, filename))
    return csv_files


def analyze_keypoints_in_file(sentences: List[Dict], keypoints: List[str], filename: str, use_embedding: bool = True) -> \
List[Dict]:
    """
    Analyze keypoints and find most similar sentences within the same file using Sentence-BERT embeddings or ROUGE-L
    """
    results = []

    print(
        f"  Analyzing {len(keypoints)} keypoints in {filename} using {'Sentence-BERT embeddings' if use_embedding else 'ROUGE-L'}...")

    for i, keypoint in enumerate(keypoints):
        print(f"    Processing keypoint {i + 1}/{len(keypoints)}: {keypoint[:50]}...")

        if use_embedding:
            # Use batch processing for efficiency
            candidate_sentences = [s['sentence'] for s in sentences]
            similarity_scores = batch_embedding_similarity_scores(keypoint, candidate_sentences)

            # Create similarity data with scores
            similarities = []
            for j, sentence_data in enumerate(sentences):
                similarities.append({
                    **sentence_data,
                    'similarity_score': similarity_scores[j] if j < len(similarity_scores) else 0.0
                })
        else:
            # Use ROUGE-L (original method)
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
                'source_file': filename,
                'keypoint': keypoint,
                'keypoint_index': i + 1,
                'best_match_sentence': best_match['sentence'],
                'best_match_score': round(best_match['similarity_score'], 4),
                'best_match_author': author_type,
                'best_match_sentence_id': best_match['sentence_id'],
                'best_match_filename': best_match['filename'],
                'similarity_method': 'sentence-bert' if use_embedding else 'rouge-l',
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
                'source_file': filename,
                'keypoint': keypoint,
                'keypoint_index': i + 1,
                'best_match_sentence': 'No match found',
                'best_match_score': 0.0,
                'best_match_author': 'NONE',
                'best_match_sentence_id': '',
                'best_match_filename': filename,
                'similarity_method': 'sentence-bert' if use_embedding else 'rouge-l',
                'top_3_matches': []
            })

    return results


def save_results_per_file(all_results: Dict[str, List[Dict]], output_dir: str = 'results_per_file'):
    """
    Save analysis results to separate CSV files for each input file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        'source_file',
        'keypoint_index',
        'keypoint',
        'best_match_sentence',
        'best_match_score',
        'best_match_author',
        'best_match_sentence_id',
        'best_match_filename',
        'similarity_method'
    ]

    # Save individual file results
    for filename, results in all_results.items():
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f'{base_name}_keypoint_analysis.csv')

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow({
                    'source_file': result['source_file'],
                    'keypoint_index': result['keypoint_index'],
                    'keypoint': result['keypoint'],
                    'best_match_sentence': result['best_match_sentence'],
                    'best_match_score': result['best_match_score'],
                    'best_match_author': result['best_match_author'],
                    'best_match_sentence_id': result['best_match_sentence_id'],
                    'best_match_filename': result['best_match_filename'],
                    'similarity_method': result.get('similarity_method', 'sentence-bert')
                })

        print(f"Results for {filename} saved to {output_file}")

    # Save combined results
    combined_output = os.path.join(output_dir, 'all_files_combined_results.csv')
    with open(combined_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filename, results in all_results.items():
            for result in results:
                writer.writerow({
                    'source_file': result['source_file'],
                    'keypoint_index': result['keypoint_index'],
                    'keypoint': result['keypoint'],
                    'best_match_sentence': result['best_match_sentence'],
                    'best_match_score': result['best_match_score'],
                    'best_match_author': result['best_match_author'],
                    'best_match_sentence_id': result['best_match_sentence_id'],
                    'best_match_filename': result['best_match_filename'],
                    'similarity_method': result.get('similarity_method', 'sentence-bert')
                })

    print(f"Combined results saved to {combined_output}")


def save_detailed_results_per_file(all_results: Dict[str, List[Dict]], all_keypoints: Dict[str, List[str]],
                                   output_dir: str = 'results_per_file'):
    """
    Save detailed analysis results to JSON files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save individual detailed results
    for filename, results in all_results.items():
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f'{base_name}_detailed_results.json')

        detailed_data = {
            'source_file': filename,
            'extracted_keypoints': all_keypoints[filename],
            'analysis_results': results,
            'similarity_method': results[0].get('similarity_method', 'sentence-bert') if results else 'sentence-bert'
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)

    # Save combined detailed results
    combined_detailed = os.path.join(output_dir, 'all_files_detailed_results.json')
    combined_data = {
        'analysis_summary': {
            'total_files': len(all_results),
            'files_processed': list(all_results.keys()),
            'similarity_method': 'sentence-bert'
        },
        'results_by_file': {}
    }

    for filename, results in all_results.items():
        combined_data['results_by_file'][filename] = {
            'extracted_keypoints': all_keypoints[filename],
            'analysis_results': results
        }

    with open(combined_detailed, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to {output_dir}")


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


def extract_keypoints_with_gpt4o(text: str, api_key: str, filename: str, max_keypoints: int = 5) -> List[str]:
    """
    Extract key points from text using ChatGPT-4o API
    """
    client = OpenAI(api_key=api_key)

    prompt = f"""
    Please analyze the following text from file "{filename}" and extract the most important key points or main ideas. 
    Return only the key points as a JSON list of strings, with each key point being a concise sentence or phrase.
    Extract up to {max_keypoints} key points, focusing on the most significant themes, arguments, or ideas specific to this conversation or document.

    Text to analyze:
    {text}

    Return format: ["key point 1", "key point 2", "key point 3", ...]
    """

    try:
        print(f"    Calling ChatGPT-4o API to extract key points for {filename}...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an expert at identifying key points and main ideas from text. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        response_text = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            keypoints = json.loads(response_text)
            if isinstance(keypoints, list):
                print(f"    Successfully extracted {len(keypoints)} key points for {filename}")
                return keypoints
            else:
                print(f"    GPT-4o response is not a list for {filename}, trying to extract manually...")
                return extract_keypoints_from_text(response_text)
        except json.JSONDecodeError:
            print(f"    Failed to parse JSON for {filename}, trying to extract key points manually...")
            return extract_keypoints_from_text(response_text)

    except Exception as e:
        print(f"    Error calling GPT-4o API for {filename}: {e}")
        print(f"    Falling back to manual key point extraction for {filename}...")
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

    return keypoints[:10]  # Limit to 10 key points per file


def get_text_from_sentences(sentences: List[Dict]) -> str:
    """
    Combine sentences from a single file into text for key point extraction
    """
    all_text = ""
    for sentence_data in sentences:
        all_text += sentence_data['sentence'] + " "
    return all_text.strip()


def print_summary_per_file(all_results: Dict[str, List[Dict]]):
    """
    Print a summary of the analysis results for each file
    """
    print("\n" + "=" * 80)
    print("KEYPOINT ANALYSIS SUMMARY (PER FILE) - Using Sentence-BERT Embeddings")
    print("=" * 80)

    total_keypoints = 0
    overall_author_counts = {'API': 0, 'USER': 0, 'USER_AND_API': 0, 'NONE': 0}

    for filename, results in all_results.items():
        print(f"\n{'=' * 50}")
        print(f"FILE: {filename}")
        print(f"{'=' * 50}")

        if not results:
            print("  No results for this file")
            continue

        author_counts = {'API': 0, 'USER': 0, 'USER_AND_API': 0, 'NONE': 0}

        for result in results:
            author = result['best_match_author']
            if author in author_counts:
                author_counts[author] += 1
                overall_author_counts[author] += 1

            print(f"  Keypoint {result['keypoint_index']}: {result['keypoint'][:60]}...")
            print(f"    Best Match Score: {result['best_match_score']} (Semantic Similarity)")
            print(f"    Author Type: {result['best_match_author']}")
            print(f"    Sentence: {result['best_match_sentence'][:80]}...")
            print()

        # File-specific statistics
        print(f"  AUTHOR DISTRIBUTION FOR {filename}:")
        for author, count in author_counts.items():
            percentage = (count / len(results)) * 100 if results else 0
            print(f"    {author}: {count} ({percentage:.1f}%)")

        avg_score = sum(r['best_match_score'] for r in results) / len(results) if results else 0
        print(f"  Average Semantic Similarity Score: {avg_score:.4f}")
        total_keypoints += len(results)

    # Overall summary
    print(f"\n{'=' * 50}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total files processed: {len(all_results)}")
    print(f"Total keypoints analyzed: {total_keypoints}")
    print(f"Similarity method: Sentence-BERT embeddings with cosine similarity")
    print(f"OVERALL AUTHOR DISTRIBUTION:")
    for author, count in overall_author_counts.items():
        percentage = (count / total_keypoints) * 100 if total_keypoints else 0
        print(f"  {author}: {count} ({percentage:.1f}%)")


def main():
    # Configuration
    csv_dir = 'output_csvs'  # Directory containing CSV files
    output_dir = 'results_per_file'  # Directory for output files
    use_embedding = True  # Set to False to use ROUGE-L instead

    print("=" * 80)
    print("KEYPOINT ANALYSIS WITH SENTENCE-BERT EMBEDDINGS")
    print("=" * 80)
    print(f"Similarity method: {'Sentence-BERT embeddings' if use_embedding else 'ROUGE-L'}")
    print(f"Device: {device}")
    print("=" * 80)

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

    # Get all CSV files
    csv_files = get_csv_files(csv_dir)
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    print(f"\nFound {len(csv_files)} CSV files to process:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # Process each file separately
    all_results = {}
    all_keypoints = {}

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\n{'-' * 60}")
        print(f"Processing file: {filename}")
        print(f"{'-' * 60}")

        # Load sentences from this file
        sentences = load_single_csv_data(csv_file)
        if not sentences:
            print(f"  No sentences found in {filename}, skipping...")
            continue

        print(f"  Loaded {len(sentences)} sentences from {filename}")

        # Extract key points for this file
        file_text = get_text_from_sentences(sentences)
        print(f"  Text length: {len(file_text)} characters")

        # If text is too long, truncate it
        if len(file_text) > 8000:  # Roughly 2000 tokens
            print(f"  Text is long, truncating to first 8000 characters...")
            file_text = file_text[:8000]

        keypoints = extract_keypoints_with_gpt4o(file_text, api_key, filename, max_keypoints=5)

        if not keypoints:
            print(f"  Failed to extract key points for {filename}, skipping...")
            continue

        print(f"  Extracted {len(keypoints)} key points:")
        for i, kp in enumerate(keypoints, 1):
            print(f"    {i}. {kp}")

        # Analyze keypoints within this file
        results = analyze_keypoints_in_file(sentences, keypoints, filename, use_embedding=use_embedding)

        # Store results
        all_results[filename] = results
        all_keypoints[filename] = keypoints

        # Add a small delay to avoid hitting API rate limits
        time.sleep(1)

    if not all_results:
        print("No files were successfully processed.")
        return

    # Save all results
    print(f"\nSaving results to {output_dir}...")
    save_results_per_file(all_results, output_dir)
    save_detailed_results_per_file(all_results, all_keypoints, output_dir)

    # Print summary
    print_summary_per_file(all_results)

    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE!")
    print(f"Results saved in directory: {output_dir}")
    print(f"Similarity method used: {'Sentence-BERT embeddings' if use_embedding else 'ROUGE-L'}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()