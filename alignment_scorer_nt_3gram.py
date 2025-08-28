import os
import re
import math
import sys
from collections import defaultdict, Counter
from difflib import SequenceMatcher

def preprocess_text(text, max_length=500000):
    """Preprocess text into character-level trigrams with punctuation as tokens"""
    # Truncate if too long to prevent memory issues
    if len(text) > max_length:
        text = text[:max_length]
        print(f"Warning: Text truncated to {max_length} characters")
    
    # Convert to lowercase and normalize whitespace
    text = re.sub(r'\s+', ' ', text.lower().strip())
    # Tokenize into individual characters (letters, spaces, punctuation)
    tokens = list(text)
    # Generate trigrams
    trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return trigrams

def calculate_lexical_overlap(text1_tokens, text2_tokens):
    """Calculate lexical overlap between two token lists"""
    set1 = set(text1_tokens)
    set2 = set(text2_tokens)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard = intersection / union if union > 0 else 0
    
    # Coverage scores
    coverage1 = intersection / len(set1) if len(set1) > 0 else 0
    coverage2 = intersection / len(set2) if len(set2) > 0 else 0
    
    return {
        'jaccard_similarity': jaccard,
        'coverage_text1': coverage1,
        'coverage_text2': coverage2,
        'avg_coverage': (coverage1 + coverage2) / 2
    }

def calculate_sequence_alignment_safe(text1, text2, max_length=10000):
    """Calculate sequence alignment with length limits"""
    # Truncate texts if too long
    if len(text1) > max_length:
        text1 = text1[:max_length]
    if len(text2) > max_length:
        text2 = text2[:max_length]
    
    matcher = SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    
    # Get matching blocks
    matching_blocks = matcher.get_matching_blocks()
    total_matches = sum(block.size for block in matching_blocks)
    
    return {
        'sequence_similarity': similarity,
        'matching_chars': total_matches,
        'total_chars_text1': len(text1),
        'total_chars_text2': len(text2)
    }

def calculate_token_alignment_safe(tokens1, tokens2, max_tokens=5000):
    """Calculate token-level alignment with limits"""
    # Truncate token lists if too long
    if len(tokens1) > max_tokens:
        tokens1 = tokens1[:max_tokens]
    if len(tokens2) > max_tokens:
        tokens2 = tokens2[:max_tokens]
    
    matcher = SequenceMatcher(None, tokens1, tokens2)
    similarity = matcher.ratio()
    
    matching_blocks = matcher.get_matching_blocks()
    total_matches = sum(block.size for block in matching_blocks)
    
    return {
        'token_similarity': similarity,
        'matching_tokens': total_matches,
        'total_tokens_text1': len(tokens1),
        'total_tokens_text2': len(tokens2)
    }

def calculate_cosine_similarity(tokens1, tokens2):
    """Calculate cosine similarity between token frequency vectors"""
    # Create frequency counters
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Get common vocabulary
    vocab = set(counter1.keys()) | set(counter2.keys())
    
    # Calculate dot product and magnitudes
    dot_product = sum(counter1[word] * counter2[word] for word in vocab)
    magnitude1 = math.sqrt(sum(count * count for count in counter1.values()))
    magnitude2 = math.sqrt(sum(count * count for count in counter2.values()))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

def calculate_edit_distance_similarity_safe(text1, text2, max_length=5000):
    """Calculate normalized edit distance similarity with length limits"""
    # Truncate if too long to prevent memory issues
    if len(text1) > max_length:
        text1 = text1[:max_length]
    if len(text2) > max_length:
        text2 = text2[:max_length]
    
    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        
        # Use space-optimized version for large texts
        if m > 1000 or n > 1000:
            prev_row = list(range(n + 1))
            curr_row = [0] * (n + 1)
            
            for i in range(1, m + 1):
                curr_row[0] = i
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        curr_row[j] = prev_row[j-1]
                    else:
                        curr_row[j] = 1 + min(prev_row[j], curr_row[j-1], prev_row[j-1])
                prev_row, curr_row = curr_row, prev_row
            
            return prev_row[n]
        else:
            # Use full matrix for smaller texts
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
    
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    
    edit_dist = edit_distance(text1, text2)
    similarity = 1 - (edit_dist / max_len)
    return max(0, similarity)

def get_file_size(filepath):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except:
        return 0

def calculate_comprehensive_alignment_score(file1_path, file2_path):
    """Calculate comprehensive alignment scores between two text files"""
    
    print(f"\nProcessing {os.path.basename(file1_path)} and {os.path.basename(file2_path)}...")
    print("Checking file sizes...")
    size1 = get_file_size(file1_path)
    size2 = get_file_size(file2_path)
    print(f"File 1 size: {size1:.2f} MB")
    print(f"File 2 size: {size2:.2f} MB")
    
    # Read files
    print("Reading files...")
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            text1 = f.read()
    except UnicodeDecodeError:
        with open(file1_path, 'r', encoding='latin-1') as f:
            text1 = f.read()
    
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            text2 = f.read()
    except UnicodeDecodeError:
        with open(file2_path, 'r', encoding='latin-1') as f:
            text2 = f.read()
    
    print(f"Text 1 length: {len(text1)} characters")
    print(f"Text 2 length: {len(text2)} characters")
    
    # Preprocess with limits
    print("Preprocessing texts...")
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    
    print(f"Tokens 1: {len(tokens1)}")
    print(f"Tokens 2: {len(tokens2)}")
    
    # Calculate various similarity metrics
    results = {}
    
    # 1. Lexical overlap
    print("Calculating lexical overlap...")
    lexical_scores = calculate_lexical_overlap(tokens1, tokens2)
    results.update(lexical_scores)
    
    # 2. Sequence alignment (character level)
    print("Calculating sequence alignment...")
    seq_scores = calculate_sequence_alignment_safe(text1.lower(), text2.lower())
    results.update(seq_scores)
    
    # 3. Token alignment
    print("Calculating token alignment...")
    token_scores = calculate_token_alignment_safe(tokens1, tokens2)
    results.update(token_scores)
    
    # 4. Cosine similarity
    print("Calculating cosine similarity...")
    results['cosine_similarity'] = calculate_cosine_similarity(tokens1, tokens2)
    
    # 5. Edit distance similarity
    print("Calculating edit distance similarity...")
    results['edit_distance_similarity'] = calculate_edit_distance_similarity_safe(text1.lower(), text2.lower())
    
    # 6. Length ratio
    len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) > 0 else 0
    results['length_ratio'] = len_ratio
    
    # 7. Comprehensive alignment score (weighted average)
    weights = {
        'avg_coverage': 0.3,
        'token_similarity': 0.25,
        'cosine_similarity': 0.2,
        'sequence_similarity': 0.15,
        'length_ratio': 0.1
    }
    
    comprehensive_score = sum(results[metric] * weight for metric, weight in weights.items())
    results['comprehensive_alignment_score'] = comprehensive_score
    
    return results

def main():
    anchor_dir = "/mnt/c/Users/dcurl/Desktop/Input/Anchor"
    heart_dir = "/mnt/c/Users/dcurl/Desktop/Input/heart"
    
    # File numbers to process (41 to 67)
    file_numbers = list(range(41, 68))
    
    print("Calculating comprehensive alignment scores for files 41.txt to 67.txt...")
    print("=" * 50)
    
    # Store results for averaging
    all_results = []
    valid_pairs = 0
    
    try:
        for num in file_numbers:
            anchor_file = os.path.join(anchor_dir, f"{num}.txt")
            heart_file = os.path.join(heart_dir, f"{num}.txt")
            
            if not (os.path.exists(anchor_file) and os.path.exists(heart_file)):
                print(f"Error: One or both files ({anchor_file}, {heart_file}) not found. Skipping.")
                continue
            
            results = calculate_comprehensive_alignment_score(anchor_file, heart_file)
            all_results.append(results)
            valid_pairs += 1
            
            # Print results for this file pair
            print(f"\nResults for {num}.txt:")
            print("LEXICAL ANALYSIS:")
            print(f"  Jaccard similarity: {results['jaccard_similarity']:.4f}")
            print(f"  Coverage (Anchor): {results['coverage_text1']:.4f}")
            print(f"  Coverage (Heart): {results['coverage_text2']:.4f}")
            print(f"  Average coverage: {results['avg_coverage']:.4f}")
            print("SEQUENCE ANALYSIS:")
            print(f"  Character-level similarity: {results['sequence_similarity']:.4f}")
            print(f"  Token-level similarity: {results['token_similarity']:.4f}")
            print(f"  Edit distance similarity: {results['edit_distance_similarity']:.4f}")
            print("VECTOR ANALYSIS:")
            print(f"  Cosine similarity: {results['cosine_similarity']:.4f}")
            print(f"  Length ratio: {results['length_ratio']:.4f}")
            print("STATISTICS:")
            print(f"  Anchor tokens: {results['total_tokens_text1']}")
            print(f"  Heart tokens: {results['total_tokens_text2']}")
            print(f"  Matching tokens: {results['matching_tokens']}")
            print(f"  Matching characters: {results['matching_chars']}")
            print(f"OVERALL ALIGNMENT SCORE: {results['comprehensive_alignment_score']:.4f}")
            print("-" * 50)
        
        # Calculate and print averages
        if valid_pairs > 0:
            print("\nAVERAGE RESULTS ACROSS ALL FILES:")
            print("=" * 50)
            avg_results = defaultdict(float)
            sum_stats = defaultdict(int)
            
            for results in all_results:
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        avg_results[key] += value
                    if key in ['total_tokens_text1', 'total_tokens_text2', 'matching_tokens', 'matching_chars']:
                        sum_stats[key] += value
            
            for key in avg_results:
                avg_results[key] /= valid_pairs
            
            print("AVERAGE LEXICAL ANALYSIS:")
            print(f"  Average Jaccard similarity: {avg_results['jaccard_similarity']:.4f}")
            print(f"  Average Coverage (Anchor): {avg_results['coverage_text1']:.4f}")
            print(f"  Average Coverage (Heart): {avg_results['coverage_text2']:.4f}")
            print(f"  Average Coverage: {avg_results['avg_coverage']:.4f}")
            print("AVERAGE SEQUENCE ANALYSIS:")
            print(f"  Average Character-level similarity: {avg_results['sequence_similarity']:.4f}")
            print(f"  Average Token-level similarity: {avg_results['token_similarity']:.4f}")
            print(f"  Average Edit distance similarity: {avg_results['edit_distance_similarity']:.4f}")
            print("AVERAGE VECTOR ANALYSIS:")
            print(f"  Average Cosine similarity: {avg_results['cosine_similarity']:.4f}")
            print(f"  Average Length ratio: {avg_results['length_ratio']:.4f}")
            print("AVERAGE STATISTICS:")
            print(f"  Average Anchor tokens: {sum_stats['total_tokens_text1'] / valid_pairs:.0f}")
            print(f"  Average Heart tokens: {sum_stats['total_tokens_text2'] / valid_pairs:.0f}")
            print(f"  Total Matching tokens: {sum_stats['matching_tokens']}")
            print(f"  Total Matching characters: {sum_stats['matching_chars']}")
            print(f"AVERAGE OVERALL ALIGNMENT SCORE: {avg_results['comprehensive_alignment_score']:.4f}")
            print("=" * 50)
        else:
            print("Error: No valid file pairs found for comparison.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
