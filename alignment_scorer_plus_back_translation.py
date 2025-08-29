import os
import re
import math
import sys
import numpy as np
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# Configuration
BASE_DIR = "/mnt/c/Users/dcurl/Desktop/Input"
ANCHOR_DIR = os.path.join(BASE_DIR, "Anchor")  # Source text
HEART_DIR = os.path.join(BASE_DIR, "Heart")    # Target language
BACK_DIR = os.path.join(BASE_DIR, "Back")      # Back translation

def preprocess_text(text, max_length=500000):
    """Preprocess text into character-level trigrams with punctuation as tokens"""
    if len(text) > max_length:
        text = text[:max_length]
        print(f"Warning: Text truncated to {max_length} characters")
    
    text = re.sub(r'\s+', ' ', text.lower().strip())
    tokens = list(text)
    trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return trigrams

def read_line_file(filepath, line_number):
    """Read a specific line from a text file (1-indexed)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if 0 <= line_number - 1 < len(lines):
                return lines[line_number - 1].strip()
            else:
                print(f"Warning: Line {line_number} not found in {filepath}")
                return ""
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            if 0 <= line_number - 1 < len(lines):
                return lines[line_number - 1].strip()
            else:
                print(f"Warning: Line {line_number} not found in {filepath}")
                return ""
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return ""

def calculate_back_translation_quality(anchor_text, back_text):
    """Calculate back translation quality score"""
    if not anchor_text or not back_text:
        return 0.0
    
    # Preprocess both texts
    anchor_tokens = preprocess_text(anchor_text)
    back_tokens = preprocess_text(back_text)
    
    if not anchor_tokens or not back_tokens:
        return 0.0
    
    # Calculate multiple similarity metrics
    scores = {}
    
    # 1. Lexical overlap
    set_anchor = set(anchor_tokens)
    set_back = set(back_tokens)
    intersection = len(set_anchor.intersection(set_back))
    union = len(set_anchor.union(set_back))
    scores['jaccard'] = intersection / union if union > 0 else 0
    
    # 2. Cosine similarity
    counter_anchor = Counter(anchor_tokens)
    counter_back = Counter(back_tokens)
    vocab = set_anchor | set_back
    
    dot_product = sum(counter_anchor[word] * counter_back[word] for word in vocab)
    mag_anchor = math.sqrt(sum(count * count for count in counter_anchor.values()))
    mag_back = math.sqrt(sum(count * count for count in counter_back.values()))
    
    scores['cosine'] = dot_product / (mag_anchor * mag_back) if mag_anchor > 0 and mag_back > 0 else 0
    
    # 3. Sequence similarity
    matcher = SequenceMatcher(None, anchor_text.lower(), back_text.lower())
    scores['sequence'] = matcher.ratio()
    
    # 4. Length ratio penalty
    len_ratio = min(len(anchor_tokens), len(back_tokens)) / max(len(anchor_tokens), len(back_tokens)) if max(len(anchor_tokens), len(back_tokens)) > 0 else 0
    scores['length_ratio'] = len_ratio
    
    # Weighted combination for back translation quality
    weights = {
        'jaccard': 0.3,
        'cosine': 0.3,
        'sequence': 0.3,
        'length_ratio': 0.1
    }
    
    quality_score = sum(scores[metric] * weight for metric, weight in weights.items())
    
    return quality_score

def calculate_translation_consistency(anchor_text, heart_text, back_text):
    """Calculate consistency scores between all three texts"""
    if not all([anchor_text, heart_text, back_text]):
        return {}
    
    # Calculate pairwise similarities
    anchor_heart = calculate_similarity_pair(anchor_text, heart_text)
    anchor_back = calculate_back_translation_quality(anchor_text, back_text)
    heart_back = calculate_similarity_pair(heart_text, back_text)
    
    # Consistency metrics
    consistency_scores = {
        'anchor_heart_similarity': anchor_heart,
        'anchor_back_similarity': anchor_back,
        'heart_back_similarity': heart_back,
        'translation_consistency': (anchor_heart + anchor_back + heart_back) / 3,
        'back_translation_quality': anchor_back
    }
    
    return consistency_scores

def calculate_similarity_pair(text1, text2):
    """Calculate similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Cosine similarity
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    vocab = set(tokens1) | set(tokens2)
    
    dot_product = sum(counter1[word] * counter2[word] for word in vocab)
    mag1 = math.sqrt(sum(count * count for count in counter1.values()))
    mag2 = math.sqrt(sum(count * count for count in counter2.values()))
    
    return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0

def calculate_alignment_confidence(anchor_text, heart_text, back_text):
    """Calculate alignment confidence based on back translation consistency"""
    if not all([anchor_text, heart_text, back_text]):
        return 0.0
    
    # Back translation quality (how well back matches original)
    back_quality = calculate_back_translation_quality(anchor_text, back_text)
    
    # Forward translation quality estimate
    forward_quality = calculate_similarity_pair(anchor_text, heart_text)
    
    # Reverse translation quality estimate  
    reverse_quality = calculate_similarity_pair(heart_text, back_text)
    
    # Confidence is higher when:
    # 1. Back translation closely matches original (high back_quality)
    # 2. There's consistency across the translation chain
    confidence = (back_quality * 0.5 + 
                 forward_quality * 0.25 + 
                 reverse_quality * 0.25)
    
    return confidence

def analyze_line_alignment(file_number, line_number):
    """Analyze alignment quality for a specific line across all three files"""
    anchor_file = os.path.join(ANCHOR_DIR, f"{file_number}.txt")
    heart_file = os.path.join(HEART_DIR, f"{file_number}.txt")
    back_file = os.path.join(BACK_DIR, f"{file_number}.txt")
    
    # Read the specific lines
    anchor_text = read_line_file(anchor_file, line_number)
    heart_text = read_line_file(heart_file, line_number)
    back_text = read_line_file(back_file, line_number)
    
    if not all([anchor_text, heart_text, back_text]):
        return None
    
    # Calculate comprehensive scores
    consistency_scores = calculate_translation_consistency(anchor_text, heart_text, back_text)
    alignment_confidence = calculate_alignment_confidence(anchor_text, heart_text, back_text)
    
    results = {
        'file_number': file_number,
        'line_number': line_number,
        'anchor_length': len(anchor_text),
        'heart_length': len(heart_text),
        'back_length': len(back_text),
        'alignment_confidence': alignment_confidence,
        **consistency_scores
    }
    
    return results

def process_file_with_back_translation(file_number, max_lines=None):
    """Process an entire file with back translation analysis"""
    print(f"\nProcessing file {file_number} with back translation analysis...")
    
    # Check if all three files exist
    anchor_file = os.path.join(ANCHOR_DIR, f"{file_number}.txt")
    heart_file = os.path.join(HEART_DIR, f"{file_number}.txt")
    back_file = os.path.join(BACK_DIR, f"{file_number}.txt")
    
    for filepath in [anchor_file, heart_file, back_file]:
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found")
            return None
    
    # Read all files
    try:
        with open(anchor_file, 'r', encoding='utf-8') as f:
            anchor_lines = [line.strip() for line in f.readlines()]
        with open(heart_file, 'r', encoding='utf-8') as f:
            heart_lines = [line.strip() for line in f.readlines()]
        with open(back_file, 'r', encoding='utf-8') as f:
            back_lines = [line.strip() for line in f.readlines()]
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        with open(anchor_file, 'r', encoding='latin-1') as f:
            anchor_lines = [line.strip() for line in f.readlines()]
        with open(heart_file, 'r', encoding='latin-1') as f:
            heart_lines = [line.strip() for line in f.readlines()]
        with open(back_file, 'r', encoding='latin-1') as f:
            back_lines = [line.strip() for line in f.readlines()]
    
    # Ensure all files have the same number of lines
    min_lines = min(len(anchor_lines), len(heart_lines), len(back_lines))
    if max_lines:
        min_lines = min(min_lines, max_lines)
    
    print(f"Processing {min_lines} lines...")
    
    line_results = []
    high_confidence_pairs = []
    low_confidence_pairs = []
    
    for line_num in range(1, min_lines + 1):
        if line_num <= len(anchor_lines) and line_num <= len(heart_lines) and line_num <= len(back_lines):
            anchor_text = anchor_lines[line_num - 1]
            heart_text = heart_lines[line_num - 1]
            back_text = back_lines[line_num - 1]
            
            if anchor_text and heart_text and back_text:
                consistency_scores = calculate_translation_consistency(anchor_text, heart_text, back_text)
                alignment_confidence = calculate_alignment_confidence(anchor_text, heart_text, back_text)
                
                result = {
                    'line_number': line_num,
                    'alignment_confidence': alignment_confidence,
                    **consistency_scores
                }
                
                line_results.append(result)
                
                # Categorize based on confidence
                if alignment_confidence > 0.7:
                    high_confidence_pairs.append((line_num, alignment_confidence))
                elif alignment_confidence < 0.3:
                    low_confidence_pairs.append((line_num, alignment_confidence))
    
    # Calculate statistics
    if line_results:
        avg_confidence = sum(r['alignment_confidence'] for r in line_results) / len(line_results)
        avg_back_quality = sum(r['back_translation_quality'] for r in line_results) / len(line_results)
        avg_consistency = sum(r['translation_consistency'] for r in line_results) / len(line_results)
        
        summary = {
            'file_number': file_number,
            'total_lines': len(line_results),
            'avg_alignment_confidence': avg_confidence,
            'avg_back_translation_quality': avg_back_quality,
            'avg_translation_consistency': avg_consistency,
            'high_confidence_count': len(high_confidence_pairs),
            'low_confidence_count': len(low_confidence_pairs),
            'line_results': line_results
        }
        
        return summary
    else:
        return None

def generate_eflomal_quality_weights(file_number, output_dir=None):
    """Generate quality weights for eflomal based on back translation analysis"""
    if output_dir is None:
        output_dir = BASE_DIR
    
    results = process_file_with_back_translation(file_number)
    if not results:
        return
    
    # Create weights file for eflomal
    weights_file = os.path.join(output_dir, f"alignment_weights_{file_number}.txt")
    
    with open(weights_file, 'w', encoding='utf-8') as f:
        for line_result in results['line_results']:
            line_num = line_result['line_number']
            confidence = line_result['alignment_confidence']
            # Write weight for this line (higher confidence = higher weight)
            f.write(f"{confidence:.6f}\n")
    
    print(f"Generated alignment weights file: {weights_file}")
    return weights_file

def main():
    """Main function to run back translation enhanced analysis"""
    file_numbers = list(range(41, 68))  # Files 41 to 67
    
    print("Enhanced Eflomal Analysis with Back Translation")
    print("=" * 60)
    
    all_summaries = []
    
    for file_num in file_numbers:
        summary = process_file_with_back_translation(file_num, max_lines=1000)  # Limit for testing
        
        if summary:
            all_summaries.append(summary)
            
            print(f"\nFile {file_num} Summary:")
            print(f"  Total lines processed: {summary['total_lines']}")
            print(f"  Average alignment confidence: {summary['avg_alignment_confidence']:.4f}")
            print(f"  Average back translation quality: {summary['avg_back_translation_quality']:.4f}")
            print(f"  Average translation consistency: {summary['avg_translation_consistency']:.4f}")
            print(f"  High confidence pairs: {summary['high_confidence_count']}")
            print(f"  Low confidence pairs: {summary['low_confidence_count']}")
            
            # Generate weights file for eflomal
            weights_file = generate_eflomal_quality_weights(file_num)
            print(f"  Weights file generated: {os.path.basename(weights_file)}")
    
    # Overall statistics
    if all_summaries:
        print("\n" + "=" * 60)
        print("OVERALL STATISTICS:")
        print("=" * 60)
        
        total_lines = sum(s['total_lines'] for s in all_summaries)
        total_high_conf = sum(s['high_confidence_count'] for s in all_summaries)
        total_low_conf = sum(s['low_confidence_count'] for s in all_summaries)
        
        avg_confidence = sum(s['avg_alignment_confidence'] for s in all_summaries) / len(all_summaries)
        avg_back_quality = sum(s['avg_back_translation_quality'] for s in all_summaries) / len(all_summaries)
        avg_consistency = sum(s['avg_translation_consistency'] for s in all_summaries) / len(all_summaries)
        
        print(f"Total lines processed: {total_lines}")
        print(f"Average alignment confidence: {avg_confidence:.4f}")
        print(f"Average back translation quality: {avg_back_quality:.4f}")
        print(f"Average translation consistency: {avg_consistency:.4f}")
        print(f"High confidence pairs: {total_high_conf} ({total_high_conf/total_lines*100:.1f}%)")
        print(f"Low confidence pairs: {total_low_conf} ({total_low_conf/total_lines*100:.1f}%)")
        
        print(f"\nRecommendations for eflomal improvement:")
        if avg_confidence > 0.6:
            print("- Good overall alignment quality. Consider using confidence weights.")
        else:
            print("- Lower alignment quality detected. Focus on high-confidence pairs.")
        
        if avg_back_quality < 0.4:
            print("- Back translation quality is low. Consider improving translation models.")
        
        confidence_ratio = total_high_conf / (total_low_conf + 1)  # +1 to avoid division by zero
        if confidence_ratio < 2:
            print("- Consider filtering out low-confidence alignments for better eflomal training.")

if __name__ == "__main__":
    main()
