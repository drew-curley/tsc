import os
import re
import math
import sys
import numpy as np
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import torch

# Configuration
BASE_DIR = "/mnt/c/Users/dcurl/Desktop/Input"
ANCHOR_DIR = os.path.join(BASE_DIR, "Anchor")  # Source text (high-resource)
HEART_DIR = os.path.join(BASE_DIR, "Heart")    # Target language (zero-resource)
BACK_DIR = os.path.join(BASE_DIR, "Back")      # Back translation (high-resource)

# Check for CUDA and set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load multilingual Sentence-BERT model (once, for efficiency)
MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE)

def preprocess_text(text):
    """Preprocess text for neural embedding (normalize, no trigrams needed)"""
    # Normalize whitespace and convert to lowercase
    return re.sub(r'\s+', ' ', text.lower().strip())

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
    """Calculate back translation quality score using neural embeddings and sequence similarity"""
    if not anchor_text or not back_text:
        return 0.0
    
    # Preprocess texts
    anchor_text = preprocess_text(anchor_text)
    back_text = preprocess_text(back_text)
    
    # Calculate multiple similarity metrics
    scores = {}
    
    # 1. Neural embedding similarity (SBERT)
    emb_anchor = MODEL.encode(anchor_text, convert_to_tensor=True, device=DEVICE)
    emb_back = MODEL.encode(back_text, convert_to_tensor=True, device=DEVICE)
    scores['neural_cosine'] = util.cos_sim(emb_anchor, emb_back)[0][0].item()
    
    # 2. Sequence similarity (retained for robustness)
    matcher = SequenceMatcher(None, anchor_text.lower(), back_text.lower())
    scores['sequence'] = matcher.ratio()
    
    # 3. Length ratio penalty
    len_ratio = min(len(anchor_text), len(back_text)) / max(len(anchor_text), len(back_text)) if max(len(anchor_text), len(back_text)) > 0 else 0
    scores['length_ratio'] = len_ratio
    
    # Weighted combination for back translation quality
    weights = {
        'neural_cosine': 0.6,  # Higher weight for neural similarity
        'sequence': 0.3,
        'length_ratio': 0.1
    }
    
    quality_score = sum(scores[metric] * weight for metric, weight in weights.items())
    
    return quality_score

def calculate_similarity_pair(text1, text2):
    """Calculate similarity between two texts using neural embeddings"""
    if not text1 or not text2:
        return 0.0
    
    # Preprocess texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # Compute neural cosine similarity
    emb1 = MODEL.encode(text1, convert_to_tensor=True, device=DEVICE)
    emb2 = MODEL.encode(text2, convert_to_tensor=True, device=DEVICE)
    return util.cos_sim(emb1, emb2)[0][0].item()

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

def process_file_with_back_translation(file_number):
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
    
    # Diagnostic: Report line counts
    print(f"Anchor lines: {len(anchor_lines)}")
    print(f"Heart lines: {len(heart_lines)}")
    print(f"Back lines: {len(back_lines)}")
    
    # Ensure all files have the same number of lines
    min_lines = min(len(anchor_lines), len(heart_lines), len(back_lines))
    print(f"Processing {min_lines} lines...")
    
    line_results = []
    high_confidence_pairs = []
    low_confidence_pairs = []
    skipped_lines = []
    
    for line_num in range(1, min_lines + 1):
        anchor_text = anchor_lines[line_num - 1]
        heart_text = heart_lines[line_num - 1]
        back_text = back_lines[line_num - 1]
        
        if anchor_text and heart_text and back_text:
            consistency_scores = calculate_translation_consistency(anchor_text, heart_text, back_text)
            alignment_confidence = calculate_alignment_confidence(anchor_text, heart_text, back_text)
            
            # Scale confidence weights
            scaled_confidence = alignment_confidence
            if alignment_confidence < 0.3:
                scaled_confidence = 0.0  # Suppress low-confidence alignments
            elif alignment_confidence > 0.7:
                scaled_confidence = min(1.0, alignment_confidence * 1.5)  # Boost high-confidence alignments
            
            result = {
                'line_number': line_num,
                'alignment_confidence': alignment_confidence,
                'scaled_confidence': scaled_confidence,
                **consistency_scores
            }
            
            line_results.append(result)
            
            # Categorize based on confidence
            if alignment_confidence > 0.7:
                high_confidence_pairs.append((line_num, alignment_confidence))
            elif alignment_confidence < 0.3:
                low_confidence_pairs.append((line_num, alignment_confidence))
        else:
            skipped_lines.append(line_num)
    
    # Report skipped lines
    if skipped_lines:
        print(f"Warning: Skipped {len(skipped_lines)} lines due to empty content: {skipped_lines[:10]}{'...' if len(skipped_lines) > 10 else ''}")
    
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
            'skipped_lines_count': len(skipped_lines),
            'line_results': line_results
        }
        
        return summary
    else:
        print("Error: No valid lines processed for this file.")
        return None

def generate_eflomal_quality_weights(file_number, output_dir=None):
    """Generate quality weights for eflomal based on back translation analysis"""
    if output_dir is None:
        output_dir = BASE_DIR
    
    results = process_file_with_back_translation(file_number)
    if not results:
        return None
    
    # Create weights file for eflomal
    weights_file = os.path.join(output_dir, f"alignment_weights_{file_number}.txt")
    
    with open(weights_file, 'w', encoding='utf-8') as f:
        for line_result in results['line_results']:
            # Use scaled confidence for eflomal weights
            confidence = line_result['scaled_confidence']
            f.write(f"{confidence:.6f}\n")
    
    print(f"Generated alignment weights file: {weights_file}")
    return weights_file

def filter_high_confidence(file_num, threshold=0.5, output_dir=None):
    """Filter high-confidence verse pairs for neural alignment (e.g., Awesome-Align)"""
    if output_dir is None:
        output_dir = BASE_DIR
    
    results = process_file_with_back_translation(file_num)
    if not results:
        return None
    
    anchor_file = os.path.join(ANCHOR_DIR, f"{file_num}.txt")
    heart_file = os.path.join(HEART_DIR, f"{file_num}.txt")
    filtered_anchor = os.path.join(output_dir, f"filtered_anchor_{file_num}.txt")
    filtered_heart = os.path.join(output_dir, f"filtered_heart_{file_num}.txt")
    
    with open(anchor_file, 'r', encoding='utf-8') as fa, open(heart_file, 'r', encoding='utf-8') as fh, \
         open(filtered_anchor, 'w', encoding='utf-8') as fa_out, open(filtered_heart, 'w', encoding='utf-8') as fh_out:
        for line_result, anchor_line, heart_line in zip(results['line_results'], fa, fh):
            if line_result['scaled_confidence'] > threshold:
                fa_out.write(anchor_line)
                fh_out.write(heart_line)
    
    print(f"Generated filtered files: {os.path.basename(filtered_anchor)}, {os.path.basename(filtered_heart)}")
    return filtered_anchor, filtered_heart

def main():
    """Main function to run neural-enhanced back translation analysis with CUDA support"""
    file_numbers = list(range(41, 68))  # Files 41 to 67 (New Testament)
    
    print(f"Neural-Enhanced Eflomal Analysis with Back Translation (Device: {DEVICE})")
    print("=" * 60)
    
    all_summaries = []
    total_lines_per_file = {}
    
    for file_num in file_numbers:
        summary = process_file_with_back_translation(file_num)
        
        if summary:
            all_summaries.append(summary)
            total_lines_per_file[file_num] = summary['total_lines']
            
            print(f"\nFile {file_num} Summary:")
            print(f"  Total lines processed: {summary['total_lines']}")
            print(f"  Skipped lines: {summary['skipped_lines_count']}")
            print(f"  Average alignment confidence: {summary['avg_alignment_confidence']:.4f}")
            print(f"  Average back translation quality: {summary['avg_back_translation_quality']:.4f}")
            print(f"  Average translation consistency: {summary['avg_translation_consistency']:.4f}")
            print(f"  High confidence pairs: {summary['high_confidence_count']}")
            print(f"  Low confidence pairs: {summary['low_confidence_count']}")
            
            # Generate weights file for eflomal
            weights_file = generate_eflomal_quality_weights(file_num)
            print(f"  Weights file generated: {os.path.basename(weights_file)}")
            
            # Generate filtered files for Awesome-Align
            filtered_anchor, filtered_heart = filter_high_confidence(file_num, threshold=0.5)
    
    # Overall statistics
    if all_summaries:
        print("\n" + "=" * 60)
        print("OVERALL STATISTICS:")
        print("=" * 60)
        
        total_lines = sum(s['total_lines'] for s in all_summaries)
        total_high_conf = sum(s['high_confidence_count'] for s in all_summaries)
        total_low_conf = sum(s['low_confidence_count'] for s in all_summaries)
        total_skipped = sum(s['skipped_lines_count'] for s in all_summaries)
        
        avg_confidence = sum(s['avg_alignment_confidence'] for s in all_summaries) / len(all_summaries)
        avg_back_quality = sum(s['avg_back_translation_quality'] for s in all_summaries) / len(all_summaries)
        avg_consistency = sum(s['avg_translation_consistency'] for s in all_summaries) / len(all_summaries)
        
        print(f"Total files processed: {len(all_summaries)}")
        print(f"Total lines processed: {total_lines}")
        print(f"Total skipped lines: {total_skipped}")
        print(f"Average alignment confidence: {avg_confidence:.4f}")
        print(f"Average back translation quality: {avg_back_quality:.4f}")
        print(f"Average translation consistency: {avg_consistency:.4f}")
        print(f"High confidence pairs: {total_high_conf} ({total_high_conf/total_lines*100:.1f}%)")
        print(f"Low confidence pairs: {total_low_conf} ({total_low_conf/total_lines*100:.1f}%)")
        
        # Detailed line count per file
        print("\nLines processed per file:")
        for file_num in file_numbers:
            lines = total_lines_per_file.get(file_num, 0)
            print(f"  File {file_num}.txt: {lines} lines")
        
        print(f"\nRecommendations for alignment improvement:")
        if avg_confidence > 0.6:
            print("- Good overall alignment quality. Confidence weights generated for eflomal.")
            print("- Consider using neural aligner (e.g., Awesome-Align) for better performance with zero-resource Heart language (see instructions below).")
        else:
            print("- Lower alignment quality detected. Focus on high-confidence pairs.")
        
        if avg_back_quality < 0.4:
            print("- Back translation quality is low. Consider improving translation models for zero-resource Heart language.")
        
        confidence_ratio = total_high_conf / (total_low_conf + 1)  # +1 to avoid division by zero
        if confidence_ratio < 2:
            print("- Consider filtering out low-confidence alignments for better training.")
        
        # Instructions for alignment (eflomal and Awesome-Align)
        print("\nInstructions for using confidence weights:")
        print("Option 1: Use with eflomal (statistical aligner):")
        print("- Ensure eflomal is installed (e.g., via pip install eflomal).")
        print("- Use generated weights files (alignment_weights_{file_number}.txt) with eflomal.")
        print("- Example command:")
        print(f"  eflomal -m 3 -s {ANCHOR_DIR}/{file_numbers[0]}.txt -t {HEART_DIR}/{file_numbers[0]}.txt --sentence-weights {BASE_DIR}/alignment_weights_{file_numbers[0]}.txt -o {BASE_DIR}/alignment_output_{file_numbers[0]}.txt")
        print("- Batch command:")
        print(f"  for i in {{41..67}}; do eflomal -m 3 -s {ANCHOR_DIR}/$i.txt -t {HEART_DIR}/$i.txt --sentence-weights {BASE_DIR}/alignment_weights_$i.txt -o {BASE_DIR}/alignment_output_$i.txt; done")
        
        print("\nOption 2: Use Awesome-Align (neural aligner, recommended for zero-resource Heart language):")
        print("- Install Awesome-Align: git clone https://github.com/neulab/awesome-align; pip install -r requirements.txt")
        print("- Ensure CUDA is enabled for faster processing (set CUDA_VISIBLE_DEVICES=0 if needed).")
        print("- Use filtered files (filtered_anchor_{file}.txt, filtered_heart_{file}.txt) for high-confidence pairs.")
        print("- Example command for file 41:")
        print(f"  CUDA_VISIBLE_DEVICES=0 awesome-align --data_file {BASE_DIR}/filtered_anchor_41.txt|||{BASE_DIR}/filtered_heart_41.txt --model_name_or_path bert-base-multilingual-cased --extraction softmax --batch_size 32 --output_file {BASE_DIR}/neural_alignment_output_41.txt")
        print("- Batch command:")
        print(f"  for i in {{41..67}}; do CUDA_VISIBLE_DEVICES=0 awesome-align --data_file {BASE_DIR}/filtered_anchor_$i.txt|||{BASE_DIR}/filtered_heart_$i.txt --model_name_or_path bert-base-multilingual-cased --extraction softmax --batch_size 32 --output_file {BASE_DIR}/neural_alignment_output_$i.txt; done")
        print("- Note: Awesome-Align uses CUDA automatically if available; adjust batch_size based on GPU memory.")
        
        # Check for expected line count
        expected_lines = 7943
        if total_lines < expected_lines:
            print(f"\nWarning: Processed {total_lines} lines, expected {expected_lines}. Possible issues:")
            print("- Some files have fewer lines than others (check Anchor, Heart, Back line counts).")
            print("- Empty lines or missing content in some files.")
            print("- Verify input files for completeness and consistency.")
    else:
        print("Error: No valid files processed.")

if __name__ == "__main__":
    main()
