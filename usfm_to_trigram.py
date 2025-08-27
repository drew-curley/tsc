import os
import re

# Input/output paths
base_dir = r"C:\Users\dcurl\Desktop\Input"
input_dir = os.path.join(base_dir, "Back")
output_dir = os.path.join(base_dir, "Back")
os.makedirs(output_dir, exist_ok=True)

# Book numbers for New Testament (Matthew to Revelation)
book_numbers = [str(i) for i in range(41, 68)]

def parse_usfm(file_path):
    verses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_verse = ''
    in_verse = False
    for line in lines:
        line = line.strip()
        if line.startswith('\\v '):
            if in_verse and current_verse:
                # Clean USFM markers
                cleaned = re.sub(r'\\[+]?[a-zA-Z]+[*]?\s*', '', current_verse)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if cleaned:
                    # Tokenize into individual characters (letters, spaces, punctuation)
                    tokens = list(cleaned.lower())
                    # Generate character-level trigrams
                    trigrams = [''.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
                    if trigrams:
                        # Join trigrams into a single line for the verse
                        verses.append(' | '.join(trigrams))
            current_verse = line.split(maxsplit=2)[2] if len(line.split(maxsplit=2)) > 2 else ''
            in_verse = True
        elif in_verse and line:
            current_verse += ' ' + line
    
    if in_verse and current_verse:
        cleaned = re.sub(r'\\[+]?[a-zA-Z]+[*]?\s*', '', current_verse)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            tokens = list(cleaned.lower())
            trigrams = [''.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
            if trigrams:
                verses.append(' | '.join(trigrams))
    
    return verses

# Process each USFM file
for num in book_numbers:
    usfm_file = os.path.join(input_dir, f"{num}.usfm")
    txt_file = os.path.join(output_dir, f"{num}.txt")
    
    if not os.path.exists(usfm_file):
        print(f"Warning: {usfm_file} not found. Skipping.")
        continue

    print(f"Processing {usfm_file}...")
    verses = parse_usfm(usfm_file)
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        for verse_trigrams in verses:
            f.write(verse_trigrams + '\n')
    
    print(f"Wrote {len(verses)} verses to {txt_file}")
