import csv
import os
import re
from pathlib import Path


def parse_usfm(file_path):
    verses = []
    current_chapter = None
    current_verse = None
    verse_text_parts = []

    # Read the entire file into memory
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Skip metadata and structural markers
        if line.startswith(('\\id', '\\ide', '\\h', '\\toc', '\\mt', '\\s5')):
            continue

        # Handle chapter markers
        if line.startswith('\\c '):
            if verse_text_parts and current_chapter is not None and current_verse is not None:
                # Save the previous verse
                verse_text = ' '.join(verse_text_parts).strip()
                # Remove everything except letters and spaces
                verse_text = re.sub(r'[^a-zA-Z\s]', '', verse_text)
                verses.append((current_chapter, current_verse, verse_text))
                verse_text_parts = []
            try:
                current_chapter = int(line[3:].strip())
                current_verse = None  # Reset verse
            except ValueError:
                print(
                    f"Warning: Invalid chapter number in {file_path}: {line}")
                continue

        # Handle verse markers
        elif line.startswith('\\v '):
            if verse_text_parts and current_chapter is not None and current_verse is not None:
                # Save the previous verse
                verse_text = ' '.join(verse_text_parts).strip()
                verse_text = re.sub(r'[^a-zA-Z\s]', '', verse_text)
                verses.append((current_chapter, current_verse, verse_text))
                verse_text_parts = []
            verse_str = line[3:].strip()  # Get content after \v
            verse_num_end = verse_str.find(' ')  # Find end of verse number
            if verse_num_end == -1 or not verse_str[:verse_num_end].strip():
                print(
                    f"Warning: Skipping invalid verse line in {file_path}: {line}")
                continue
            try:
                current_verse = int(verse_str[:verse_num_end].strip())
                verse_text = verse_str[verse_num_end:].strip()
                if current_chapter is not None:
                    verse_text_parts.append(verse_text)
                else:
                    print(
                        f"Warning: Verse found before chapter in {file_path}: {line}")
            except ValueError as e:
                print(
                    f"Error parsing verse number in {file_path}: {line}. Error: {e}")
                continue

        # Handle continuation lines (e.g., \p, \q, \f, \w, or plain text)
        elif current_verse is not None and current_chapter is not None:
            if line.startswith(('\\p', '\\q', '\\f', '\\w')):
                text = line[3:].strip() if line[3:].strip() else ' '
                verse_text_parts.append(text)
            else:
                verse_text_parts.append(line.strip())  # Append plain text

    # Save the last verse
    if verse_text_parts and current_chapter is not None and current_verse is not None:
        verse_text = ' '.join(verse_text_parts).strip()
        verse_text = re.sub(r'[^a-zA-Z\s]', '', verse_text)
        verses.append((current_chapter, current_verse, verse_text))

    return verses


# Define New Testament books (41 to 67) with NIV versification
books = [
    {'abbrev': 'mat', 'number': 41, 'chapters': 28, 'verses': [
        25, 23, 17, 25, 48, 34, 29, 34, 38, 42, 30, 50, 58, 36, 39, 28, 27, 35, 30, 34, 46, 46, 39, 51, 46, 75, 66, 20]},
    {'abbrev': 'mrk', 'number': 42, 'chapters': 16, 'verses': [
        45, 28, 35, 41, 43, 56, 37, 38, 50, 52, 33, 44, 37, 72, 47, 20]},
    {'abbrev': 'luk', 'number': 43, 'chapters': 24, 'verses': [
        80, 52, 38, 44, 39, 49, 50, 56, 62, 42, 54, 59, 35, 35, 32, 31, 37, 43, 48, 47, 38, 71, 56, 53]},
    {'abbrev': 'jhn', 'number': 44, 'chapters': 21, 'verses': [
        51, 25, 36, 54, 47, 71, 53, 59, 41, 42, 57, 50, 38, 31, 27, 33, 26, 40, 42, 31, 25]},
    {'abbrev': 'act', 'number': 45, 'chapters': 28, 'verses': [
        26, 47, 26, 37, 42, 15, 60, 40, 43, 48, 30, 25, 52, 28, 41, 40, 34, 28, 41, 38, 40, 30, 35, 27, 27, 32, 44, 31]},
    {'abbrev': 'rom', 'number': 46, 'chapters': 16, 'verses': [
        32, 29, 31, 25, 21, 23, 25, 39, 33, 21, 36, 21, 14, 23, 33, 27]},
    {'abbrev': '1co', 'number': 47, 'chapters': 16, 'verses': [
        31, 16, 23, 21, 13, 20, 40, 13, 27, 33, 34, 31, 13, 40, 58, 24]},
    {'abbrev': '2co', 'number': 48, 'chapters': 13, 'verses': [
        24, 17, 18, 18, 21, 18, 16, 24, 15, 18, 33, 21, 14]},
    {'abbrev': 'gal', 'number': 49, 'chapters': 6,
        'verses': [24, 21, 29, 31, 26, 18]},
    {'abbrev': 'eph', 'number': 50, 'chapters': 6,
        'verses': [23, 22, 21, 32, 33, 24]},
    {'abbrev': 'php', 'number': 51, 'chapters': 4, 'verses': [30, 30, 21, 23]},
    {'abbrev': 'col', 'number': 52, 'chapters': 4, 'verses': [29, 23, 25, 18]},
    {'abbrev': '1th', 'number': 53, 'chapters': 5,
        'verses': [10, 20, 13, 18, 28]},
    {'abbrev': '2th', 'number': 54, 'chapters': 3, 'verses': [12, 17, 18]},
    {'abbrev': '1ti', 'number': 55, 'chapters': 6,
        'verses': [20, 15, 16, 16, 25, 21]},
    {'abbrev': '2ti', 'number': 56, 'chapters': 4, 'verses': [18, 26, 17, 22]},
    {'abbrev': 'tit', 'number': 57, 'chapters': 3, 'verses': [16, 15, 15]},
    {'abbrev': 'phm', 'number': 58, 'chapters': 1, 'verses': [25]},
    {'abbrev': 'heb', 'number': 59, 'chapters': 13, 'verses': [
        14, 18, 19, 16, 14, 20, 28, 13, 28, 39, 40, 29, 25]},
    {'abbrev': 'jas', 'number': 60, 'chapters': 5,
        'verses': [27, 26, 18, 17, 20]},
    {'abbrev': '1pe', 'number': 61, 'chapters': 5,
        'verses': [25, 25, 22, 19, 14]},
    {'abbrev': '2pe', 'number': 62, 'chapters': 3, 'verses': [21, 22, 18]},
    {'abbrev': '1jn', 'number': 63, 'chapters': 5,
        'verses': [10, 29, 24, 21, 21]},
    {'abbrev': '2jn', 'number': 64, 'chapters': 1, 'verses': [13]},
    {'abbrev': '3jn', 'number': 65, 'chapters': 1, 'verses': [15]},
    {'abbrev': 'jud', 'number': 66, 'chapters': 1, 'verses': [25]},
    {'abbrev': 'rev', 'number': 67, 'chapters': 22, 'verses': [
        20, 29, 22, 11, 14, 17, 17, 13, 21, 11, 19, 18, 18, 20, 8, 21, 18, 24, 21, 15, 27, 21]},
]

# Set up directories
home = Path.home()
input_dir = home / 'Desktop' / 'input'
anchor_dir = input_dir / 'Anchor'
back_dir = input_dir / 'Back'
heart_dir = input_dir / 'Heart'
output_dir = input_dir / 'Results'

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

for book in books:
    book_number = book['number']
    anchor_file = anchor_dir / f'{book_number}.usfm'
    back_file = back_dir / f'{book_number}.usfm'
    heart_file = heart_dir / f'{book_number}.usfm'

    if not anchor_file.exists() or not back_file.exists() or not heart_file.exists():
        print(
            f"Missing files for book number {book_number} ({book['abbrev']}), skipping.")
        continue

    print(f"Processing book: {book['abbrev']} ({book_number}.usfm)")
    anchor_verses = parse_usfm(anchor_file)
    back_verses = parse_usfm(back_file)
    heart_verses = parse_usfm(heart_file)

    output_file = output_dir / f'{book["abbrev"]}.csv'
    # Explicitly delete the output file if it exists to ensure overwriting
    if output_file.exists():
        try:
            output_file.unlink()
            print(f"Deleted existing file: {output_file}")
        except PermissionError:
            print(
                f"Error: Cannot delete {output_file}. File may be in use or lacks write permissions.")
            continue

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(['index', 'anchor', 'back', 'heart'])
        for chap in range(1, book['chapters'] + 1):
            max_verse = book['verses'][chap - 1]
            for vers in range(1, max_verse + 1):
                index = f"{book['number']:02d}{chap:03d}{vers:03d}"
                a_text = next(
                    (text for c, v, text in anchor_verses if c == chap and v == vers), '')
                b_text = next(
                    (text for c, v, text in back_verses if c == chap and v == vers), '')
                h_text = next(
                    (text for c, v, text in heart_verses if c == chap and v == vers), '')
                writer.writerow([index, a_text, b_text, h_text])

    print(f"Generated {output_file}")
