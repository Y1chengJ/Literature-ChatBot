import pandas as pd
import glob
import json
from collections import Counter
from datetime import datetime
import os

def analyze_arxiv_files():
    # Find all CSV and JSON files with 'arxiv_papers' prefix
    csv_files = glob.glob('*.csv')
    json_files = glob.glob('*.json')
    
    total_stats = {
        'total_papers': 0,
        'unique_papers': set(),
        'papers_by_category': Counter(),
        'papers_by_date': Counter(),
        'files_analyzed': [],
        'file_sizes': {}
    }
    
    # Analyze CSV files
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            file_stats = analyze_file(file, df)
            update_total_stats(total_stats, file_stats)
        except Exception as e:
            print(f"Error processing CSV file {file}: {str(e)}")

    # Analyze JSON files
    for file in json_files:
        try:
            df = pd.read_json(file, lines=True)
            file_stats = analyze_file(file, df)
            update_total_stats(total_stats, file_stats)
        except Exception as e:
            print(f"Error processing JSON file {file}: {str(e)}")

    print_analysis(total_stats)

def analyze_file(filename, df):
    file_stats = {
        'filename': filename,
        'paper_count': len(df),
        'unique_papers': set(df['arxiv_id'] if 'arxiv_id' in df.columns else []),
        'file_size': os.path.getsize(filename),
        'papers_by_category': Counter(),
        'papers_by_date': Counter()
    }
    
    # Analyze categories if available
    if 'categories' in df.columns:
        for categories in df['categories']:
            if isinstance(categories, str):
                # Handle string representation of list
                try:
                    cat_list = eval(categories)
                    for cat in cat_list:
                        file_stats['papers_by_category'][cat] += 1
                except:
                    file_stats['papers_by_category'][categories] += 1
            elif isinstance(categories, list):
                for cat in categories:
                    file_stats['papers_by_category'][cat] += 1

    # Analyze dates if available
    if 'published' in df.columns:
        df['published'] = pd.to_datetime(df['published'])
        date_counts = df['published'].dt.date.value_counts()
        file_stats['papers_by_date'].update(date_counts.to_dict())

    return file_stats

def update_total_stats(total_stats, file_stats):
    total_stats['total_papers'] += file_stats['paper_count']
    total_stats['unique_papers'].update(file_stats['unique_papers'])
    total_stats['papers_by_category'].update(file_stats['papers_by_category'])
    total_stats['papers_by_date'].update(file_stats['papers_by_date'])
    total_stats['files_analyzed'].append(file_stats['filename'])
    total_stats['file_sizes'][file_stats['filename']] = file_stats['file_size']

def print_analysis(stats):
    print("\n=== ArXiv Crawler Analysis ===")
    print(f"\nTotal files analyzed: {len(stats['files_analyzed'])}")
    print(f"Total papers found: {stats['total_papers']}")
    print(f"Unique papers: {len(stats['unique_papers'])}")
    
    print("\nTop 5 categories:")
    for category, count in stats['papers_by_category'].most_common(5):
        print(f"  {category}: {count} papers")
    
    print("\nPapers by date (top 5 dates):")
    for date, count in sorted(stats['papers_by_date'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {date}: {count} papers")
    
    print("\nFile sizes:")
    total_size = 0
    for filename, size in stats['file_sizes'].items():
        size_mb = size / (1024 * 1024)  # Convert to MB
        total_size += size_mb
        print(f"  {filename}: {size_mb:.2f} MB")
    print(f"\nTotal size of all files: {total_size:.2f} MB")

if __name__ == "__main__":
    analyze_arxiv_files()