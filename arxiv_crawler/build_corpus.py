from datasets import Dataset
import pandas as pd
import numpy as np

def build_corpus(csv_file='data/merged_arxiv_papers.csv'):
    try:
        # load CSV file
        print("Loading CSV file...")
        df = pd.read_csv(csv_file)
        
        # Drop duplicates
        print("Generating numeric IDs...")
        num_rows = len(df)
        ids = [str(i).zfill(6) for i in range(num_rows)] 
        
        # Convert to Dataset format
        print("Converting to Dataset format...")
        dataset = Dataset.from_pandas(df)
        
        # Add ID column
        print("Adding ID column...")
        dataset = dataset.add_column("id", ids)
        
        # Reorder columns
        print("\nDataset Info:")
        print(f"Number of examples: {len(dataset)}")
        print("\nFeatures:", dataset.features)
        
        # Save to disk
        print("\nSaving dataset...")
        dataset.save_to_disk("./data/arxiv_corpus")
        
        # Print sample entries
        print("\nSample entries:")
        for i in range(min(100, len(dataset))):
            print(f"\nSample {i+1}:")
            print(f"ID: {dataset[i]['id']}")
            print(f"Title: {dataset[i]['title']}")
            print(f"Published Date: {dataset[i]['published_date']}")
        
        return dataset
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    dataset = build_corpus()
