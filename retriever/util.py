import csv
import numpy as np


def normalize(embeddings):
    """Normalize embeddings to unit length."""
    norms = np.sqrt(np.sum(embeddings**2, axis=-1, keepdims=True))
    return embeddings / np.maximum(norms, 1e-10)


def load_tsv_to_dict(filepath, header=False):
    """Load TSV file into dictionary."""
    result = {}
    with open(filepath, 'r', encoding='utf8') as fIn:
        reader = csv.reader(fIn, delimiter='\t')
        if header:
            next(reader)  # Skip header row
        for row in reader:
            if len(row) >= 2:
                result[row[0]] = int(row[1])
    return result


def save_dict_to_tsv(dic, filepath, keys=None):
    """Save dictionary to TSV file."""
    with open(filepath, 'w', encoding='utf8') as fOut:
        if keys is not None:
            fOut.write(f"{keys[0]}\t{keys[1]}\n")
        for key, value in dic.items():
            fOut.write(f"{key}\t{value}\n")
