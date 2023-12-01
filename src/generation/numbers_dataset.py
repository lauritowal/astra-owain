import random
import string
import pandas as pd
from pathlib import Path

"""
    Generates a dataset of random strings, where half of the examples contain at least one number
    and are labeled true, and the other half contain only letters and are labeled false.
"""

script_dir = Path(__file__).resolve().parent

NUM_EXAMPLES = 100

def random_string(length, with_number=False):
    characters = string.ascii_lowercase + ('0123456789' if with_number else '')
    return ''.join(random.choice(characters) for _ in range(length))

def generate_dataset(num_data_points=NUM_EXAMPLES):
    dataset = []
    for _ in range(num_data_points // 2):
        # Generate strings with a number (label True)
        dataset.append((random_string(random.randint(5, 15), with_number=True), True))
        # Generate strings without a number (label False)
        dataset.append((random_string(random.randint(5, 15), with_number=False), False))

    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Convert to DataFrame
    return pd.DataFrame(dataset, columns=['Input', 'Label'])

df = generate_dataset()

# Saving the DataFrame as a CSV file
csv_file = script_dir / '../../datasets/numbers.csv'
df.to_csv(csv_file, index=False)
