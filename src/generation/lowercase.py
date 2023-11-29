import pandas as pd
import random
import string

"""
    Generates a dataset of random strings of lowercase letters and mixed case letters, 
    where half of the examples are labeled true and half are false.
"""
NUM_EXAMPLES = 100

# Function to generate a random string of lowercase letters
def generate_lower_string(length):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# Function to generate a random string with at least one uppercase letter
def generate_mixed_string(length):
    s = ''.join(random.choices(string.ascii_letters, k=length))
    while s.islower() or s.isupper():
        s = ''.join(random.choices(string.ascii_letters, k=length))
    return s

# Generating the dataset
data = []
for _ in range(NUM_EXAMPLES // 2):
    # Generate true examples
    data.append({"Input": generate_lower_string(random.randint(10, 30)), "Label": True})
    # Generate false examples
    data.append({"Input": generate_mixed_string(random.randint(10, 30)), "Label": False})

# Creating a DataFrame
df = pd.DataFrame(data)

# Saving the DataFrame as a CSV file
csv_file = '../datasets/lowercase.csv'
df.to_csv(csv_file, index=False)
