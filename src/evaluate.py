from datetime import datetime
import json
import os
import logging
import pandas as pd

from pathlib import Path
from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

script_dir = Path(__file__).resolve().parent

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile_path = script_dir / f'../logs/experiment_{timestamp}.log'
logging.basicConfig(filename=logfile_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_EXAMPLES = 100

# Load the JSON file containing prompt examples
json_file_path =  script_dir / '../in_context_learning/lowercase.json'
with open(json_file_path, 'r') as file:
    prompt_data = json.load(file)

# Extracting and formatting the prompt examples
formatted_prompt_examples = "\n".join(
    f"Input: \"{example['Input']}\" Label: {example['Label']}" for example in prompt_data['examples']
)

# Load the dataset
testset_path = script_dir / '../datasets/lowercase.csv'
testset = pd.read_csv(testset_path)[:NUM_EXAMPLES]

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

logging.info(f"JSON file path: {json_file_path}")
logging.info(f"CSV file path: {testset_path}")

# Initialize counter for correct classifications
correct = 0
failed_examples = []

# Classify each input in the dataset using the formatted prompt examples
for index, row in testset.iterrows():
    input_text = row['Input']
    label = row['Label']
    full_prompt = f"{prompt_data['instructions']}\n{formatted_prompt_examples}\nInput: \"{input_text}\" Label: "

    logging.info(f"full_prompt: {full_prompt}")

    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4-1106-preview", # use gpt4 for better results
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

        # Extract and process the response
        predicted_label = response.choices[0].message.content
        predicted_label = True if predicted_label == "True" else False
            
        logging.info(f"Predicted label for {input_text}: {predicted_label}")
        logging.info(f"Actual label for {input_text}: {label}")

        # Check if the prediction is correct
        correct += predicted_label == label
    except Exception as e:
        logging.warning(f"Failed to classify input '{input_text}': {e}")

        # Log the failed example and the error message
        failed_examples.append({"input": input_text, "error": str(e)})

# Calculate accuracy
accuracy = correct / len(testset)
logging.info(f"Experiment completed. Accuracy: {accuracy}")
