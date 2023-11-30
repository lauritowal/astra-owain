import io
import json
import os
import logging
import pandas as pd
from pathlib import Path
from pprint import pformat
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

script_dir = Path(__file__).resolve().parent

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = script_dir / f'../logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_dir / f"experiment_{timestamp}.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
testset_df = pd.read_csv(testset_path)[:NUM_EXAMPLES]

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

logging.info(f"JSON file path: {json_file_path}")
logging.info(f"CSV file path: {testset_path}")

instruction = f"{prompt_data['instructions']}\n{formatted_prompt_examples}\n"

testset = ""
for index, row in testset_df.iterrows():
    input_text = row['Input']
    testset += f"Input: \"{input_text}\" Label: \n"

try:
    # Prompt language model with instruction and testset
    full_prompt = [{"role": "system", "content": instruction + testset}]
    logging.info(f"full prompt: {pformat(full_prompt)}")

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4", #-1106-preview", # use gpt4 for better results
        messages=full_prompt
    )
    content = response.choices[0].message.content

    # Transform the string data into a format readable by pandas
    # Replace 'Input: ' with empty string and ' Label: ' with a comma
    formatted_content = content.replace('Input: "', '').replace('" Label: ', ',')
    csv_data = "Input,Label\n" + formatted_content

    # Use StringIO to convert the string data to a file-like object
    csv_file_like_object = io.StringIO(csv_data)

    # Read the CSV data from the file-like object
    csv_df = pd.read_csv(csv_file_like_object)

    # write CSV file to disk
    results_dir = script_dir / f'../results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(results_dir / f"experiment_{timestamp}.csv", index=False)

    logging.info(f"content of response: {pformat(csv_df)}")

    # Evaluate the results
    # Initialize counter for correct classifications
    correct = 0
    for index, row in csv_df.iterrows():
        input_text = row['Input']
        predicted_label = row['Label']
        ground_truth_label = testset_df.iloc[index]['Label']

        # predicted_label = True if predicted_label == "True" else False
        logging.info(f"Predicted label for {input_text}: {predicted_label}")
        logging.info(f"Actual label for {input_text}: {ground_truth_label}")

        correct += predicted_label == ground_truth_label

    # Calculate accuracy
    accuracy = correct / NUM_EXAMPLES
    logging.info(f"Experiment completed. Accuracy: {accuracy}")

except Exception as e:
    logging.warning(f"Failed full prompt: {pformat(full_prompt)}\n\n error: {pformat(str(e))}")


