import io
import ast
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

NUM_EXAMPLES = 100
ARTICULATION_PROMPT = "After that, can you articulate the pattern you have found and used for classification of the inputs in one short sentence? Separate the last sentence from the rest with \n\n"
TASK = "compliments" 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = script_dir / '../logs' / TASK
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_dir / f"experiment_{TASK}_{timestamp}.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the JSON file containing prompt examples
json_file_path =  script_dir / f'../in_context_learning/{TASK}.json'
with open(json_file_path, 'r') as file:
    prompt_data = json.load(file)

# Extracting and formatting the prompt examples
formatted_prompt_examples = "\n".join(
    f"Input: \"{example['Input']}\" Label: {example['Label']}" for example in prompt_data['examples']
)

# Load the dataset
testset_path = script_dir / f'../datasets/{TASK}.csv'
testset_df = pd.read_csv(testset_path)[:NUM_EXAMPLES]

# shuffle the dataset
testset_df = testset_df.sample(frac=1).reset_index(drop=True)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

logging.info(f"JSON file path: {json_file_path}")
logging.info(f"CSV file path: {testset_path}")

instruction = f"{prompt_data['instructions']}\n{formatted_prompt_examples}\n"

testset = ""
for index, row in testset_df.iterrows():
    input_text = row['Input']
    testset += f"Input: \"{input_text}\" Label: \n"

# Prompt language model with instruction and testset
full_prompt = [
    {"role": "system", "content": instruction + " " + ARTICULATION_PROMPT + testset},
]
logging.info(f"full prompt: {pformat(full_prompt)}")

try:
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=full_prompt
    )
except Exception as e:
    logging.warning(f"Failed full prompt: {pformat(full_prompt)}\n\n error: {pformat(str(e))}")

content = response.choices[0].message.content

# Transform the string data into a format readable by pandas
# Replace 'Input: ' with empty string and ' Label: ' with a comma
content = content.replace('Input: ', '').replace('Label: ', '')
content = content.split('\n\n')
articulation = content[-1] # last sentence is the articulation

csv_data = "Input,Label\n" + content[0]
predictions = pd.read_csv(io.StringIO(csv_data))

logging.info(f"content of response: {pformat(predictions)}")

# Evaluate the results
# Initialize counter for correct classifications
correct = 0
for index, row in predictions.iterrows():
    input_text = row['Input']
    
    predicted_label = row['Label']
    if type(predicted_label) == str:
        predicted_label = ast.literal_eval(predicted_label.strip())

    ground_truth_label = testset_df.iloc[index]['Label']

    logging.info(f"Correctly classified {input_text}? {predicted_label == ground_truth_label}")
    if ground_truth_label == predicted_label:
        correct += 1
        logging.info(f"Num correctly classified: {correct}")
    else:
        logging.info(f"predicted_label: {predicted_label}; ground_truth_label: {ground_truth_label}")

# Calculate accuracy
accuracy = correct / NUM_EXAMPLES
logging.info(f"Experiment completed. Accuracy: {accuracy}")

logging.info(f"Articulation {articulation}")

# Save results
results_dir = script_dir / '../results' / TASK
results_dir.mkdir(parents=True, exist_ok=True)
results_path = results_dir / f"experiment_{TASK}_{timestamp}.json"

results = {
    "examples": predictions.to_dict(orient='records'), 
    "accuracy": accuracy, 
    "articulation": articulation
}
with open(results_path, 'w') as file:
    json.dump(results, file, indent=4)