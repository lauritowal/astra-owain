import json
import os
from pprint import pprint
import pandas as pd
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
script_dir = Path(__file__).resolve().parent

NUM_EXAMPLES = 100

def classify_string(s):
    return s.islower()


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

# Initialize counter for correct classifications
correct = 0
failed_examples = []

# Classify each input in the dataset using the formatted prompt examples
for index, row in testset.iterrows():
    input_text = row['Input']
    full_prompt = f"{prompt_data['instructions']}\n{formatted_prompt_examples}\nInput: \"{input_text}\" Label: "

    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # use gpt4 for better results
            messages=[
                {"role": "system", "content": full_prompt} # TODO: probably instead of for-loop send all messages at once
            ]
        )

        # Extract and process the response
        predicted_label = response.choices[0].message.content
        predicted_label = True if predicted_label == "True" else False

        print("Input:", input_text)
        print("Predicted label:", predicted_label)
        print("Actual label:", classify_string(input_text))

        # Check if the prediction is correct
        correct += predicted_label == classify_string(input_text)
    except Exception as e:
        print(f"Failed to classify input: \"{input_text}\"")
        print(f"Error: {e}")
        # Log the failed example and the error message
        failed_examples.append({"input": input_text, "error": str(e)})


# Calculate accuracy
accuracy = correct / len(testset)
print(f"Accuracy: {accuracy}")