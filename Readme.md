# Create env and install packages
```
conda create -n astra python=3.12
conda activate astra
python -m pip install -r requirements.txt
```

# Create .env file and add API_KEY
OPENAI_API_KEY="<Your API KEY>"

# Generate datasets
The folder contains scripts to generate datasets, which can be used for classification. 
The generated datasets will land under `datasets/<TASK>`. Example:
```
python src/generation/lovercase_dataset.py
```

## Reverted labels
I used GPT-4 to generate additional datasets by reverting the labels of already generated datasets and the labels of the examples for the few-shot-prompting e.g. lowercase -> uppercase


# Set task in src/evaluate.py 
available tasks:
1. compliments
2. lowercase
3. uppercase
4. numbers
5. no_numbers
e.g.
```
TASK = "compliments" 
```

# Run src/evaluate.py
`python src/evaluate.py`

# Checkout the results in the results folder
