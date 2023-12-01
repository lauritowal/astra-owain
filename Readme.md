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

