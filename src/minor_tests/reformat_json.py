import json

# Load your JSON data
with open('squad_dataset.json', 'r') as file:
    data = json.load(file)

# Write the formatted JSON to a new file
with open('squad_dataset_reformat.json', 'w') as file:
    json.dump(data, file, indent=4)