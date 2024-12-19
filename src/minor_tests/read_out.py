# import json
# import os

# with open('squad_dataset.json', 'r') as file:
#     data = json.load(file)
# os.makedirs('squad_data', exist_ok=True)
# for index, item in enumerate(data['data']):
#     file_name = f'squad_data/context_{index}.txt'
#     with open(file_name, 'w') as context_file:
#         for paragraph in item['paragraphs']:
#             context = paragraph['context']
#             context_file.wri0te(context + '\n\n')




# import json
# import os

# # Load your JSON data
# with open('squad_dataset.json', 'r') as file:
#     data = json.load(file)

# # Create directory if it doesn't exist
# os.makedirs('squad_question', exist_ok=True)

# # Store each question in a separate file
# for index, item in enumerate(data['data']):
#     for para_index, paragraph in enumerate(item['paragraphs']):
#         for qa_index, qa in enumerate(paragraph['qas']):
#             question = qa['question']
#             # Define the file name
#             file_name = f'squad_question/question_{index}_{para_index}_{qa_index}.txt'
#             # Write the question to a file
#             with open(file_name, 'w') as question_file:
#                 question_file.write(question)




import json

# Load the JSON data from a file
with open('PubMedQA_dataset.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Loop through each entry in the JSON data
for key, value in data.items():
    question = value.get("LONG_ANSWER", "")
    
    # Write the question to a file named after the key
    with open(f"pubmedqa_answer/{key}.txt", "w", encoding='utf-8') as file:
        file.write(question)