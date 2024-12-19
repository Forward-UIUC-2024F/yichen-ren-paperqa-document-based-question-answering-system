import os
import requests
import json
import re
import glob

# Constants
MAX_TOKENS = 2000   
CHUNK_SIZE = 2000
OUTPUT_DIR = 'segmented_texts_temp3'
SUMMARY_DIR = 'summaries_temp3'
ANSWER_DIR = 'pubmedqa_answer_Llama3_abstract'
CONTEXT_DIR = 'pubmedqa_context/TXT'
QUESTION_DIR = 'pubmedqa_question'
DATASET_FILE = 'PubMedQA_dataset.json'  # Path to your JSON dataset

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(ANSWER_DIR, exist_ok=True)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# def split_text_into_chunks(text, max_chunk_size):
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     chunks = []
#     current_chunk = ''
#     for sentence in sentences:
#         if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence
#         else:
#             current_chunk += ' ' + sentence
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     print("Finished splitting text into chunks.")
#     return chunks

# def save_chunks_to_files(chunks):
#     filenames = []
#     for i, chunk in enumerate(chunks):
#         filename = os.path.join(OUTPUT_DIR, f'chunk_{i + 1}.txt')
#         with open(filename, 'w', encoding='utf-8') as f:
#             f.write(chunk)
#         filenames.append(filename)
#     print("Finished saving chunks to files.")
#     return filenames

def call_llama_api(prompt):
    url = 'http://localhost:11434/api/generate'
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    data = json.dumps(payload)
    response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})

    if response.status_code == 200:
        response_test = response.text
        data = json.loads(response_test)
        return data.get("response", "")
    else:
        print("Error:", response.status_code, response.text)
        return ""

# def summarize_text(text):
#     prompt = f"Please summarize the following text: {text}"
#     result = call_llama_api(prompt)
#     print("Finished summarizing text.")
#     return result

# def identify_topic(text):
#     prompt = f"Please identify the main topic of the following text: {text}"
#     result = call_llama_api(prompt)
#     print("Finished identifying topic.")
#     return result

# def process_chunks(filenames):
#     summaries = []
#     for filename in filenames:
#         with open(filename, 'r') as f:
#             chunk = f.read()
#         topic = identify_topic(chunk)
#         summary = summarize_text(chunk)
#         summaries.append((filename, topic, summary))
#         summary_filename = os.path.join(SUMMARY_DIR, f'summary_{os.path.basename(filename)}')
#         with open(summary_filename, 'w') as f:
#             f.write(f"Topic: {topic}\n\nSummary: {summary}")
#     print("Finished processing chunks.")
#     return summaries

def answer_question(question, context):
    prompt = f"Please answer the question based on the following text: Text: {context} Question: {question}. Be as detailed and concrete as you can"
    result = call_llama_api(prompt)
    return result

def clear_directory(directory):
    files = glob.glob(os.path.join(directory, '*'))
    for f in files:
        os.remove(f)
    print(f"Finished clearing directory: {directory}")

def process_files():
    # Load JSON dataset
    with open(DATASET_FILE, 'r') as f:
        dataset = json.load(f)

    context_files = glob.glob(os.path.join(CONTEXT_DIR, '*.txt'))
    for context_file in context_files:
        context_base = os.path.basename(context_file)
        context_id = context_base.split('.')[0]
        
        # Extract question and context from JSON dataset
        if context_id not in dataset:
            print(f"No data found for context ID: {context_id}")
            continue

        question = dataset[context_id].get("QUESTION", "")
        context = dataset[context_id].get("CONTEXTS", "")
        answer = answer_question(question, context)
        answer_filename = os.path.join(ANSWER_DIR, f'{context_id}.txt')
        with open(answer_filename, 'w', encoding='utf-8') as f:
            f.write(answer)
        print('Finished answering')

        clear_directory(OUTPUT_DIR)
        clear_directory(SUMMARY_DIR)

def main():
    process_files()
    print("All questions have been processed and answers have been saved.")

if __name__ == "__main__":
    main()