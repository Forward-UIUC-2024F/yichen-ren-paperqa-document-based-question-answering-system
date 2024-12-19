import os
import requests
import json
import re
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer, util

# Constants
MAX_TOKENS = 1000  
CHUNK_SIZE = 1000
OUTPUT_DIR = 'segmented_texts'
SUMMARY_DIR = 'summaries'
ANSWER_DIR = 'selfmade_answer_Llama3_1000_dense_path_retrieval'
CONTEXT_DIR = 'selfmade_context'
QUESTION_DIR = 'selfmade_question'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(ANSWER_DIR, exist_ok=True)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_text_into_chunks(text, max_chunk_size):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += ' ' + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    print("Finished splitting text into chunks.")
    return chunks

def save_chunks_to_files(chunks):
    filenames = []
    for i, chunk in enumerate(chunks):
        filename = os.path.join(OUTPUT_DIR, f'chunk_{i + 1}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chunk)
        filenames.append(filename)
    print("Finished saving chunks to files.")
    return filenames

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
        return data["response"]
    else:
        print("Error:", response.status_code, response.text)

def summarize_text(text):
    # not in use anymore
    prompt = f"Please summarize the following text: {text}"
    result = call_llama_api(prompt)
    print("Finished summarizing text.")
    return result
    return

def identify_topic(text):
    # not in use anymore
    prompt = f"Please identify the main topic of the following text: {text}"
    result = call_llama_api(prompt)
    print("Finished identifying topic.")
    return result
    return

def process_chunks(filenames):
    summaries = []
    for filename in filenames:
        with open(filename, 'r') as f:
            chunk = f.read()
        topic = identify_topic(chunk)
        summary = summarize_text(chunk)
        summaries.append((filename, topic, summary))
        summary_filename = os.path.join(SUMMARY_DIR, f'summary_{os.path.basename(filename)}')
        with open(summary_filename, 'w') as f:
            f.write(f"Topic: {topic}\n\nSummary: {summary}")
    print("Finished processing chunks.")
    return summaries

def answer_question(question, context):
    prompt = f"Please answer the question: {question}, based on the following text: Text: {context}. Use as much original context as possbile"
    result = call_llama_api(prompt)
    return result

def clear_directory(directory):
    files = glob.glob(os.path.join(directory, '*'))
    for f in files:
        os.remove(f)
    print(f"Finished clearing directory: {directory}")

def calculate_similarity(question, chunks):  # dense path retrieval
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embedding = model.encode(chunks, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(question_embedding, chunk_embedding)
    return similarity.item()

def process_files():    
    context_files = glob.glob(os.path.join(CONTEXT_DIR, '*.txt'))
    for context_file in context_files:
        context_base = os.path.basename(context_file)
        context_number = context_base.split('.')[0]
        
        context = read_file(context_file)
        chunks = split_text_into_chunks(context, CHUNK_SIZE)
        chunk_filenames = save_chunks_to_files(chunks)
        # summaries = process_chunks(chunk_filenames)

        question_files = glob.glob(os.path.join(QUESTION_DIR, f'{context_number}*.txt'))
        
        if not question_files:
            print(f"No questions found for context: {context_number}")
            continue

        for question_file in question_files:
            question = read_file(question_file)
            max_similarity = 0
            max_similarity_index = 0
            similarity = 0
            for i in range(len(chunk_filenames)):
                similarity = calculate_similarity(question, read_file(chunk_filenames[i]))
                if max_similarity < similarity:
                    max_similarity = similarity
                    max_similarity_index = i
            # most_similar_chunk = calculate_similarity(question, [read_file(f) for f in chunk_filenames])
            # answer = answer_question(question, most_similar_chunk)
            answer = answer_question(question, read_file(chunk_filenames[max_similarity_index]))

            answer_filename = os.path.join(ANSWER_DIR, os.path.basename(question_file))
            with open(answer_filename, 'w', encoding='utf-8') as f:
                f.write(f"the chosen chunk is {max_similarity_index + 1}")
                f.write('\n')
                f.write(answer)
                print('Finished answering')
        
        clear_directory(OUTPUT_DIR)
        clear_directory(SUMMARY_DIR)

def main():
    process_files()
    print("All questions have been processed and answers have been saved.")

if __name__ == "__main__":
    main()