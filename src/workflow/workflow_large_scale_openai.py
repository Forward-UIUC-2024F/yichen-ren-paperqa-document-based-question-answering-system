import os
import openai
import re
import glob
import time
import json
import requests

openai.api_key = 'xxxx'

# Constants
MAX_TOKENS = 2048
CHUNK_SIZE = 1500
OUTPUT_DIR = 'segmented_texts'
SUMMARY_DIR = 'summaries'
ANSWER_DIR = 'squad_answer_GPT3.5-Turbo'
CONTEXT_DIR = 'squad_context'
QUESTION_DIR = 'squad_question'

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
    return chunks

def save_chunks_to_files(chunks):
    filenames = []
    for i, chunk in enumerate(chunks):
        filename = os.path.join(OUTPUT_DIR, f'chunk_{i + 1}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chunk)
        filenames.append(filename)
    return filenames

def call_openai_api(messages, max_tokens):
    retries = 5
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']
        except openai.error.RateLimitError:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Please try again later.")
                return None

def summarize_text(text):
    messages_for_openai = [
        {"role": "system", "content": "Please summarize the following text."},
        {"role": "user", "content": text}
    ]
    return call_openai_api(messages_for_openai, 150)

def identify_topic(text):
    messages = [
        {"role": "system", "content": "Please identify the main topic of the following text."},
        {"role": "user", "content": text}
    ]
    return call_openai_api(messages, 50)

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
    return summaries

def is_relevant(question, text_chunk):
    messages = [
        {"role": "system", "content": "Determine if the following question is relevant to the text, only return a single word yes if the question is relevant, only return a single word no if the question is not relevant."},
        {"role": "user", "content": f"Text: {text_chunk}\n\nQuestion: {question}"}
    ]
    relevance = call_openai_api(messages, 50)
    return "yes" in relevance if relevance else False

def answer_question(question, context):
    messages_for_openai = [
        {"role": "system", "content": "Please answer the question using the least word and being as neat as possible, based on the following text."},
        {"role": "user", "content": f"Text: {context}\n\nQuestion: {question}"}
    ]
    return call_openai_api(messages_for_openai, 200)

def process_files():
    context_files = glob.glob(os.path.join(CONTEXT_DIR, 'context_*.txt'))
    for context_file in context_files:
        context_index = os.path.basename(context_file).split('_')[1].split('.')[0]
        context = read_file(context_file)
        
        question_files = glob.glob(os.path.join(QUESTION_DIR, f'question_{context_index}_*.txt'))
        for question_file in question_files:
            question = read_file(question_file)
            # context = ''  # comment this line if using for given-question given-answer situations
            answer = answer_question(question, context)
            
            if answer:
                answer_filename = os.path.join(ANSWER_DIR, os.path.basename(question_file))
                print(answer_filename, "has been answered")
                with open(answer_filename, 'w', encoding='utf-8') as f:
                    f.write(answer)

def main():
    process_files()
    print("All questions have been processed and answers have been saved.")

if __name__ == "__main__":
    main()