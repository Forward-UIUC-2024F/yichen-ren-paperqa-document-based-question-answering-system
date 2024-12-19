import os
import openai
import re
import glob
import shutil

openai.api_key = 'xxxx'

# can be modified for further fine-tune
MAX_TOKENS = 2048  
CHUNK_SIZE = 1500  
OUTPUT_DIR = 'segmented_texts'
SUMMARY_DIR = 'summaries'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Splits the input text into chunks of max_chunk_size tokens
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

# Saves each chunk to a separate file
def save_chunks_to_files(chunks):
    filenames = []
    for i, chunk in enumerate(chunks):
        filename = os.path.join(OUTPUT_DIR, f'chunk_{i + 1}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chunk)
        filenames.append(filename)
    return filenames

# Summarizes the input text using the OpenAI GPT model
def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Please summarize the following text."},
            {"role": "user", "content": text}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

# select content out of GPT responses
def identify_topic(text):
    """Identifies the topic of the input text using the OpenAI GPT model."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Please identify the main topic of the following text."},
            {"role": "user", "content": text}
        ],
        max_tokens=50
    )
    return response['choices'][0]['message']['content']


# Processes the chunks to identify topics and summarize them
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

# Generates a summary for the entire text based on the stored summaries
def generate_overall_summary():
    all_summaries = ''
    for file in glob.glob(os.path.join(SUMMARY_DIR, '*.txt')):
        with open(file, 'r') as f:
            all_summaries += f.read() + '\n\n'
    overall_summary = summarize_text(all_summaries)
    # Store the overall summary in a new file
    with open(os.path.join(SUMMARY_DIR, 'overall_summary.txt'), 'w') as f:
        f.write(overall_summary)
    return overall_summary

# pre-req for answering questions, judge to which section the query is relavant to
def is_relevant(question, text_chunk):
    try:
        # Use GPT to compare the relevance of the question to the text chunk
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Determine if the following question is relevant to the text, only return a single word yes if the question is relavant, only return a signle word no if the question is not relavant."},
                {"role": "user", "content": f"Text: {text_chunk}\n\nQuestion: {question}"}
            ],
            max_tokens=50
        )
        relevance = response['choices'][0]['message']['content'].strip().lower()
        return "yes" in relevance
    except Exception as e:
        print(f"Error determining relevance: {e}")
        return False

# Answers a question by finding the most relevant part of the text.
def answer_question(question):
    relevant_chunk = None
    for file in glob.glob(os.path.join(SUMMARY_DIR, '*.txt')):        
        with open(file, 'r') as f:
            chunk_text = f.read()
        if is_relevant(question, chunk_text):
            relevant_chunk = chunk_text
            break
    if relevant_chunk:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Please answer the question as detailed as possible based on the following text."},
                {"role": "user", "content": f"Text: {relevant_chunk}\n\nQuestion: {question}"}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content']
    return "No relevant section found to answer the question."

def delete_inprocess_files():
    dir1 = 'segmented_texts'
    dir2 = 'summaries'
    if os.path.exists(dir1) and os.path.isdir(dir1):
        shutil.rmtree(dir1)
        print(f"Directory '{dir1}' has been deleted.")
    if os.path.exists(dir2) and os.path.isdir(dir2):
        shutil.rmtree(dir2)
        print(f"Directory '{dir2}' has been deleted.")

def main():
    with open('example_paper.txt', 'r') as f:
        long_text = f.read()
    chunks = split_text_into_chunks(long_text, CHUNK_SIZE)
    chunk_files = save_chunks_to_files(chunks)
    process_chunks(chunk_files)
    overall_summary = generate_overall_summary()
    print("Overall summary has been generated and saved.")
    question = input("Please enter your question: ")
    answer = answer_question(question)
    print("Answer:", answer)
    with open('answer.txt', 'w') as f:
        f.write(answer)
    delete_inprocess_files()

if __name__ == "__main__":
    main()