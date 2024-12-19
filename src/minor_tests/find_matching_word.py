import os

def highlight_matching_words_html(text_c, text_d):
    words_c = set(text_c.split())
    words_d = text_d.split()
    
    highlighted_text = [
        f"<span style='background-color: yellow;'>{word}</span>" if word in words_c else word 
        for word in words_d
    ]
    
    return ' '.join(highlighted_text)

def process_directories(dir_a, dir_b, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(dir_a):
        file_c_path = os.path.join(dir_a, filename)
        file_d_path = os.path.join(dir_b, filename)
        
        if os.path.isfile(file_c_path) and os.path.isfile(file_d_path):
            with open(file_c_path, 'r', encoding='utf-8') as file_c:
                text_c = file_c.read()
            with open(file_d_path, 'r', encoding='utf-8') as file_d:
                text_d = file_d.read()

            highlighted_text = highlight_matching_words_html(text_c, text_d)
            output_path = os.path.join(output_dir, filename + '.html')
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(f"<html><body>{highlighted_text}</body></html>")

# Example usage
dir_a = 'pubmedqa_answer_Llama3_1000_dense_path_retrieval'
dir_b = 'pubmedqa_answer'
output_dir = 'path/to/output/directory'

process_directories(dir_a, dir_b, output_dir)