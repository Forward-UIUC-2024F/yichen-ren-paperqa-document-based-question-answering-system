from PyPDF2 import PdfReader
import os

pdf_folder = '../pubmedqa_context'
output_folder = '../pubmedqa_context_txt'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        try:
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            
            text_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_file.write(text + '\n')

            print(f"Text from '{filename}' has been extracted and saved to '{text_file_path}'")
        except Exception as e:
            print(e)