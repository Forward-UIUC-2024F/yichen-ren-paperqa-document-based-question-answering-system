import fitz  # PyMuPDF
import torch
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration


# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    :param pdf_path: Path to the PDF file.
    :return: String containing the extracted text.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text


# Step 2: Chunk the text for indexing
def chunk_text(text, chunk_size=512):
    """
    Splits the text into smaller chunks.
    :param text: The full text to be chunked.
    :param chunk_size: Size of each chunk (in characters).
    :return: List of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Step 3: Index the text chunks using FAISS for retrieval
def build_faiss_index(text_chunks, tokenizer):
    """
    Encodes and indexes the text chunks using FAISS.
    :param text_chunks: List of text chunks.
    :param tokenizer: Hugging Face tokenizer for RAG.
    :return: FAISS index and tokenized chunks.
    """
    chunk_encodings = tokenizer(text_chunks, return_tensors="pt", padding=True, truncation=True)
    index = faiss.IndexFlatIP(chunk_encodings.input_ids.size(1))  # Inner product index
    index.add(chunk_encodings.input_ids.numpy())
    return index, chunk_encodings


# Step 4: Query the retriever and generate an answer
def generate_answer(question, model, tokenizer, retriever, index, text_chunks):
    """
    Generates an answer to a question using RAG with retrieved documents.
    :param question: The user's question.
    :param model: The RAG model.
    :param tokenizer: The RAG tokenizer.
    :param retriever: The RAG retriever.
    :param index: FAISS index of text chunks.
    :param text_chunks: Original text chunks.
    :return: Generated answer and retrieved document tokens.
    """
    # Tokenize the question
    question_inputs = tokenizer(question, return_tensors="pt")

    # Retrieve relevant documents
    retriever.index = index
    retriever.index_id_to_doc_id = {i: chunk for i, chunk in enumerate(text_chunks)}
    docs_dict = retriever(question, return_tensors="pt")

    # Generate an answer based on the retrieved documents
    with torch.no_grad():
        outputs = model.generate(
            input_ids=question_inputs["input_ids"],
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            output_attentions=True,  # Ensure attentions are returned
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decode the generated answer
    generated_answer_ids = outputs.sequences
    generated_answer = tokenizer.decode(generated_answer_ids[0], skip_special_tokens=True)

    return generated_answer, outputs, question_inputs, generated_answer_ids


# Step 5: Visualize Attention Map
def visualize_attention(outputs, tokenizer, question_inputs, generated_answer_ids):
    """
    Visualizes the attention map between the question tokens and generated answer tokens.
    :param outputs: Output from the model generation, including attention.
    :param tokenizer: Tokenizer used for decoding tokens.
    :param question_inputs: Tokenized question input.
    :param generated_answer_ids: Token IDs for the generated answer.
    """
    # Extract attention weights from the last decoder layer
    attentions = outputs.decoder_attentions[-1]  # List of attention maps from each layer
    attentions = attentions.mean(dim=1)  # Average across attention heads

    # Convert token IDs back to words for better visualization
    input_tokens = tokenizer.convert_ids_to_tokens(question_inputs['input_ids'][0])
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_answer_ids[0])

    # Visualize the attention map between the question and the generated answer
    attention_map = attentions[0].cpu().detach().numpy()

    # Plot the attention heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_map, xticklabels=input_tokens, yticklabels=generated_tokens, cmap="Blues", cbar=True)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel("Input Tokens (Question)")
    plt.ylabel("Generated Tokens")
    plt.title("Attention Map")
    plt.show()


# Main function to run the RAG-based PDF QA system
def main():
    # Load the RAG model and tokenizer
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-base",
        index_name="custom",  # We’ll create a custom index of the PDF text
        use_dummy_dataset=True  # Will be replaced with our data
    )
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", output_attentions=True)

    # Path to your PDF file
    pdf_path = "intelligent_scissor.pdf"

    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text (first 500 characters): {pdf_text[:500]}")

    # Step 2: Chunk the text for indexing
    text_chunks = chunk_text(pdf_text)
    print(f"Number of text chunks: {len(text_chunks)}")

    # Step 3: Build FAISS index
    index, chunk_encodings = build_faiss_index(text_chunks, tokenizer)

    # Step 4: Ask a question and generate an answer
    question = "how does the algorithm decide the weight of intelligent scissor?"
    generated_answer, outputs, question_inputs, generated_answer_ids = generate_answer(
        question, model, tokenizer, retriever, index, text_chunks
    )

    # Print the generated answer
    print(f"Generated Answer: {generated_answer}")

    # Step 5: Visualize the attention map
    visualize_attention(outputs, tokenizer, question_inputs, generated_answer_ids)


if __name__ == "__main__":
    main()