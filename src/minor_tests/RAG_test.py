import os
import faiss
import spacy
import fitz
import nltk
nltk.download('punkt')
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def split_into_chunks(doc, chunk_size=100):
    """
    Splits a document into manageable chunks.
    Args:
        doc (str): The document to split.
        chunk_size (int): Number of characters per chunk.
    Returns:
        list: List of document chunks.
    """
    return [doc[i:i + chunk_size] for i in range(0, len(doc), chunk_size)]

# Load spaCy's pre-trained model for NER (Named Entity Recognition)
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    """
    Extracts named entities from the text using spaCy.
    Args:
        text (str): The text to extract entities from.
    Returns:
        list: List of tuples containing entity text and entity label.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_relations(text):
    """
    Extracts relations between entities using spaCy's dependency parsing.
    Args:
        text (str): The text to extract relations from.
    Returns:
        list: List of tuples containing (subject, relation, object).
    """
    doc = nlp(text)
    relations = []

    for sent in doc.sents:
        subject = None
        relation = None
        obj = None

        for token in sent:
            # Find a subject (nsubj or similar)
            if 'subj' in token.dep_:
                subject = token.text
            
            # Find a verb or action (acts as relation)
            if token.pos_ == 'VERB':
                relation = token.text
            
            # Find an object (dobj or similar)
            if 'obj' in token.dep_:
                obj = token.text

        # If we found a valid subject-relation-object triple, store it
        if subject and relation and obj:
            relations.append((subject, relation, obj))

    return relations

def remove_duplicates(entities, relations):
    """
    Removes duplicate entities and relations from the lists.
    Args:
        entities (list): List of entity tuples.
        relations (list): List of relation tuples.
    Returns:
        tuple: A tuple containing unique entities and unique relations.
    """
    unique_entities = list(set(entities))  # Removing duplicate entities
    unique_relations = list(set(relations))  # Removing duplicate relations
    
    return unique_entities, unique_relations

# Load a pre-trained model for embedding generation.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    """
    Generates an embedding for the given text using Sentence Transformers.
    Args:
        text (str): The text to embed.
    Returns:
        np.array: Embedding vector.
    """
    return embedding_model.encode(text)

# Initialize FAISS index
dimension = 384  # The embedding dimension for all-MiniLM-L6-v2
faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

# Store entities and relations in a separate dictionary
entity_db = {}
relation_db = {}

def store_in_vector_db(text_chunks, entities, relations):
    """
    Stores the embeddings of text chunks, entities, and relations into FAISS for
    fast retrieval.
    Args:
        text_chunks (list of str): List of text chunks to embed and store.
        entities (list of tuples): List of entities to embed and store.
        relations (list of tuples): List of relations to embed and store.
    """
    # Store embeddings for text chunks
    for chunk in text_chunks:
        embedding = generate_embedding(chunk)
        faiss_index.add(embedding.reshape(1, -1))  # Add to FAISS index
    
    # Store embeddings for entities
    for entity, _ in entities:
        entity_embedding = generate_embedding(entity)
        entity_db[entity] = entity_embedding
    
    # Store embeddings for relations
    for source, relation, target in relations:
        relation_embedding = generate_embedding(f"{source} {relation} {target}")
        relation_db[(source, target)] = relation_embedding

def retrieve_text_chunks(query, k=5):
    """
    Retrieves the top-k most relevant text chunks based on the query embedding.
    Args:
        query (str): The user's query.
        k (int): Number of top results to retrieve.
    Returns:
        list: Indices of the top-k most similar text chunks.
    """
    query_embedding = generate_embedding(query)
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), k)
    return indices.flatten(), distances.flatten()

def retrieve_entities_and_relations(relevant_chunks):
    """
    Retrieves entities and relations related to the relevant text chunks.
    Args:
        relevant_chunks (list): List of relevant text chunks.
    Returns:
        list: List of relevant entities and relations.
    """
    relevant_entities = []
    relevant_relations = []
    
    # Extract entities and relations from the relevant chunks
    for chunk in relevant_chunks:
        entities = extract_entities(chunk)
        relations = extract_relations(chunk)
        relevant_entities.extend(entities)
        relevant_relations.extend(relations)
    
    return relevant_entities, relevant_relations

# Load GPT-2 model and tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

def format_entities_and_relations(entities, relations):
    """
    Formats the entities and relations into a readable string for inclusion in the prompt.
    Args:
        entities (list): List of entities.
        relations (list): List of relations.
    Returns:
        str: A formatted string containing the entities and relations.
    """
    entity_str = "Entities: " + ", ".join([f"{ent[0]} ({ent[1]})" for ent in entities])
    relation_str = "Relations: " + ", ".join([f"{rel[0]} {rel[1]} {rel[2]}" for rel in relations])
    
    return entity_str + "\n" + relation_str

def generate_response(prompt):
    """
    Generates a response using GPT-2 based on the provided prompt.
    Args:
        prompt (str): The prompt to input into GPT-2.
    Returns:
        str: The generated response.
    """
    inputs = gpt_tokenizer(prompt, return_tensors="pt")
    outputs = gpt_model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response_with_entities_and_relations(query, relevant_chunks, entities, relations):
    """
    Generates a response based on the query, relevant chunks, entities, and relations.
    Args:
        query (str): The user's query.
        relevant_chunks (list): List of relevant text chunks.
        entities (list): List of relevant entities.
        relations (list): List of relevant relations.
    Returns:
        str: The generated response.
    """
    # Combine relevant text chunks as context
    context = " ".join(relevant_chunks)
    
    # Format entities and relations for the prompt
    entity_relation_info = format_entities_and_relations(entities, relations)
    
    # Create the full prompt including context, entities, and relations
    prompt = (
        f"Answer the query '{query}' based on the following context:\n{context}\n\n"
        f"{entity_relation_info}"
    )
    
    # Generate response using GPT-2
    response = generate_response(prompt)
    return response

def extract_text_from_pdf(pdf_path):
    """
    Extracts the text from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    # Open the PDF file
    doc = fitz.open(pdf_path)
    text = ""
    
    # Iterate through each page and extract text
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load page
        text += page.get_text("text")  # Extract text from the page
    
    doc.close()  # Close the document
    return text

def rag_workflow_from_pdf(pdf_path, query):
    """
    Full RAG workflow that extracts text from a PDF, indexes the content,
    retrieves relevant chunks based on a query, and generates a response
    considering entities and relations.
    Args:
        pdf_path (str): The path to the PDF file.
        query (str): The user's query.
    Returns:
        str: The generated response.
    """
    # Step 1: Extract text from the PDF
    document_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Split the document text into chunks for indexing
    text_chunks = split_into_chunks(document_text)
    
    all_entities = []
    all_relations = []

    # Indexing Phase
    for chunk in text_chunks:
        # Extract entities and relations from each chunk
        entities = extract_entities(chunk)
        relations = extract_relations(chunk)
        
        # Remove duplicates
        unique_entities, unique_relations = remove_duplicates(entities, relations)
        
        # Store chunks, entities, and relations in the vector db
        store_in_vector_db([chunk], unique_entities, unique_relations)
        
        # Collect all entities and relations for retrieval
        all_entities.extend(unique_entities)
        all_relations.extend(unique_relations)
    
    # Retrieval Phase
    indices, _ = retrieve_text_chunks(query)
    relevant_chunks = [text_chunks[i] for i in indices if i < len(text_chunks)]
    
    # Retrieve entities and relations
    relevant_entities, relevant_relations = retrieve_entities_and_relations(relevant_chunks)
    
    # Generation Phase: Generate a response based on the retrieved context, entities, and relations
    response = generate_response_with_entities_and_relations(query, relevant_chunks, relevant_entities, relevant_relations)
    
    return response


def ask_openai(question):
    # Your OpenAI API key
    API_KEY = os.getenv('OPENAI_API_KEY')
    
    # API endpoint for OpenAI (GPT-3.5 or other versions can be specified in the model parameter)
    url = "https://api.openai.com/v1/chat/completions"

    # Headers and parameters for the POST request
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Data payload with the messages and other parameters, like temperature and max tokens
    data = {
        "model": "gpt-3.5-turbo",  # Choose model according to your API plan and requirements
        "messages": [{"role": "user", "content": f"Q: {question}\nA:"}],
        "temperature": 0.5,
        "max_tokens": 150
    }

    # Making the POST request to the OpenAI API
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON and extract the text
        response_data = response.json()
        answer = response_data['choices'][0]['message']['content'].strip()
        return answer
    else:
        # Print error if something went wrong
        print("Failed to fetch response: ", response.status_code)
        print(response.text)
        return None



#################### Main Workflow ####################

pdf_path = "intelligent_scissor.pdf"
query = "What potential problems might be encountered when an object, originally from a different background color, is composited into a new scene?"


try:
    # Run the RAG workflow with the PDF file
    output = rag_workflow_from_pdf(pdf_path, query)
    response = ask_openai(output)
    print("intermediate step is:", output)
    print("response is:", response)
    output = rag_workflow_from_pdf(pdf_path, query)# Run the RAG workflow with the PDF file    return merged_chunks        merged_chunks.append(current_chunk.strip())
    def auto_merge_chunks(text_chunks, query, max_token_length=1024):
        """
        Automatically merges chunks based on relevance to the query.
        Args:
            text_chunks (list of str): List of text chunks.
            query (str): The user's query.
            max_token_length (int): Maximum token length for the LLM.
        Returns:
            list: List of merged chunks.
        """
        def is_relevant(chunk, query):
            # Check relevance of a chunk to the query
            chunk_embedding = generate_embedding(chunk)
            query_embedding = generate_embedding(query)
            similarity = np.dot(chunk_embedding, query_embedding) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding))
            return similarity > 0.5  # Threshold for relevance

        def merge_chunks(chunks):
            # Merge chunks if their combined length is within the max token length
            merged_chunks = []
            current_chunk = ""
            for chunk in chunks:
                if len(gpt_tokenizer.encode(current_chunk + " " + chunk)) <= max_token_length:
                    current_chunk += " " + chunk
                else:
                    merged_chunks.append(current_chunk.strip())
                    current_chunk = chunk
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
            return merged_chunks

        # Step 1: Check relevance of each chunk
        relevant_chunks = [chunk for chunk in text_chunks if is_relevant(chunk, query)]

        # Step 2: Merge relevant chunks
        merged_chunks = merge_chunks(relevant_chunks)

        return merged_chunks

    # Use the auto_merge_chunks function in the RAG workflow
    def rag_workflow_from_pdf_with_auto_merge(pdf_path, query):
        document_text = extract_text_from_pdf(pdf_path)
        text_chunks = split_into_chunks(document_text)
        
        all_entities = []
        all_relations = []

        for chunk in text_chunks:
            entities = extract_entities(chunk)
            relations = extract_relations(chunk)
            unique_entities, unique_relations = remove_duplicates(entities, relations)
            store_in_vector_db([chunk], unique_entities, unique_relations)
            all_entities.extend(unique_entities)
            all_relations.extend(unique_relations)
        
        merged_chunks = auto_merge_chunks(text_chunks, query)
        relevant_entities, relevant_relations = retrieve_entities_and_relations(merged_chunks)
        response = generate_response_with_entities_and_relations(query, merged_chunks, relevant_entities, relevant_relations)
        
        return response

    # Run the updated RAG workflow with auto merging
    output = rag_workflow_from_pdf_with_auto_merge(pdf_path, query)
    response = ask_openai(output)
    print("final merged sentence is:", output)
    print("response is:", response)
    def split_into_paragraphs(doc):
        """
        Splits a document into paragraphs.
        Args:
            doc (str): The document to split.
        Returns:
            list: List of paragraphs.
        """
        return doc.split('\n\n')

    def split_into_sentences(paragraph):
        """
        Splits a paragraph into sentences.
        Args:
            paragraph (str): The paragraph to split.
        Returns:
            list: List of sentences.
        """
        return sent_tokenize(paragraph)

    def split_into_word_clusters(sentence, num_clusters=3):
        """
        Splits a sentence into clusters of words with similar semantic meaning.
        Args:
            sentence (str): The sentence to split.
            num_clusters (int): Number of clusters to form.
        Returns:
            list: List of word clusters.
        """
        words = word_tokenize(sentence)
        vectorizer = TfidfVectorizer()
        if len(words) == 0:
            return []
        X = vectorizer.fit_transform(words)
        
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        
        clusters = [[] for _ in range(num_clusters)]
        for word, label in zip(words, kmeans.labels_):
            clusters[label].append(word)
        
        return [' '.join(cluster) for cluster in clusters]

    def split_document(doc):
        """
        Splits a document into hierarchical chunks: paragraphs, sentences, and word clusters.
        Args:
            doc (str): The document to split.
        Returns:
            list: List of hierarchical chunks.
        """
        paragraphs = split_into_paragraphs(doc)
        hierarchical_chunks = []

        for paragraph in paragraphs:
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                word_clusters = split_into_word_clusters(sentence)
                hierarchical_chunks.extend(word_clusters)
        
        return hierarchical_chunks

    # Use the new splitting function in the RAG workflow
    def rag_workflow_from_pdf_with_hierarchical_chunks(pdf_path, query):
        document_text = extract_text_from_pdf(pdf_path)
        text_chunks = split_document(document_text)
        
        all_entities = []
        all_relations = []

        for chunk in text_chunks:
            entities = extract_entities(chunk)
            relations = extract_relations(chunk)
            unique_entities, unique_relations = remove_duplicates(entities, relations)
            store_in_vector_db([chunk], unique_entities, unique_relations)
            all_entities.extend(unique_entities)
            all_relations.extend(unique_relations)
        
        merged_chunks = auto_merge_chunks(text_chunks, query)
        relevant_entities, relevant_relations = retrieve_entities_and_relations(merged_chunks)
        response = generate_response_with_entities_and_relations(query, merged_chunks, relevant_entities, relevant_relations)
        
        return response

    # Run the updated RAG workflow with hierarchical chunks
    output = rag_workflow_from_pdf_with_hierarchical_chunks(pdf_path, query)
    response = ask_openai(output)
    print("intermediate step is:", output)
    print("response is:", response)
except Exception as e:
    pass