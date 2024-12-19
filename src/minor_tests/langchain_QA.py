import os
import logging
logging.basicConfig(level=logging.ERROR)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# user specified settings
pdf_path = "intelligent_scissor.pdf"
QUESTION = 'how do '

faiss_index_path = f"{os.path.splitext(pdf_path)[0]}_index"
def is_index_empty(index_path):
    return not os.path.exists(index_path)

if is_index_empty(faiss_index_path):
    loader = PyPDFLoader(pdf_path, extract_images=False)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(faiss_index_path)
else:
    vector_store = FAISS.load_local(faiss_index_path, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, model='gpt-3.5-turbo-instruct'),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

def answer_question(question):
    return retrieval_qa.run(question)

answer = answer_question(QUESTION)
print('\n\n\n\n\n\n')
print("ANSWER:", answer)

retrieved_docs = retrieval_qa.retriever.invoke(QUESTION)
print('\n\n\n\n\n\n')
print(f"The chosen chunk is\n:{retrieved_docs[0].page_content}")