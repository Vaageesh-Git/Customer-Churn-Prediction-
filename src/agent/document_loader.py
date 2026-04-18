import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_documents(directory_path: str = "./knowledge_base"):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created {directory_path}. Please add your.txt files.")
        return


    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    raw_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
    )
    
    chunked_docs = text_splitter.split_documents(raw_documents)
    return chunked_docs