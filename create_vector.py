from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DataPath = "data/"
vector_path = "vectorstore/db_faiss"

def create_vector_data():
    loader = DirectoryLoader(DataPath,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':'cpu'})
    
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(vector_path)

if __name__ == '__main__':
    create_vector_data()