from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def text_split(documents: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_documents(documents=documents)
    return chunks

def load_pdf(directory: str):
    loader = DirectoryLoader(directory,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def create_context(docsearch, query: str):
    retriever = docsearch.as_retriever()
    matched_docs = docsearch.similarity_search(query=  query, k = 3)

    #combine them
    context = ""
    for i, d in enumerate(matched_docs):
        context = context + (f"\n## Document {i}\n")
        context = context + (d.page_content)

    return context
