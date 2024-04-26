from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from load_dotenv import load_dotenv
import os

#load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#print(PINECONE_API_KEY)

#load the data files
extracted_data = load_pdf("data/")

#split them to chunks
text_chunks = text_split(extracted_data)

#load the embeddings model
embeddings = download_hugging_face_embeddings()

#prepare the vector store -new or existing
docsearch = None
index_name = "medical-chatbot2"
has_existing_index = True
if (has_existing_index):
    #the vector store already has the data
    docsearch = PineconeVectorStore.from_existing_index(
        index_name = index_name, 
        embedding=embeddings)
else:
    #upload the data now
    batch_size = 300  # Adjust this value based on your requirement
    chunks = [t.page_content for t in text_chunks]
    
    # Split the chunks array into smaller arrays
    chunks_small = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
  
    # Loop through each small array and upload it
    for i, small_chunk in enumerate(chunks_small):
        print(f"Uploading batch {i+1} out of {len(chunks_small)}")
        docsearch=PineconeVectorStore.from_texts(small_chunk, embeddings, index_name=index_name)
        print(f"Finished uploading batch {i+1}")