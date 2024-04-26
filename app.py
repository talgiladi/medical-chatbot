from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings, create_context
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot2"

docsearch = PineconeVectorStore.from_existing_index(
        index_name = index_name, 
        embedding=embeddings)

prompt = PromptTemplate(
    template = prompt_template,
    input_variables = ["context","question"] )

llm = CTransformers(
    model = "model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type = "llama",
    config = {"max_new_tokens":512,
              "temperature": 0.8}
)

chain = LLMChain(llm= llm, prompt = prompt)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    print("user query: " + msg)
    context = create_context(docsearch=docsearch, query = msg)
    response = chain.run(context = context, question = msg)
    print("llm response: "+response)
    return str(response)


#

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
