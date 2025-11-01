# Day2_Chroma.py

from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

texts = [
    "Chroma is lightweight and fast.",
    "It stores embeddings locally using SQLite."
]

db = Chroma.from_texts(texts, embedding=embeddings, persist_directory="./chroma_store")

query = "Where are embeddings stored?"
results = db.similarity_search(query)

print(results[0].page_content)
