import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader,TextLoader,WebBaseLoader
from dotenv import load_dotenv

# Load API key (for later use)
load_dotenv()

# --- Define file paths ---
word_path = r"F:\Agentic AI\Langchain\LangChain_RAG_App_3\docs\SANTOSH KUMAR SABAT CV.docx"
text_path = r"F:\Agentic AI\Langchain\LangChain_RAG_App_3\docs\Berhampur.txt"
web_url = "https://en.wikipedia.org/wiki/Puri"

# --- Load data ---
loaders = [
    UnstructuredWordDocumentLoader(word_path),
    TextLoader(text_path),
    WebBaseLoader(web_url)
]

documents=[]

for loader in loaders:
    docs = loader.load()
    documents.extend(docs)

print(f"✅ Loaded {len(documents)} documents from multiple sources!")
print(f"Sample text:\n{documents[2].page_content[:300]}...")