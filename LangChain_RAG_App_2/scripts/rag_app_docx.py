import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

docs_folder = "../docs"
index_path = "../App2_faiss_index_docx"

# Load and split .docx files
all_docs = []
for file in os.listdir(docs_folder):
    if file.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(os.path.join(docs_folder, file))
        docs = loader.load()
        all_docs.extend(docs)

print(f"📄 Loaded {len(all_docs)} documents from {docs_folder}")

# Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)
print(f"✂️ Split into {len(chunks)} chunks")

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(index_path)
print(f"✅ FAISS index created at: {index_path}")

# Build QA chain
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.invoke({"query": query})
    print("\n💬 Answer:", answer["result"])