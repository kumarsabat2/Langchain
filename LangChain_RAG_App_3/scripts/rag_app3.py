from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredWordDocumentLoader,TextLoader,WebBaseLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

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


# 1️⃣ Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"✅ Total chunks created: {len(chunks)}")

# 2️⃣ Convert chunks into embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# 3️⃣ Create FAISS vectorstore and save locally
vectorstore = Chroma.from_documents(chunks, embeddings,persist_directory="chroma_index" )
# 4️⃣ Persist data to disk
vectorstore.persist()
print("✅ Chroma vectorstore created and saved in 'chroma_index'")

# 1️⃣ Initialize retriever from Chroma
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # k = number of chunks to fetch
# 2️⃣ Initialize LLM (GPT model)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# 3️⃣ Build retrieval-based question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
 #4️⃣ Ask your question
while True:
    query = input("\n💬 Enter your question: ")
    if query.lower() == "exit":
        break
    answer = qa_chain.invoke({"query": query})
    print("\n🧠 Answer:")
    print(answer["result"])