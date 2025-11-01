import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def index_docs(docs_folder: str = "../docs", index_path: str = "../faiss_index"):
    """Read .txt files, create embeddings, and store them locally with FAISS."""
    load_dotenv()  # load the .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in .env file")

    # 1️⃣ Load all .txt documents
    docs = []
    for fname in os.listdir(docs_folder):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_folder, fname), encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        print("⚠️ No .txt files found in docs folder.")
        return

    # 2️⃣ Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 3️⃣ Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4️⃣ Save FAISS index locally
    vectorstore.save_local(index_path)
    print(f"✅ Index created and saved at: {index_path}")

from langchain_core.caches import BaseCache
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def ask_question(index_path: str = "../faiss_index"):
    """Ask a question using the saved FAISS index."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in .env file")

    # Load embeddings and FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever and LLM
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

    # Build the Retrieval-QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Ask user for input
    print("🤖 Ask a question about your documents (type 'exit' to quit):")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting QA mode.")
            break

        answer = qa_chain.run(query)
        print("\n💬 Answer:\n", answer)

if __name__ == "__main__":
    print("Select an option:")
    print("1. Create / Update index")
    print("2. Ask questions")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        index_docs()
    elif choice == "2":
        ask_question()
    else:
        print("❌ Invalid choice.")
