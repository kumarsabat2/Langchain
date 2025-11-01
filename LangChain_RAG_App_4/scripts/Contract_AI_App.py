import os
from langchain_community.document_loaders import Docx2txtLoader

documents=[]
folder_path="F:\Agentic AI\Langchain\LangChain_RAG_App_4\docs"
for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue

        docs = loader.load()
        documents.extend(docs)
        print(f"✅ Loaded {filename}")


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2️⃣ Split long documents into smaller chunks

splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100,separators=["\n\n", "\n", ".", " "])

chunks = splitter.split_documents(documents)
print(f"🪓 Split into {len(chunks)} chunks")

# 3️⃣ Create embeddings
embeddings = OpenAIEmbeddings(api_key=api_key)
# 4️⃣ Store in Chroma vector database
persist_directory = "../chroma_db"
vectorstore =Chroma.from_documents(documents=chunks,embedding=embeddings,persist_directory=persist_directory)
print("✅ Chroma vectorstore created and saved locally!")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# 3️⃣ Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)
# 4️⃣ Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    verbose=True
)
# 5️⃣ Add a post-review template
def analyze_answer(query, answer):
    review_prompt = f"""
    You are a compliance and contract analyst AI.

    Based on the following answer, rate the risk level and suggest an action.
    Answer: {answer}

    Return the output in this format:
    - Summary:
    - Risk Level (Low/Medium/High):
    - Recommended Action:
    """
    review = llm.invoke(review_prompt)
    return review.content


# 6️⃣ Interactive Q&A loop
def run_policy_reviewer():
    print("\n🤖 Smart Policy & Contract Reviewer Ready!")
    print("Ask about your contracts or policies (type 'exit' to quit).\n")

    while True:
        query = input("🧑‍💼 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        base_answer = qa_chain.invoke({"query": query})
        final_report = analyze_answer(query, base_answer["result"])

        print("\n💬 AI Analysis:")
        print(final_report)
        print("-" * 60)
if __name__ == "__main__":
    run_policy_reviewer()





