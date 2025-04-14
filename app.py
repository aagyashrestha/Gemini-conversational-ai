# app.py
import uuid
import os
from flask import Flask, request, Response
from dotenv import load_dotenv
from memory_store import get_user_memory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from your .env file!")

app = Flask(__name__)

# Hardcoded URL used for RAG
HARD_CODED_URL = "https://site.chimpvine.com"

# Build vectorstore (RAG) from the hardcoded URL
def create_vectorstore_from_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = create_vectorstore_from_url(HARD_CODED_URL)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question")
    user_id = data.get("user_id") or str(uuid.uuid4())

    if not question:
        return Response("Please provide 'question'.", status=400)

    try:
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.7,
            google_api_key=api_key
        )

        # Get memory for this user
        memory = get_user_memory(user_id)

        # Create the RAG chain (without automatic memory management)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False  # optional
        )

        # Fetch last k exchanges from memory and construct chat_history
        chat_history = memory.chat_memory.messages[-14:]  # user-bot-user-bot...
        # Convert messages to (user, bot) tuples
        history_pairs = []
        for i in range(0, len(chat_history), 2):
            user_msg = chat_history[i].content if i < len(chat_history) else ""
            bot_msg = chat_history[i + 1].content if i + 1 < len(chat_history) else ""
            history_pairs.append((user_msg, bot_msg))

        # Invoke the chain with both question and chat_history
        result = qa_chain.invoke({
            "question": question,
            "chat_history": history_pairs
        })["answer"]

        # Append latest exchange to memory manually
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(result)

        # Build full chat log
        chat_log = ""
        for user_msg, bot_msg in history_pairs:
            chat_log += f"{{user: {user_msg}}}\n"
            chat_log += f"{{bot: {bot_msg}}}\n"
        chat_log += f"{{user: {question}}}\n"
        chat_log += f"{{bot: {result}}}\n"

        res = Response(chat_log.strip(), mimetype="text/plain")
        res.headers["X-User-ID"] = user_id
        return res

    except Exception as e:
        print("Error:", e)
        return Response(f"Error: {str(e)}", status=500)


if __name__ == '__main__':
    app.run(debug=True)