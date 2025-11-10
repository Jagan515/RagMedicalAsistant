from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, docs_to_prompt_input
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from src.prompt import prompt_template
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# Env vars (ensure these are present in your .env)
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY", "")
os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
# Disable tokenizers parallelism to avoid the forking warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Embeddings + Pinecone vectorstore
embeddings = download_hugging_face_embeddings()

index_name = "rag-medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM (ChatOpenAI)
chat = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

parser = StrOutputParser()

docs_to_prompt_runnable = RunnableLambda(
    lambda inputs: docs_to_prompt_input(inputs["docs"], inputs["question"])
)
# make sure we call chat.invoke(...) where `chat` is the LLM object (not shadowed)
question_answer_chain = RunnableLambda(
    lambda inputs: chat.invoke(prompt.format(**inputs))
)

# RAG chain: retriever provides docs, question forwarded
# initial runnable - call the vectorstore directly (stable API)
initial_doc_runnable = RunnableLambda(
    lambda inputs: {
        "docs": docsearch.similarity_search(inputs["question"], k=3),
        "question": inputs["question"],
    }
)

# bind chat to avoid name-capture problems
question_answer_chain = RunnableLambda(
    lambda inputs, _chat=chat: _chat.invoke(prompt.format(**inputs))
)

rag_chain = (
    initial_doc_runnable
    | docs_to_prompt_runnable
    | question_answer_chain
    | parser
)




@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat_route():  # renamed to avoid shadowing the `chat` variable
    try:
        # get message from client
        msg_text = request.form.get("msg") or request.args.get("msg") or ""
        logger.info("User message: %s", msg_text)

        # Invoke the chain with the expected input shape
        result = rag_chain.invoke({"question": msg_text})
        logger.info("RAG result: %s", result)

        # Return as plain text or JSON
        return jsonify({"response": str(result)})

    except Exception as e:
        logger.exception("Error while processing request")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # note: use debug=False in production
    app.run(host="0.0.0.0", port=8080, debug=True)