# chat2.py

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  # Groq LLM wrapper

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Read Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY is not set. Please check your .env file.")

# ✅ Initialize Groq LLM with the specified model
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ✅ Setup RetrievalQA with a custom prompt
def setup_retrieval_qa(db):
    retriever = db.as_retriever(similarity_score_threshold=0.6)

    prompt_template = """Your name is VAYAZH. You are an expert in Agriculture. 
Provide short and brief practical advice. 
If you don't know the answer, simply respond with 'Don't know.'

CONTEXT: {context}
QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        input_key='query',
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True
    )
    return chain
