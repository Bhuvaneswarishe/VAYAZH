import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage
from langchain_core.language_models import LLM
from typing import List
import os

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Make GeminiLLM a LangChain-compatible LLM
class GeminiLLM(LLM):
    model: str = "gemini-2.0-flash"

    def _call(self, prompt: str, stop=None) -> str:
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else "I'm unable to provide an answer at the moment."

    def invoke(self, input_text: str) -> str:
        return self._call(input_text)

    @property
    def _llm_type(self) -> str:
        return "custom"

# Create an instance of the Gemini model wrapped in LangChain's format
llm = GeminiLLM()

# ✅ Fix `setup_retrieval_qa` to work with the new LLM
def setup_retrieval_qa(db):
    retriever = db.as_retriever(similarity_score_threshold=0.6)

    prompt_template = """Your name is VAYAZH. You are an expert in Agriculture. 
Provide short and brief with practical advice  
If you don't know the answer, simply respond with 'Don't know.'

CONTEXT: {context}
QUESTION: {question}"""


    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize the RetrievalQA chain with LangChain-compatible Gemini model
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
