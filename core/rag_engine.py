import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from core.vector_store import build_vector_store, load_vector_store, get_retriever

load_dotenv()

SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
    )


def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain(vector_store):
    retriever = get_retriever(vector_store, k=4)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    # Clean LCEL mapping: question feeds both the retriever and the prompt
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain


def build_rag_chain(transcript: str):
    """Index a new transcript and return a ready-to-use RAG chain."""
    vector_store = build_vector_store(transcript)
    return _build_chain(vector_store)


def load_rag_chain():
    """Rebuild the RAG chain from the previously persisted vector store."""
    vector_store = load_vector_store()
    return _build_chain(vector_store)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer