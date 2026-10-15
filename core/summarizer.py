import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # Summarize each chunk
    chunk_summaries = map_chain.batch([{"text": chunk} for chunk in chunks])
    combined = "\n\n".join(chunk_summaries)

    # Short transcripts: if there is only 1 chunk, avoid unnecessary duplicate calls
    if len(chunks) <= 1:
        return combined

    combine_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    combine_chain = combine_prompt | llm | StrOutputParser()

    return combine_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()
    time.sleep(1)

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 12 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({"text": transcript[:2000]}).strip().strip('"')