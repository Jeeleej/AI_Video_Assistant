# Action items, decisions, open questions

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    return prompt | get_llm() | StrOutputParser()


def _run(system_prompt: str, transcript: str) -> str:
    return build_chain(system_prompt).invoke({"text": transcript})


def extract_action_items(transcript: str) -> str:
    return _run(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        transcript,
    )


def extract_key_decisions(transcript: str) -> str:
    return _run(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        transcript,
    )


def extract_questions(transcript: str) -> str:
    return _run(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'",
        transcript,
    )