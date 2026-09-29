"""Shared reader (rule 4): one answer template for every method, temperature 0.
Template: LongMemEval official CoT history-chats prompt (src/generation/run_generation.py,
xiaowu0162/LongMemEval @ 9e0b455)."""
from __future__ import annotations

from datetime import datetime

TEMPLATE = ("I will give you several history chats between you and a user. Please answer the question based on the "
            "relevant chat history. Answer the question step by step: first extract all the relevant information, "
            "and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}"
            "\nQuestion: {}\nAnswer (step by step):")


def reader_prompt(question: str, context: str, question_date: datetime | None) -> str:
    date = question_date.strftime("%Y/%m/%d (%a) %H:%M") if question_date else "unknown"
    return TEMPLATE.format(context, date, question)


def read(client, question: str, context: str, question_date: datetime | None, qid: str | None = None,
         max_tokens: int = 1024) -> str:
    return client.complete(reader_prompt(question, context, question_date), stage="reader", qid=qid,
                           max_tokens=max_tokens).text
