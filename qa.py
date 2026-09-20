from model import llm

ANSWER_TEMPLATE = (
    "Answer the question using the chat history below.\n\n"
    "History:\n{history}\n\n"
    "Current date: {date}\n"
    "Question: {question}\n"
    "Answer:"
)

JUDGE_TEMPLATES = {
    "default": (
        "Does the model response contain the correct answer? "
        "Answer yes or no only.\n\n"
        "Question: {question}\n"
        "Correct answer: {answer}\n"
        "Model response: {response}\n"
    ),
    "temporal-reasoning": (
        "Does the model response contain the correct answer? "
        "Off-by-one day/week/month errors are still correct. "
        "Answer yes or no only.\n\n"
        "Question: {question}\n"
        "Correct answer: {answer}\n"
        "Model response: {response}\n"
    ),
    "knowledge-update": (
        "Does the model response contain the updated correct answer? "
        "Answer yes or no only.\n\n"
        "Question: {question}\n"
        "Correct answer: {answer}\n"
        "Model response: {response}\n"
    ),
    "single-session-preference": (
        "Does the response satisfy the rubric / personal preference? "
        "Answer yes or no only.\n\n"
        "Question: {question}\n"
        "Rubric: {answer}\n"
        "Model response: {response}\n"
    ),
}


def generate_answer(entry: dict, history: str) -> str:
    prompt = ANSWER_TEMPLATE.format(
        history=history,
        date=entry["question_date"],
        question=entry["question"],
    )
    return llm.invoke(prompt).content.strip()


def judge_answer(entry: dict, response: str) -> bool:
    qtype = entry["question_type"]
    key = qtype if qtype in JUDGE_TEMPLATES else "default"
    prompt = JUDGE_TEMPLATES[key].format(
        question=entry["question"],
        answer=entry["answer"],
        response=response,
    )
    verdict = llm.invoke(prompt).content.strip().lower()
    return "yes" in verdict
