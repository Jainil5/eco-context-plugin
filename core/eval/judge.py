"""LongMemEval official judge prompts, verbatim from src/evaluation/evaluate_qa.py
(xiaowu0162/LongMemEval @ 9e0b455). Official decoding: temperature 0, max_tokens 10, label = 'yes' in reply."""
from __future__ import annotations

_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
         "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
         "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
         "only contains a subset of the information required by the answer, answer no. ")
_QA = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."


def anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if abstention:
        t = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes "
             "if the model correctly identifies the question as unanswerable. The model could say that the information "
             "is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
             "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
             "Answer yes or no only.")
    elif task in ("single-session-user", "single-session-assistant", "multi-session"):
        t = _BASE + _QA
    elif task == "temporal-reasoning":
        t = (_BASE + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for "
             "the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days "
             "when the answer is 18), the model's response is still correct. " + _QA)
    elif task == "knowledge-update":
        t = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
             "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
             "information along with an updated answer, the response should be considered as correct as long as the "
             "updated answer is the required answer." + _QA)
    elif task == "single-session-preference":
        t = ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
             "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
             "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
             "utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}"
             "\n\nIs the model response correct? Answer yes or no only.")
    else:
        raise NotImplementedError(task)
    return t.format(question, answer, response)


def judge(client, q, response: str, max_tokens: int = 10) -> tuple[bool, str]:
    prompt = anscheck_prompt(q.qtype, q.question, q.answer, response, abstention=q.is_abstention)
    reply = client.complete(prompt, stage="judge", qid=q.qid, max_tokens=max_tokens).text
    return "yes" in reply.lower(), reply
