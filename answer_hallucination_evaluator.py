from langchain import hub
from langchain_openai import AzureChatOpenAI
import os
import dotenv
dotenv.load_dotenv()
from langsmith.evaluation import evaluate
from langsmith import Client

client = Client()

def predict_rag_answer(example: dict):
    """Use this for answer evaluation"""
    # response = rag_bot.get_answer(example["input_question"])
    # return {"answer": response["answer"]}
    # return {"answer": "A isotope of hydrogen with 1 proton and 3 neutrons"}
    return {"answer": example["generated_answer"], "contexts": example["context"]}

def predict_rag_answer_with_context(example: dict):
    """Use this for evaluation of retrieved documents and hallucinations"""
    # response = rag_bot.get_answer(example["input_question"])
    return {"answer": example.outputs["generated_answer"], "contexts": example.outputs["context"]}


# Prompt
grade_prompt_hallucinations = prompt = hub.pull("langchain-ai/rag-answer-hallucination")

def answer_hallucination_evaluator(run, example) -> dict:
    """
    A simple evaluator for generation hallucination
    """

    # RAG inputs
    input_question = example.inputs['question']
    contexts = example.outputs["context"]

    # RAG answer
    prediction = example.outputs["generated_answer"]

    # LLM grader
    llm = AzureChatOpenAI( openai_api_version="2024-04-01-preview",
    azure_deployment="Eternos-gpt-35-turbo-16k",
    openai_api_key="4dc8934e0bdb4fa6bf5c300f2469bae7")

    # Structured prompt
    answer_grader = grade_prompt_hallucinations | llm

    # Get score
    score = answer_grader.invoke({"documents": contexts,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_hallucination", "score": score}




def perform_answer_hallucination_evaluation(dataset_id):
    experiment_results = evaluate(
    predict_rag_answer,
    data=dataset_id,
    evaluators=[answer_hallucination_evaluator],
    experiment_prefix="answer_hallucination_evaluator",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )