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
    return {"answer": example["generated_answer"], "contexts": example["context"]}

def predict_rag_answer_with_context(example: dict):
    """Use this for evaluation of retrieved documents and hallucinations"""
    # response = rag_bot.get_answer(example["input_question"])
    return {"answer": example.outputs["generated_answer"], "contexts": example.outputs["context"]}

# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")

def answer_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """

    # Get question, ground truth answer, RAG chain answer
    input_question = example.outputs["question"]
    reference = example.outputs["actual_answer"]
    prediction = example.outputs["generated_answer"]



    # LLM grader
    llm = AzureChatOpenAI( openai_api_version="2024-04-01-preview",
    azure_deployment="Eternos-gpt-35-turbo-16k",
    openai_api_key="4dc8934e0bdb4fa6bf5c300f2469bae7")

    # print("llm is ",llm)
    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    print("score is ",score)
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}




def perform_answer_evaluation(dataset_id):
    experiment_results = evaluate(
    predict_rag_answer,
    data=dataset_id,
    evaluators=[answer_evaluator],
    experiment_prefix="rag-answer-vs-reference",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )

