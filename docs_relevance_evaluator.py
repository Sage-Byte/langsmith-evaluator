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
    print("example is ",example)
    return {"answer": example["generated_answer"], "contexts": example["context"]}

def predict_rag_answer_with_context(example: dict):
    """Use this for evaluation of retrieved documents and hallucinations"""
    # response = rag_bot.get_answer(example["input_question"])

    return {"answer": example.outputs["generated_answer"], "contexts": example.outputs["context"]}

# Grade prompt
grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")

def docs_relevance_evaluator(run, example) -> dict:
    """
    A simple evaluator for document relevance
    """
    # print("run is ",run)
    print("outputs is ",example.outputs["context"])
    # print("example.inputs is ",example.inputs['inputs/input_question'])
    # RAG inputs
    input_question = example.inputs['question']
    contexts = example.outputs["context"]

    # LLM grader
    llm = AzureChatOpenAI( openai_api_version="2024-04-01-preview",
    azure_deployment="Eternos-gpt-35-turbo-16k",
    openai_api_key="4dc8934e0bdb4fa6bf5c300f2469bae7")

    # Structured prompt
    answer_grader = grade_prompt_doc_relevance | llm

    # Get score
    score = answer_grader.invoke({"question":input_question,
                                  "documents":contexts})
    score = score["Score"]

    return {"key": "document_relevance", "score": score}





def perform_docs_relevance_evaluation(dataset_id):
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_id,
        # data=dataset_name,
        evaluators=[docs_relevance_evaluator],
        experiment_prefix="docs_relevance_evaluator",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
        )