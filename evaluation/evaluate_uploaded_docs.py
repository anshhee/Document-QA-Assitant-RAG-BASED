from openai import AsyncOpenAI 

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory 

import asyncio
import json
from pathlib import Path

from ragas.metrics.collections import Faithfulness, AnswerRelevancy


def load_questions(path: str | Path) -> list[dict]:
    """Load evaluation questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("The questions file must contain a JSON list.")

    return questions


async def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
    faithfulness_metric: Faithfulness,
    relevancy_metric: AnswerRelevancy,
) -> dict:
    """Score one answer using RAGAS metrics."""
    faithfulness = await faithfulness_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    relevancy = await relevancy_metric.ascore(
        user_input=question,
        response=answer,
    )

    return {
        "question": question,
        "answer": answer,
        "faithfulness": faithfulness.value,
        "answer_relevancy": relevancy.value,
    }


async def main() -> None:
    # Model and embedding configuration will be added next.
    
    # We are deliberately not running scores until these are configured.
    client =  AsyncOpenAI(
        base_url = "http://localhost:11434/v1",
        api_key = "not-needed" #required to import
    )
    try:
        llm = llm_factory(
            model = "qwen2.5:3b",
            provider = "openai",
            client = client, 
        )

        embeddings = embedding_factory(
            model = "ollama/nomic-embed-text",
            provider = "litellm",
            base_url = "http://localhost:11434",
        )
        faithfulness_metric = Faithfulness(llm = llm)
        relevancy_metric = AnswerRelevancy(llm = llm , embeddings = embeddings,)
                  
   
        print("RAGAS Evaluation models Configured successfully")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
