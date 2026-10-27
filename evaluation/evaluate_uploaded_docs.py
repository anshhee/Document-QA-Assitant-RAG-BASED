# pyrefly: ignore[missing-import]
from openai import AsyncOpenAI
from src.rag.service import (collection , collection_lock ,  generate_answer ,)

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

import re
import asyncio
import json
import sys
from pathlib import Path

import ollama
from ragas.metrics.collections import Faithfulness, AnswerRelevancy


def load_document_chunks(document_id: str) -> list[dict]:
    """Load all stored chunks belonging to one uploaded document."""
    with collection_lock:
        result = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    chunks = []

    for text, metadata in zip(
        result["documents"],
        result["metadatas"],
    ):
        if text and text.strip():
            chunks.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                }
            )

    chunks.sort(
        key=lambda chunk: chunk["metadata"].get("chunk_index", 0)
    )

    return chunks


def generate_questions(
    chunks: list[dict],
    num_questions: int = 5,
) -> list[str]:
    """Generate standalone evaluation questions from a document."""

    if not chunks:
        raise ValueError("Cannot generate questions from an empty document.")

    if num_questions < 1:
        raise ValueError("num_questions must be at least 1.")

    sample_count = min(8, len(chunks))

    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(i * (len(chunks) - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]

    document_text = "\n\n".join(
        f"Chunk {index + 1}:\n{chunks[index]['text'][:1500]}"
        for index in indices
    )

    prompt = f"""

You are generating evaluation questions for a document-based RAG system.

Based ONLY on the document excerpts below:

- Each question must be understandable on its own.
- Each question must be answerable directly from the document.
- Ask one specific thing per question.
- Avoid broad questions with multiple parts.
- Avoid questions about figures, images, or tables unless their contents are explicitly described in the text.
- Cover different topics where possible.  


Document excerpts:
{document_text}
""".strip()

    response = ollama.generate(
        model="qwen2.5:3b",
        prompt=prompt,
    )

    raw_output = response.get("response", "").strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.split("\n", 1)[-1]

        if raw_output.endswith("```"):
            raw_output = raw_output[:-3].strip()

    questions = None 
    try:
        questions = json.loads(raw_output)
    except json.JSONDecodeError:
        repaired_output = re.sub(
            r'"\s*\n\s*"',
            '",\n"',
            raw_output,
        )

        try: 
            questions =  json.loads(repaired_output)
        except json.JSONDecodeError:
            numbered_questions = re.findall(
                r"^\s*\d+[.)]\s*(.+?)\s*$",
                raw_output,
                flags=re.MULTILINE,
            )

            if len(numbered_questions) >= num_questions:
                questions = numbered_questions[:num_questions]

            else:
                raise ValueError(
                    f"Qwen returned invalid question format.\n"
                    f"Parsed Question: {questions}\n"
                    f"Raw Output: {raw_output}"
                )


    if (
        not isinstance(questions, list)
        or len(questions) != num_questions
        or not all(
            isinstance(question, str) and question.strip()
            for question in questions
        )
    ):

        raise ValueError(
        f"Qwen returned {len(questions) if isinstance(questions, list) else 'non-list'} "
        f"questions instead of {num_questions}.\n"
        f"Questions: {questions}"
        )

    return [question.strip() for question in questions]


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


async def evaluate_uploaded_document(document_id: str) -> Path:
    """Evaluate one uploaded document and save its RAGAS results."""

    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty.")

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
    )

    try:
        llm = llm_factory(
            model="qwen2.5:3b",
            provider="openai",
            client=client,
        )

        embeddings = embedding_factory(
            model="ollama/nomic-embed-text",
            provider="litellm",
            base_url="http://localhost:11434",
        )

        faithfulness_metric = Faithfulness(llm=llm)

        relevancy_metric = AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        )

        chunks = load_document_chunks(document_id)

        if not chunks:
            raise ValueError(
                f"No chunks found for document_id={document_id}"
            )

        questions = generate_questions(chunks)
        results = []

        for index, question in enumerate(questions, start=1):
            print(
                f"Evaluating question "
                f"{index}/{len(questions)}..."
            )

            rag_result = generate_answer(
                query=question,
                k=5,
                document_id=document_id,
            )

            answer = rag_result["answer"]


            retrieved_chunks = rag_result.get("chunks", []) 
            contexts = [
                chunk["text"]
                for chunk in rag_result.get("chunks", [])
                if chunk.get("text")
            ]

            evaluation = await evaluate_response(
                question=question,
                answer=answer,
                contexts=contexts,
                faithfulness_metric=faithfulness_metric,
                relevancy_metric=relevancy_metric,
            )

            evaluation["document_id"] = document_id
            evaluation["model_used"] = rag_result.get("model_used")
            evaluation["contexts"] = contexts 
            evaluation["num_contexts"] = len(contexts)
            evaluation["retrieved_chunks"] = [
                {
                    "chunk_id" : chunk.get("chunk_id"),
                    "text" : chunk.get("text"),
                    "page" : chunk.get("page"),
                    "section" : chunk.get("section"),
                    "filename" : chunk.get("filename"),
                    "distance": chunk.get("distance"),
                }
                for chunk in retrieved_chunks
            ] 

            results.append(evaluation)

        output_dir = Path("evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / "uploaded_documents_ragas_results.json"
        )

        counter = 1

        while output_path.exists():
            output_path = (
                output_dir
                / f"uploaded_documents_ragas_results_{counter}.json"
            )
            counter += 1

        average_faithfulness = (
            sum(
                result["faithfulness"]
                for result in results
            ) / len(results)
            if results 
            else 0.0
        )

        average_answer_relevancy = (
            sum(
                result["answer_relevancy"]
                for result in results
            
        ) / len(results)
        if results 
        else 0.0
        )

        output_data = {
            "document_id": document_id,
            "num_questions": len(questions),
            "results": results,
            "summary": {
                "average_faithfulness": (average_faithfulness),
                "average_answer_relevancy": (average_answer_relevancy),
            },

            "results": results,      
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                output_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path

    finally:
        await client.close()


async def main() -> None:
    """Allow manual command-line testing."""

    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python evaluation/evaluate_uploaded_docs.py "
            "<document_id>"
        )

    document_id = sys.argv[1]

    output_path = await evaluate_uploaded_document(
        document_id=document_id
    )

    print(f"Saved evaluation results to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())


