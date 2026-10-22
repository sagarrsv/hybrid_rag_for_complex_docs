
"""
evaluation/generation_eval.py
=============================================================
Standalone generation evaluation for Hybrid V2 Decay@3.

Experiments:
    retrieved  : Decay@3 -> multimodal generator
    oracle     : Gold pages -> multimodal generator
    text_only  : Decay@3 -> text-only context -> generator

Independent of evaluation/eval_harness.py.
=============================================================
"""

import json
import time
import random
from collections import deque
from pathlib import Path
from typing import Optional

from config import (
    GENERATOR_CONTEXT_PAGES,
    GENERATOR_RPM,
    GENERATOR_TPM,
    GENERATOR_RPD,
    GENERATOR_RETRIEVAL_MODE,
    GENERATION_RESULTS_DIR,
    snapshot,
)


def save_jsonl(path: Path, record: dict):
    """Append one JSON record immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def load_completed(path: Path) -> set:
    """Recover already completed questions."""
    completed = set()

    if not path.exists():
        return completed

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("status") == "completed":
                completed.add(record["qid"])

    return completed


class RequestLimiter:
    """
    Local RPM and estimated TPM pacing.

    This protects one running evaluation process.

    It does not know about requests from other processes
    using the same Google Cloud project.
    """

    def __init__(
        self,
        rpm: int,
        tpm: int,
        rpd: int,
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd

        self.requests = deque()
        self.tokens = deque()
        self.calls_this_run = 0

    def wait(self, estimated_tokens: int):
        """
        Wait until one more request fits both rolling
        60-second windows.
        """
        # Reserve room for generated output and for
        # system instructions not in the token estimate.
        estimated_tokens += 1000

        if estimated_tokens > self.tpm:
            raise ValueError(
                "Single request exceeds configured TPM."
            )

        if self.calls_this_run >= self.rpd:
            raise RuntimeError(
                "Local RPD budget exhausted."
            )

        while True:
            now = time.monotonic()

            while (
                self.requests
                and now - self.requests[0] >= 60
            ):
                self.requests.popleft()

            while (
                self.tokens
                and now - self.tokens[0][0] >= 60
            ):
                self.tokens.popleft()

            recent_tokens = sum(
                count for _, count in self.tokens
            )

            rpm_ok = len(self.requests) < self.rpm
            tpm_ok = (
                recent_tokens + estimated_tokens
                <= self.tpm
            )

            if rpm_ok and tpm_ok:
                break

            # Conservative polling.
            time.sleep(1)

        now = time.monotonic()

        self.requests.append(now)
        self.tokens.append(
            (now, estimated_tokens)
        )

        self.calls_this_run += 1


def build_oracle_hits(record: dict, retriever):
    """
    Use the annotated gold pages directly.

    The retrieval model is not invoked.
    """
    hits = []

    for gold in record["gold_pages"]:
        doc_id = str(gold["doc_id"])
        page_num = int(gold["page_num"])

        page_type = retriever.get_page_type(
            doc_id,
            page_num,
        )

        if page_type is None:
            raise ValueError(
                f"Gold page missing from Qdrant: "
                f"{doc_id}, page {page_num}"
            )

        hits.append({
            "doc_id": doc_id,
            "page_num": page_num,
            "page_type": page_type,
            "source": "oracle",
            "content": "",
        })

    return hits

def generate_with_retry(generator, query, context, limiter):
    # One generation request only.
    # No token-count API calls and no automatic retries.
    limiter.wait(0)
    return generator.generate(query, context)

'''
def generate_with_retry(
    generator,
    query,
    context,
    limiter,
):
    """
    Retry transient Gemini API failures.

    Initial request and each retry are counted
    against the local rate-limit scheduler.
    """
    from google.genai import errors

    for attempt in range(5):
        # Counting is used only to estimate the
        # size of the actual generation request.
        # Estimate tokens for the actual generation mode.
        comparison = generator.compare_token_modes(query, context)

        mode = (
            "text_image"
            if generator.layout_text_enabled
            else "image_only"
        )

        estimated_tokens = (
            comparison[mode]["gemini_counted_input_tokens"]
            if generator.provider == "gemini"
            else comparison[mode]["groq_projection"]["estimated_input_tokens"]
        )

        limiter.wait(estimated_tokens)

        try:
            return generator.generate(
                query,
                context,
            )

        except errors.APIError as exc:
            code = getattr(exc, "code", None)

            if code not in (429, 500, 502, 503, 504):
                raise

            if attempt == 4:
                raise

            # Exponential backoff after transient failure.
            delay = min(
                60,
                4 * (2 ** attempt)
                + random.uniform(0, 1),
            )

            print(
                f"API error {code}; "
                f"retrying after {delay:.1f}s"
            )

            time.sleep(delay)
'''
def summarize_generation_tokens(results_path: Path):
    """
    Aggregate token usage across all completed queries.

    Reads the JSONL file instead of maintaining a running
    counter, so resumed evaluations aren't double-counted.
    """
    actual_input = 0
    actual_output = 0
    actual_total = 0

    gemini_text_image = 0
    gemini_image_only = 0

    groq_text_image = 0
    groq_image_only = 0

    completed = 0
    missing_counts = 0

    if not results_path.exists():
        return {}

    # Deduplicate by QID to handle accidental reruns.
    records = {}

    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)

                if record.get("status") == "completed":
                    records[record["qid"]] = record

    for record in records.values():
        completed += 1

        usage = record.get("usage", {})
        comparison = record.get("token_comparison", {})

        # Missing usage must not silently count as zero.
        if not all(
            usage.get(k) is not None
            for k in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        ):
            missing_counts += 1
        else:
            actual_input += usage["input_tokens"]
            actual_output += usage["output_tokens"]
            actual_total += usage["total_tokens"]

        text_image = comparison.get("text_image", {})
        image_only = comparison.get("image_only", {})

        gemini_text_image += (
            text_image.get("gemini_counted_input_tokens")
            or 0
        )
        gemini_image_only += (
            image_only.get("gemini_counted_input_tokens")
            or 0
        )

        groq_text_image += (
            text_image.get("groq_projection", {})
            .get("estimated_input_tokens", 0)
        )
        groq_image_only += (
            image_only.get("groq_projection", {})
            .get("estimated_input_tokens", 0)
        )

    summary = {
        "completed_queries": completed,
        "queries_missing_actual_usage": missing_counts,
        "actual_generation": {
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "total_tokens": actual_total,
        },
        "gemini_counted_input": {
            "text_image": gemini_text_image,
            "image_only": gemini_image_only,
            "savings": (
                gemini_text_image - gemini_image_only
            ),
        },
        "groq_projected_input": {
            "text_image": groq_text_image,
            "image_only": groq_image_only,
            "savings": (
                groq_text_image - groq_image_only
            ),
        },
    }

    # Per-query averages for comparison.
    if completed:
        summary["per_query_average"] = {
            "actual_total_tokens": (
                actual_total / completed
            ),
            "groq_text_image_input": (
                groq_text_image / completed
            ),
            "groq_image_only_input": (
                groq_image_only / completed
            ),
        }

    summary_path = (
        results_path.parent / "token_summary.json"
    )

    with summary_path.open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return summary


def run_generation_evaluation(
    golden_path: str,
    retriever,
    context_builder,
    generator,
    experiment: str = "retrieved",
    output_dir: str = GENERATION_RESULTS_DIR,
    max_queries: Optional[int] = None,
):
    """
    Evaluate a generator using existing gold annotations.

    This runner collects predictions and metadata.
    It does not calculate answer correctness itself.
    """
    valid_experiments = {
        "retrieved",
        "oracle",
        "text_only",
    }

    if experiment not in valid_experiments:
        raise ValueError(
            f"Unknown experiment: {experiment}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Use a separate directory for each model/configuration
    # to avoid mixing results from different experiments.
    run_dir = output / generator.model_name / experiment
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    config_path = run_dir / "config.json"

    with open(golden_path, encoding="utf-8") as handle:
        dataset = json.load(handle)

    if max_queries is not None:
        dataset = dataset[:max_queries]

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **snapshot(),
                "experiment": experiment,
                "golden_path": golden_path,
                "generator_model": generator.model_name,
            },
            handle,
            indent=2,
        )

    completed = load_completed(results_path)

    limiter = RequestLimiter(
        rpm=GENERATOR_RPM,
        tpm=GENERATOR_TPM,
        rpd=GENERATOR_RPD,
    )

    print(
        f"Generation experiment: {experiment}\n"
        f"Model: {generator.model_name}\n"
        f"Dataset: {len(dataset)} queries\n"
        f"Already completed: {len(completed)}"
    )
    # Warm up only the retriever used for generation.
    # Exclude warmup latency from evaluation metrics.
    if experiment != "oracle":
        retriever.search(
            "hybrid_v2_d3",
            "benchmark system warmup query",
            k=3
        )

    for index, record in enumerate(dataset, start=1):
        qid = record["qid"]

        if qid in completed:
            continue

        query = record["question"]

        try:
            # -----------------------------------------
            # Step 1: Obtain pages.
            # -----------------------------------------
            start = time.perf_counter()

            if experiment == "oracle":
                if not record.get("answerable", True):
                    # No gold evidence exists for genuinely
                    # unanswerable questions.
                    hits = []
                else:
                    hits = build_oracle_hits(
                        record,
                        retriever,
                    )
            else:
                hits, _ = retriever.search(
                    GENERATOR_RETRIEVAL_MODE,
                    query,
                    k=GENERATOR_CONTEXT_PAGES,
                )

            retrieval_latency = (
                time.perf_counter() - start
            )

            # -----------------------------------------
            # Step 2: Build context.
            # -----------------------------------------
            start = time.perf_counter()

            context = context_builder.build(
                retrieved_docs=hits,
                max_pages=(
                    None if experiment == "oracle"
                    else GENERATOR_CONTEXT_PAGES
                ),
                include_images=(
                    experiment != "text_only"
                ),
            )

            context_latency = (
                time.perf_counter() - start
            )

            # -----------------------------------------
            # Step 3: Generate the answer.
            # -----------------------------------------
            if context["page_count"] == 0:
                result = {
                    "answer": "Insufficient evidence.",
                    "answerable": False,
                    "citations": [],
                    "usage": {},
                    "generation_latency": 0.0,
                    "token_comparison": "Not performed",
                    "generation_mode": "text"
                }
            else:
                result = generate_with_retry(
                    generator,
                    query,
                    context,
                    limiter,
                )

            # -----------------------------------------
            # Step 4: Save the prediction and trace.
            # -----------------------------------------
            prediction = {
                "qid": qid,
                "status": "completed",
                "experiment": experiment,
                "model": generator.model_name,
                "question": query,
                "reference_answer":
                    record.get("answer"),
                "gold_answerable":
                    record.get("answerable", True),
                "predicted_answerable":
                    result["answerable"],
                "prediction": result["answer"],
                "citations": result["citations"],
                "gold_pages":
                    record.get("gold_pages", []),
                "retrieved_pages": [
                    {
                        "doc_id": hit["doc_id"],
                        "page_num": hit["page_num"],
                        "page_type": hit["page_type"],
                    }
                    for hit in hits
                ],
                "evidence_type":
                    record.get("evidence_type"),
                "difficulty":
                    record.get("difficulty"),
                "reasoning_type":
                    record.get("reasoning_type"),
                "page_count":
                    context["page_count"],
                "image_count":
                    context["image_count"],
                "usage": result["usage"],
                "retrieval_latency":
                    retrieval_latency,
                "context_latency":
                    context_latency,
                "generation_latency":
                    result["generation_latency"],
                "token_comparison": 
                    result.get("token_comparison", {}),
                "generation_mode": 
                    result.get("generation_mode", "text_image"),
            }

            save_jsonl(
                results_path,
                prediction,
            )

            completed.add(qid)

            # ---------------------------------------------------
            # Clean per-query evaluation output
            # ---------------------------------------------------
            total_latency = (
                retrieval_latency
                + context_latency
                + result["generation_latency"]
            )

            difficulty = record.get("difficulty", "N/A")

            print("\n" + "=" * 85)
            print(
                f"[{index}/{len(dataset)}] QID: {qid} "
                f"| Difficulty: {str(difficulty).upper()}"
            )

            print(
                f"Latency: {total_latency:.2f}s total "
                f"| Retrieval: {retrieval_latency:.2f}s "
                f"| Context: {context_latency:.2f}s "
                f"| Generation: {result['generation_latency']:.2f}s"
            )
            print("=" * 85)

            print(f"\nQUESTION:\n{query}")

            print(f"\nGENERATED ANSWER:\n{result['answer']}")

            print(
                f"\nREFERENCE ANSWER:\n"
                f"{record.get('answer', 'N/A')}"
            )

            print("\nRETRIEVED PAGES:")
            if hits:
                for rank, hit in enumerate(hits, start=1):
                    print(
                        f"  {rank}. {hit['doc_id']} "
                        f"| Page {hit['page_num']} "
                        f"| {hit['page_type']}"
                    )
            else:
                print("  None")

            print("\nGOLD PAGES:")
            if record.get("gold_pages"):
                for gold in record["gold_pages"]:
                    print(
                        f"  {gold['doc_id']} "
                        f"| Page {gold['page_num']}"
                    )
            else:
                print("  None")

            usage = result.get("usage", {})

            print(
                f"\nACTUAL GENERATION TOKENS: "
                f"Input={usage.get('input_tokens', 'N/A')} | "
                f"Output={usage.get('output_tokens', 'N/A')} | "
                f"Total={usage.get('total_tokens', 'N/A')}"
            )

            print(f"Page images sent: {context['image_count']}")
            print("=" * 85 + "\n")


        except Exception as exc:
            save_jsonl(
                errors_path,
                {
                    "qid": qid,
                    "status": "error",
                    "error": str(exc),
                },
            )

            print(f"[ERROR] {qid}: {exc}")

            # Do not continue blindly if the API
            # indicates exhausted daily quota.
            message = str(exc).lower()

            if (
                "quota" in message
                or "resource_exhausted" in message
                or "local rpd budget" in message
            ):
                raise

    # Outside the loop — runs once after all queries
    summary = summarize_generation_tokens(results_path)

    print("\n--- Generation Token Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {results_path}")
    return results_path

