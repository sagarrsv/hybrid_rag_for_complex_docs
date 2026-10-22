"""Reference-based answer quality judging, independent of API provider."""
import json
import re
from pathlib import Path
import time

from data_loader import load_jsonl_records, load_unique_predictions
from dotenv import load_dotenv
load_dotenv()

VERSION = "v1.2-readable"

# This judge compares the generated answer with the gold reference.
# It does not inspect retrieved pages or decide whether citations are grounded.
CORRECTNESS_PROMPT = """You are evaluating the correctness of research-paper QA answers.

Evaluation rules:
1. Compare the generated answer with the question and gold reference.
2. Accept equivalent wording and mathematically equivalent calculations.
3. Require only the details the question actually requests.
4. Penalize incorrect numbers, units, orderings, and added false claims.
5. For an unanswerable question, evaluate whether abstention is correct.
6. Treat the gold reference as authoritative for this evaluation.

Scoring rubric:
- correctness = 0: central answer is incorrect.
- correctness = 1: partly correct, but missing or incorrect key information.
- correctness = 2: all requested essential facts are correct.
- completeness = true only if all requested facts are included.
- factual_error = true if the answer asserts a materially wrong fact.

Respond with ONLY valid JSON containing:
- correctness: integer (0, 1, or 2)
- completeness: boolean
- factual_error: boolean
- missing_facts: array of short strings
- wrong_facts: array of short strings
- rationale: short string
"""


def parse_judge_json(text):
    """Decode the judge response, accepting an optional surrounding markdown JSON fence."""
    # Remove optional fenced-code formatting before decoding JSON.
    text=text.strip()
    text=re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    return json.loads(text)


def evaluate_answer_correctness(args):
    """Judge reference-based correctness of saved answers."""
    # Import the provider SDK only when correctness scoring is requested.
    from groq import Groq

    # The SDK reads GROQ_API_KEY from the environment. No key is hard-coded.
    judge_client = Groq()
    output_path = Path(args.output)

    # Use a compound key because multiple generator models / experiments can
    # contain the same QID. Each should receive its own independent judgment.
    # Resume: skip only successful prior evaluations.
    successful_judgments = set()
    if output_path.exists():
        for previous in load_jsonl_records(output_path):
            if previous.get("status") == "ok":
                successful_judgments.add((
                    previous.get("model"),
                    previous.get("experiment"),
                    previous.get("qid"),
                ))

    submitted_requests = 0
    for prediction in load_unique_predictions(args.predictions):
        evaluation_id = (
            prediction.get("model"),
            prediction.get("experiment"),
            prediction.get("qid"),
        )

        if evaluation_id in successful_judgments:
            continue  # Already evaluated in a previous run.
        if submitted_requests >= args.max_requests:
            break  # Respect the explicit API-request budget.

        # The reference is used only for answer quality. Retrieved pages are
        # deliberately excluded: evidence grounding is judged separately.
        # Give the judge only the question, reference, and generated answer.
        judge_input = {
            "question": prediction["question"],
            "reference_answer": prediction["reference_answer"],
            "generated_answer": prediction.get("prediction", ""),
            "gold_answerable": prediction.get("gold_answerable"),
            "predicted_answerable": prediction.get("predicted_answerable"),
        }

        # Count an attempt BEFORE the network request. Even a failed attempt
        # may consume an API quota; we never retry it automatically.
        # Reserve one local attempt BEFORE the API call; never auto-retry.

        # Space requests apart to stay within 30 RPM.
        if submitted_requests > 0:
            time.sleep(2.1)

        submitted_requests += 1
        try:
            completion = judge_client.chat.completions.create(
                model=args.model,
                temperature=0,
                max_tokens=1200,
                messages=[
                    {"role": "system", "content": CORRECTNESS_PROMPT},
                    {"role": "user", "content": json.dumps(judge_input, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )

            # JSON mode ensures JSON syntax, but not our domain-specific
            # fields or types, so validate the judgment after parsing.
            # Parse JSON and verify the expected field types.
            judgment = parse_judge_json(completion.choices[0].message.content)
            if judgment.get("correctness") not in (0, 1, 2):
                raise ValueError("Judge correctness must be 0, 1, or 2")
            if not isinstance(judgment.get("completeness"), bool):
                raise ValueError("Judge completeness must be a boolean")
            if not isinstance(judgment.get("factual_error"), bool):
                raise ValueError("Judge factual_error must be a boolean")
            for field in ("missing_facts", "wrong_facts"):
                if not isinstance(judgment.get(field), list):
                    raise ValueError(f"Judge {field} must be a list")
            if not isinstance(judgment.get("rationale"), str):
                raise ValueError("Judge rationale must be a string")

            # Keep judge metadata separate from generator metadata.
            result_row = {
                "qid": evaluation_id[2],
                "model": evaluation_id[0],  # Generator model, not judge model.
                "experiment": evaluation_id[1],
                "status": "ok",
                "judge_model": args.model,
                "rubric_version": VERSION,
                "judgment": judgment,
                "judge_tokens": {
                    "input": getattr(completion.usage, "prompt_tokens", None),
                    "output": getattr(completion.usage, "completion_tokens", None),
                },
            }
        except Exception as exc:
            # Preserve failures for investigation; no hidden retry or
            # replacement with a guessed judgment.
            result_row = {
                "qid": evaluation_id[2],
                "model": evaluation_id[0],
                "experiment": evaluation_id[1],
                "status": "error",
                "error": str(exc),
            }

        # Append immediately to make interrupted runs safely resumable.
        # Persist each outcome immediately so interruption loses no progress.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
        print(f"{evaluation_id[2]}: {result_row['status']}")

    print(f"New correctness judge requests: {submitted_requests}")


