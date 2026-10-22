"""Run a few prepared faithfulness requests through Claude's regular API."""

import argparse
import json
from pathlib import Path

from data_loader import load_jsonl_records, save_jsonl_records
from correctness_evaluator import parse_judge_json

VALID_STATUSES = {"supported", "partial", "unsupported", "unverifiable"}


def test_faithfulness_requests(args):
    """Judge a small number of prepared questions without using Batch API."""
    from anthropic import Anthropic

    # Load requests already built by prepare-batch, including page evidence.
    requests = json.loads(Path(args.requests).read_text(encoding="utf-8"))
    manifest = {
        row["custom_id"]: row
        for row in load_jsonl_records(args.manifest)
    }

    # Keep successful results so a second test never spends credits twice.
    existing = load_jsonl_records(args.output) if Path(args.output).exists() else []
    completed = {
        row["custom_id"]
        for row in existing
        if row.get("status") == "ok"
    }
    pending = [
        req for req in requests
        if req["custom_id"] not in completed
    ][:args.max_requests]

    print(f"Pending requests selected: {len(pending)}")
    for req in pending:
        meta = manifest.get(req["custom_id"], {})
        print(" -", meta.get("qid", req["custom_id"]))

    # Prevent accidental paid requests when inspecting the test setup.
    if not args.confirm:
        print("DRY RUN: Add --confirm to call Claude.")
        return

    client = Anthropic()
    for req in pending:
        custom_id = req["custom_id"]
        meta = manifest.get(custom_id, {})
        print(f"Judging {meta.get('qid', custom_id)} ...", flush=True)

        # Send the identical prompt, retrieved pages, and model parameters
        # that the asynchronous batch submission would send.
        try:
            message = client.messages.create(**req["params"])
            response_text = "\n".join(
                block.text for block in message.content
                if block.type == "text"
            )

            # Parse the same claim-level JSON used by the batch collector.
            judgment = parse_judge_json(response_text)
            if (
                not isinstance(judgment.get("claims"), list)
                or judgment.get("overall") not in VALID_STATUSES
            ):
                raise ValueError("Invalid faithfulness judgment schema")

            record = {
                "custom_id": custom_id,
                **meta,
                "status": "ok",
                "judgment": judgment,
                "judge_usage": message.usage.model_dump(),
            }
            print("Result:", judgment.get("overall"))
        except Exception as exc:
            # Save the failure; never auto-retry a potentially billable call.
            record = {
                "custom_id": custom_id,
                **meta,
                "status": "error",
                "error": str(exc),
            }
            print("Error:", exc)

        # Persist each result immediately, even if a later call fails.
        existing = [r for r in existing if r.get("custom_id") != custom_id]
        existing.append(record)
        save_jsonl_records(args.output, existing)

    print(f"Saved evaluation results: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--qids",nargs="+",default=None)
    args = parser.parse_args()
    if args.max_requests < 1 or args.max_requests > 3:
        parser.error("--max-requests must be between 1 and 3 for this test")
    test_faithfulness_requests(args)


if __name__ == "__main__":
    main()
