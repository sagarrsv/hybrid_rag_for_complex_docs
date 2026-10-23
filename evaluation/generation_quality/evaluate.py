"""Command-line entry point for generation evaluation stages."""
import argparse

from deterministic_metrics import run_offline_evaluation
from correctness_evaluator import evaluate_answer_correctness
from faithfulness_evaluator import (
    prepare_faithfulness_requests,
    submit_faithfulness_batch,
    collect_faithfulness_results,
)
from results_aggregator import combine_evaluation_results

def main():
    """Route each CLI subcommand to its evaluation stage."""
    # Commands are explicit: 'offline' and 'prepare-batch' use no API.
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate generation metadata, reference correctness, '
            'and multimodal faithfulness.'
        )
    )
    commands = parser.add_subparsers(dest='cmd', required=True)

    # Evaluate retrieval/citation metadata and model performance offline.
    offline = commands.add_parser('offline')
    offline.add_argument('--predictions', nargs='+', required=True)
    offline.add_argument('--golden', required=True)
    offline.add_argument('--output', default='evaluation/generation_quality/eval_results/offline.jsonl')
    offline.add_argument(
        '--summary', default='evaluation/generation_quality/eval_results/offline_summary.json'
    )
    offline.set_defaults(func=run_offline_evaluation)

    # Grade existing answers against their references (billable judge calls).
    correctness = commands.add_parser('correctness')
    correctness.add_argument('--predictions', nargs='+', required=True)
    correctness.add_argument('--model', default='openai/gpt-oss-120b')
    correctness.add_argument(
        '--output', default='evaluation/generation_quality/eval_results/correctness.jsonl'
    )
    correctness.add_argument('--max-requests', type=int, default=10)
    correctness.set_defaults(func=evaluate_answer_correctness)

    # Build multimodal Anthropic batch requests without submitting them.
    prepare = commands.add_parser('prepare-batch')
    prepare.add_argument('--predictions', nargs='+', required=True)
    prepare.add_argument('--pdf-dir', required=True)
    prepare.add_argument('--dpi', type=int, default=150)
    prepare.add_argument(
        '--page-numbering', choices=['one', 'zero'], default='one'
    )
    prepare.add_argument('--max-page-chars', type=int, default=14000)
    prepare.add_argument('--model', default='claude-sonnet-5-5')
    prepare.add_argument('--max-output', type=int, default=2400)
    prepare.add_argument('--only-visual', action='store_true')
    prepare.add_argument(
        '--output', default='evaluation/generation_quality/eval_results/claude_requests.json'
    )
    prepare.add_argument(
        '--manifest', default='evaluation/generation_quality/eval_results/claude_manifest.jsonl'
    )
    prepare.set_defaults(func=prepare_faithfulness_requests)

    prepare.add_argument(
    "--qids",
    nargs="+",
    default=None,
    help="Evaluate only the specified question IDs.",
)
    # Submit the prepared batch only when --confirm is provided.
    submit = commands.add_parser('submit-batch')
    submit.add_argument(
        '--requests', default='evaluation/generation_quality/eval_results/claude_requests.json'
    )
    submit.add_argument('--max-requests', type=int, default=0)
    submit.add_argument(
        '--batch-id-file', default='evaluation/generation_quality/eval_results/batch_id.txt'
    )
    submit.add_argument('--confirm', action='store_true')
    submit.set_defaults(func=submit_faithfulness_batch)

    # Fetch judgments for an already-submitted Anthropic batch.
    collect = commands.add_parser('collect-batch')
    collect.add_argument('--batch-id')
    collect.add_argument(
        '--batch-id-file', default='evaluation/generation_quality/eval_results/batch_id.txt'
    )
    collect.add_argument(
        '--manifest', default='evaluation/generation_quality/eval_results/claude_manifest.jsonl'
    )
    collect.add_argument(
        '--output', default='evaluation/generation_quality/eval_results/faithfulness.jsonl'
    )
    collect.set_defaults(func=collect_faithfulness_results)

    # Merge saved judgments without making any model calls.
    merge = commands.add_parser('merge')
    merge.add_argument('--offline', default='evaluation/generation_quality/eval_results/offline.jsonl')
    merge.add_argument(
        '--correctness', default='evaluation/generation_quality/eval_results/correctness.jsonl'
    )
    merge.add_argument(
        '--faithfulness', default='evaluation/generation_quality/eval_results/faithfulness.jsonl'
    )
    merge.add_argument('--output', default='evaluation/generation_quality/eval_results/final.jsonl')
    merge.add_argument(
        '--summary', default='evaluation/generation_quality/eval_results/final_summary.json'
    )
    merge.set_defaults(func=combine_evaluation_results)

    # Execute only the explicitly selected evaluation stage.
    arguments = parser.parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
