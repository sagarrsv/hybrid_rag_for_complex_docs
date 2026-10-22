"""Offline evidence coverage, citation, abstention and usage metrics."""
import json
from pathlib import Path

from data_loader import (
    load_unique_predictions, page_identity, save_jsonl_records,
)
from results_aggregator import summarize_evaluation_results

def evaluate_metadata_and_usage(record, gold):
    """Score retrieval, citations, abstention, and timing without an API."""
    # Compare the actual retrieved pages against annotated gold pages.
    retrieved_pages = record.get('retrieved_pages') or []
    gold_pages = (
        gold.get('gold_pages')
        or record.get('gold_pages')
        or []
    )
    retrieved_ids = {page_identity(p) for p in retrieved_pages}
    gold_ids = {page_identity(p) for p in gold_pages}

    # Map retrieved pages to the citation IDs shown in the model prompt.
    page_to_citation = {
        page_identity(page): f'P{index}'
        for index, page in enumerate(retrieved_pages, 1)
    }

    # Citation validity checks only page existence and identifier matching.
    # Evidence support is evaluated separately by the multimodal judge.
    citations = record.get('citations') or []
    citation_checks = []
    for citation in citations:
        expected_id = page_to_citation.get(page_identity(citation))
        citation_checks.append({
            'citation': citation,
            'valid_page': expected_id is not None,
            'valid_id_mapping': bool(
                expected_id and citation.get('citation_id') == expected_id
            ),
            'expected_id': expected_id,
        })

    # Evaluate whether the model correctly answered or abstained.
    gold_answerable = bool(
        gold.get('answerable', record.get('gold_answerable', True))
    )
    predicted_answerable = record.get('predicted_answerable')
    abstention_correct = (
        gold_answerable == predicted_answerable
        if isinstance(predicted_answerable, bool)
        else None
    )

    # Use actual model token metadata, without cost estimation.
    usage = record.get('usage') or {}

    # Sum latency only when all three stages were recorded.
    retrieval_time = record.get('retrieval_latency')
    context_time = record.get('context_latency')
    generation_time = record.get('generation_latency')
    time_parts = [retrieval_time, context_time, generation_time]
    total_time = (
        sum(time_parts)
        if all(isinstance(value, (int, float)) for value in time_parts)
        else None
    )

    # Keep one auditable row per generated answer.
    return {
        'qid': record['qid'],
        'model': record.get('model'),
        'experiment': record.get('experiment'),
        'evidence_type': gold.get('evidence_type'),
        'difficulty': gold.get('difficulty'),
        'reasoning_type': gold.get('reasoning_type'),
        'prediction': record.get('prediction'),
        'reference_answer': gold.get('answer'),
        'question': gold.get('question'),
        'retrieved_count': len(retrieved_pages),
        'gold_count': len(gold_ids),
        'gold_pages_retrieved': len(gold_ids & retrieved_ids),
        'complete_gold_coverage': (
            gold_ids.issubset(retrieved_ids) if gold_ids else None
        ),
        'gold_page_recall': (
            len(gold_ids & retrieved_ids) / len(gold_ids)
            if gold_ids else None
        ),
        'citation_count': len(citations),
        'citation_checks': citation_checks,
        'all_citations_valid': (
            all(
                check['valid_page'] and check['valid_id_mapping']
                for check in citation_checks
            ) if citation_checks else None
        ),
        'has_citation': bool(citation_checks),
        'gold_answerable': gold_answerable,
        'predicted_answerable': predicted_answerable,
        'abstention_correct': abstention_correct,
        'input_tokens': usage.get('input_tokens'),
        'output_tokens': usage.get('output_tokens'),
        'total_tokens': usage.get('total_tokens'),
        'retrieval_latency': retrieval_time,
        'context_latency': context_time,
        'generation_latency': generation_time,
        'total_latency': total_time,
        'generation_mode': record.get('generation_mode'),
        'page_count': record.get('page_count'),
        'image_count': record.get('image_count'),
    }


def run_offline_evaluation(args):
    """Save deterministic per-question evaluations and summary metrics."""
    # Index gold records by QID for a stable join.
    with open(args.golden, encoding='utf-8') as handle:
        gold_by_id = {item['qid']: item for item in json.load(handle)}

    # Skip predictions absent from the selected golden dataset.
    evaluations = []
    for prediction in load_unique_predictions(args.predictions):
        qid = prediction['qid']
        if qid not in gold_by_id:
            print(f'WARNING: {qid} not in gold dataset, skipping')
            continue
        evaluations.append(
            evaluate_metadata_and_usage(prediction, gold_by_id[qid])
        )

    # Persist both individual judgments and their aggregate statistics.
    save_jsonl_records(args.output, evaluations)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summarize_evaluation_results(evaluations), indent=2),
        encoding='utf-8',
    )
    print(f'Wrote {len(evaluations)} evaluations to {args.output}')
    print(f'Summary: {args.summary}')


