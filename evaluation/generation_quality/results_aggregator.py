"""Merge judge results and write reproducible evaluation reports."""

import collections
import csv
import json
import statistics
from pathlib import Path

from data_loader import load_jsonl_records, save_jsonl_records

# Manual decisions apply only to the named generator and experiment.
# Set an override only after inspecting the retrieved PDF evidence.
# Keep the original Claude judgment in the saved per-question record.
MANUAL_FAITHFULNESS_OVERRIDES = {
    "gemini-3.8-flash": {
        "retrieved": {
            "F-008": "Accepted after manual review of plotted values",
            "F-009": "Approximate graph reading within reasonable tolerance",
            "F-011": "Accepted after manual inspection of figure comparisons",
            "F-012": "Explanation supported by evaluation-protocol evidence",
            "F-015": "Omitted benchmark is completeness, not faithfulness",
            "T-015": "Manual review overrules judge's evidence interpretation",
            "X-040": "Minor overstatement accepted as nonmaterial",
        }
    }
}

# Preserve clear flags for cases that remain partially supported.
MANUAL_REVIEW_NOTES = {
    "gemini-3.8-flash": {
        "retrieved": {
            "F-016": "Incorrect full model-backbone ranking",
            "X-037": "Causal claim about memory efficiency remains unsupported",
        }
    }
}


def _mean(rows, field):
    """Average recorded numeric values; ignore missing values."""
    values = [
        r[field] for r in rows
        if isinstance(r.get(field), (int, float))
        and not isinstance(r[field], bool)
    ]
    return round(statistics.mean(values), 3) if values else None


def _fraction(rows, field):
    """Return numerator, denominator and proportion for boolean data."""
    values = [r[field] for r in rows if isinstance(r.get(field), bool)]
    return (sum(values), len(values), sum(values) / len(values)) if values else (0, 0, None)


def _format_rate(rows, field):
    """Display a rate with its exact evaluated denominator."""
    good, total, value = _fraction(rows, field)
    return f"{100 * value:.1f}% ({good}/{total})" if value is not None else "N/A"


def _format_number(value, decimals=1):
    return f"{value:,.{decimals}f}" if value is not None else "N/A"


def _conditional_correctness(rows):
    """Score correctness only where all annotated gold pages were retrieved."""
    eligible = [r for r in rows if r.get("complete_gold_coverage") is True]
    return _format_rate(eligible, "strict_correct")


def _summarize_group(rows):
    """Provide machine-readable metrics for a group of questions."""
    def rate(field):
        return _fraction(rows, field)[2]

    eligible = [r for r in rows if r.get("complete_gold_coverage") is True]
    answered = [r for r in rows if r.get("predicted_answerable") is True]

    return {
        "count": len(rows),
        "retrieval_complete_coverage": rate("complete_gold_coverage"),
        "answer_correctness": rate("strict_correct"),
        "correctness_given_complete_retrieval": _fraction(
            eligible, "strict_correct"
        )[2],
        "correctness_given_complete_retrieval_count": len(eligible),
        "faithfulness_judge_fully_supported": rate(
            "faithfulness_fully_supported_raw"
        ),
        "faithfulness_adjudicated_fully_supported": rate(
            "faithfulness_fully_supported"
        ),
        "faithfulness_adjudicated_among_answered": _fraction(
            answered, "faithfulness_fully_supported"
        )[2],
        "citation_presence_rate": rate("has_citation"),
        "mean_generation_seconds": _mean(rows, "generation_latency"),
        "mean_total_seconds": _mean(rows, "total_latency"),
        "mean_input_tokens": _mean(rows, "input_tokens"),
        "mean_output_tokens": _mean(rows, "output_tokens"),
        "mean_total_tokens": _mean(rows, "total_tokens"),
    }


def summarize_evaluation_results(rows):
    """Summarize overall and bucket-level evaluation results."""
    groups = collections.defaultdict(list)
    for row in rows:
        for field in (
            "model", "experiment", "evidence_type",
            "difficulty", "reasoning_type",
        ):
            groups[f"{field}={row.get(field)}"].append(row)

        key = (
            f"model={row.get('model')}"
            f"|evidence_type={row.get('evidence_type')}"
        )
        groups[key].append(row)

    return {
        "total": len(rows),
        "overall": _summarize_group(rows),
        "groups": {
            name: _summarize_group(group)
            for name, group in sorted(groups.items())
        },
    }


def _markdown_table(headers, records):
    """Create a compact Markdown table without wide multi-line cells."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(v) for v in record) + " |"
              for record in records]
    return "\n".join(lines)


def _report_sections(rows):
    """Build README-ready tables and optional CSV exports."""
    ordered_groups = [
        ("evidence_type", ["text", "table", "figure"]),
        ("difficulty", ["easy", "medium", "hard"]),
        ("reasoning_type", sorted({r.get("reasoning_type") for r in rows
                                   if r.get("reasoning_type")})),
    ]
    sections = {}

    # Focus the main table on answer quality and evidence coverage.
    overall = [
        ("Questions evaluated", str(len(rows))),
        ("Answer correctness", _format_rate(rows, "strict_correct")),
        ("Correctness given complete retrieval", _conditional_correctness(rows)),
        ("Complete gold-page retrieval", _format_rate(rows, "complete_gold_coverage")),
        ("Faithfulness (Claude, unadjusted)", _format_rate(
            rows, "faithfulness_fully_supported_raw"
        )),
        ("Faithfulness (manually adjudicated)", _format_rate(
            rows, "faithfulness_fully_supported"
        )),
        ("Adjudicated faithfulness, answered only", _format_rate(
            [r for r in rows if r.get("predicted_answerable") is True],
            "faithfulness_fully_supported"
        )),
    ]
    sections["main"] = (["Metric", "Result"], overall)

    # Preserve both correctness and faithfulness within each bucket.
    for field, labels in ordered_groups:
        body = []
        for label in labels:
            group = [r for r in rows if r.get(field) == label]
            if not group:
                continue
            body.append((
                label.replace("_", " ").title(),
                len(group),
                _format_rate(group, "strict_correct"),
                _conditional_correctness(group),
                _format_rate(group, "complete_gold_coverage"),
                _format_rate(group, "faithfulness_fully_supported"),
            ))
        sections[field] = (
            ["Group", "N", "Correctness", "Given retrieval",
             "Retrieval", "Adjudicated faithfulness"],
            body,
        )

    # Keep latency and token accounting outside the main README table.
    performance = []
    for label, group in [
        ("Overall", rows),
        *[(name.title(), [r for r in rows if r.get("evidence_type") == name])
          for name in ("text", "table", "figure")],
    ]:
        performance.append((
            label,
            _format_number(_mean(group, "generation_latency")),
            _format_number(_mean(group, "total_latency")),
            _format_number(_mean(group, "input_tokens"), 0),
            _format_number(_mean(group, "output_tokens"), 0),
            _format_number(_mean(group, "total_tokens"), 0),
        ))
    sections["performance"] = (
        ["Group", "Gen (s)", "Total (s)",
         "Input tokens", "Output tokens", "Total tokens"],
        performance,
    )
    return sections


def write_report_tables(rows, summary_path):
    """Save Markdown and CSV report tables beside the JSON summary."""
    base = Path(summary_path).parent
    base.mkdir(parents=True, exist_ok=True)
    sections = _report_sections(rows)
    titles = {
        "main": "Main evaluation results",
        "evidence_type": "Performance by evidence type",
        "difficulty": "Performance by difficulty",
        "reasoning_type": "Performance by reasoning type",
        "performance": "Latency and token usage",
    }
    parts = ["# Generation evaluation report", ""]
    for name, (headers, records) in sections.items():
        parts.extend([f"## {titles[name]}", "",
                      _markdown_table(headers, records), ""])
        with (base / f"report_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(records)

    parts.extend([
        "## Scoring notes", "",
        "- Answer correctness: reference-based score of 2 (fully correct).",
        "- Complete retrieval: every annotated gold page appears in retrieved pages.",
        "- Given retrieval: correctness among questions with complete gold-page coverage.",
        "- Faithfulness: support of stated claims by retrieved evidence, not answer completeness.",
        "- Manual overrides are limited to the documented QIDs and retain Claude's raw verdict.",
        "- Adjudicated faithfulness over all questions includes abstentions; the answered-only rate is separate.",
        "- Percentages reflect available judgments, not necessarily all N when a judge result is missing.",
        "",
    ])
    path = base / "report_tables.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _index_successful_judgments(path):
    """Index saved judge responses without losing distinct models or experiments."""
    if not path or not Path(path).exists():
        return {}
    index = {}
    for record in load_jsonl_records(path):
        key = (record.get("model"), record.get("experiment"), record.get("qid"))
        index[key] = record
    return index


def _apply_manual_review(row):
    """Record any explicit human adjudication without overwriting Claude."""
    model = row.get("model")
    experiment = row.get("experiment")
    qid = row.get("qid")
    overrides = MANUAL_FAITHFULNESS_OVERRIDES.get(model, {}).get(experiment, {})
    notes = MANUAL_REVIEW_NOTES.get(model, {}).get(experiment, {})

    # Apply overrides only when a real judge judgment exists.
    if isinstance(row.get("faithfulness_fully_supported_raw"), bool):
        if qid in overrides:
            row["faithfulness_fully_supported"] = True
            row["faithfulness_review_status"] = "manual_override_supported"
            row["faithfulness_review_reason"] = overrides[qid]
        elif qid in notes:
            row["faithfulness_review_status"] = "manual_review_retained"
            row["faithfulness_review_reason"] = notes[qid]


def combine_evaluation_results(args):
    """Merge existing scores and write JSONL, JSON, Markdown and CSV reports."""
    offline_rows = list(load_jsonl_records(args.offline))
    correctness = _index_successful_judgments(args.correctness)
    faithfulness = _index_successful_judgments(args.faithfulness)

    for row in offline_rows:
        key = (row["model"], row["experiment"], row["qid"])
        correct_record = correctness.get(key)
        faith_record = faithfulness.get(key)

        # Copy reference-based correctness scores without changing the judge result.
        if correct_record and correct_record.get("status") == "ok":
            judgment = correct_record["judgment"]
            row["correctness_judgment"] = judgment
            row["strict_correct"] = judgment["correctness"] == 2

        # Store raw faithfulness before applying an explicitly documented review.
        if faith_record and faith_record.get("status") == "ok":
            judgment = faith_record["judgment"]
            row["faithfulness_judgment"] = judgment
            raw_supported = judgment.get("overall") == "supported"
            row["faithfulness_fully_supported_raw"] = raw_supported
            row["faithfulness_fully_supported"] = raw_supported
            row["faithfulness_review_status"] = "judge_only"

            # Compute claim-level scores based solely on the original judge claims.
            claims = judgment.get("claims") or []
            supported = sum(c.get("status") == "supported" for c in claims)
            verifiable = sum(
                c.get("status") in ("supported", "unsupported")
                for c in claims
            )
            row["supported_claims"] = supported
            row["verifiable_claims"] = verifiable
            row["claim_support_ratio"] = (
                supported / verifiable if verifiable else None
            )

        # Manual decisions affect only the adjudicated summary field.
        _apply_manual_review(row)

    # Save the merged per-question evaluation and aggregate JSON.
    save_jsonl_records(args.output, offline_rows)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summarize_evaluation_results(offline_rows), indent=2),
        encoding="utf-8",
    )

    # Produce ready-to-copy Markdown tables and separate spreadsheet-friendly CSVs.
    report_path = write_report_tables(offline_rows, summary_path)
    print(f"Merged {len(offline_rows)} records: {args.output}")
    print(f"JSON summary: {summary_path}")
    print(f"README tables: {report_path}")
