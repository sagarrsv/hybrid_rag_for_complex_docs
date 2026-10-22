"""Shared JSONL and prediction loading helpers."""
import json
from pathlib import Path

def load_jsonl_records(path):
    """Read valid JSON objects from a JSONL file."""
    # Stream one line at a time to support large prediction files.
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            # Report the source line when JSON parsing fails.
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'{path}:{line_number}: {exc}'
                ) from exc


def save_jsonl_records(path, rows):
    """Save evaluation rows as JSONL."""
    # Ensure the output directory exists.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Each question gets its own record for later merging.
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def page_identity(page):
    """Return the unique document and page identifier."""
    return (
        str(page.get('doc_id', '')),
        int(page.get('page_num', -1)),
    )


def load_unique_predictions(paths):
    """Load completed predictions, keeping one per model and question."""
    # Combine multiple experiment files using a compound identity.
    predictions = {}
    for path in paths:
        for record in load_jsonl_records(path):
            if record.get('status') != 'completed':
                continue

            identity = (
                record.get('model'),
                record.get('experiment'),
                record.get('qid'),
            )

            # Warn when the same experiment contains conflicting answers.
            previous = predictions.get(identity)
            if previous and previous.get('prediction') != record.get('prediction'):
                print(f'WARNING: duplicate answer, latest wins: {identity}')
            predictions[identity] = record

    return list(predictions.values())


