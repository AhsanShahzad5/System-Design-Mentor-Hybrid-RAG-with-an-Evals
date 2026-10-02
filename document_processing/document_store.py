import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document


def document_path(output_directory: Path, book_id: str) -> Path:
    return output_directory / f"{book_id}.jsonl"


def save_documents(path: Path, documents: Sequence[Document]) -> None:
    """Atomically serialize LangChain documents as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for document in documents:
                record = {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                temporary_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_documents(path: Path) -> list[Document]:
    """Reconstruct LangChain documents from a JSON Lines artifact."""
    documents: list[Document] = []

    with path.open(encoding="utf-8") as document_file:
        for line_number, line in enumerate(document_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                documents.append(
                    Document(
                        page_content=record["page_content"],
                        metadata=record["metadata"],
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid document at {path}:{line_number}"
                ) from error

    return documents
