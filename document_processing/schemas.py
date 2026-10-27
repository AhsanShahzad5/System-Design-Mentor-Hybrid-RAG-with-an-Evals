from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BookConfig:
    book_id: str
    pdf_path: Path
    title: str = ""
    section_title_overrides: dict[tuple[int, str], str] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    start_page: int
