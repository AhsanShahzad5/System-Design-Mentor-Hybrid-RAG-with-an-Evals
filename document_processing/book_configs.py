from pathlib import Path

from document_processing.schemas import BookConfig

BOOK_1 = BookConfig(
    book_id="designing-ml-systems-chip-huyen",
    pdf_path=Path("books_data/1-Designing ML Systems - Chip Huyen.pdf"),
    section_title_overrides={
        (1, "A NOTE FOR EARLY RELEASE READERS"): "Introduction",
    },
)

BOOK_2 = BookConfig(
    book_id="system-design-interview-alex-xu",
    pdf_path=Path("books_data/2-System Design Interview by Alex Xu.pdf"),
)

BOOK_3 = BookConfig(
    book_id="designing-data-intensive-applications-martin-kleppmann",
    pdf_path=Path(
        "books_data/3-Designing Data Intensive Applications by Martin Kleppmann.pdf"
    ),
)

BOOK_4 = BookConfig(
    book_id="machine-learning-system-design-interview-aminian-xu",
    pdf_path=Path(
        "books_data/4-Machine Learning System Design Interview (Ali Aminian, Alex Xu).pdf"
    ),
)

BOOK_5 = BookConfig(
    book_id="ai-engineering-chip-huyen",
    pdf_path=Path("books_data/5-AI Engineering_Chip Huyen.pdf"),
)

BOOKS = (BOOK_1, BOOK_2, BOOK_3, BOOK_4, BOOK_5)
