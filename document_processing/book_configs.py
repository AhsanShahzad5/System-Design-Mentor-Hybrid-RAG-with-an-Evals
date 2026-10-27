from document_processing.schemas import BookConfig
from settings import BOOKS_DIRECTORY

BOOK_1 = BookConfig(
    book_id="designing-ml-systems-chip-huyen",
    pdf_path=BOOKS_DIRECTORY / "1-Designing ML Systems - Chip Huyen.pdf",
    title="Designing Machine Learning Systems",
    section_title_overrides={
        (1, "A NOTE FOR EARLY RELEASE READERS"): "Introduction",
    },
)

BOOK_2 = BookConfig(
    book_id="system-design-interview-alex-xu",
    pdf_path=BOOKS_DIRECTORY / "2-System Design Interview by Alex Xu.pdf",
    title="System Design Interview: An Insider's Guide",
)

BOOK_3 = BookConfig(
    book_id="designing-data-intensive-applications-martin-kleppmann",
    pdf_path=BOOKS_DIRECTORY
    / "3-Designing Data Intensive Applications by Martin Kleppmann.pdf",
    title="Designing Data-Intensive Applications",
)

BOOK_4 = BookConfig(
    book_id="machine-learning-system-design-interview-aminian-xu",
    pdf_path=BOOKS_DIRECTORY
    / "4-Machine Learning System Design Interview (Ali Aminian, Alex Xu).pdf",
    title="Machine Learning System Design Interview",
)

BOOK_5 = BookConfig(
    book_id="ai-engineering-chip-huyen",
    pdf_path=BOOKS_DIRECTORY / "5-AI Engineering_Chip Huyen.pdf",
    title="AI Engineering",
)

BOOKS = (BOOK_1, BOOK_2, BOOK_3, BOOK_4, BOOK_5)

# Human-readable titles for source labels, keyed by the stored book_id.
BOOK_TITLES = {book.book_id: book.title for book in BOOKS if book.title}
