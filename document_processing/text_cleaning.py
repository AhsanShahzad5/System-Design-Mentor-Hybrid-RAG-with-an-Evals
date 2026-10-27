"""Clean extraction artifacts out of PyMuPDF Markdown before it is stored.

Each rule targets an artifact seen in the processed books:
  - strikethrough / underline / superscript markup leaked from PDF styling
  - picture-text comment markers and <br> inside OCR'd figure text
  - U+FFFD replacement characters from failed glyph mapping (mostly apostrophes)
  - DDIA-style soft hyphenation "compo‐ nents" split across a line
  - table captions shattered into one-letter italic fragments ("_T a b l_")
"""

import re

_TAGS = re.compile(r"</?(?:u|sup|sub|b|i|em|strong)\s*>", re.IGNORECASE)
_BREAKS = re.compile(r"<br\s*/?>", re.IGNORECASE)
_PICTURE_MARKERS = re.compile(r"<!--\s*(?:Start|End) of picture text\s*-->", re.IGNORECASE)
_SOFT_HYPHEN_SPLIT = re.compile(r"(\w)‐\s+(\w)")
_REPLACEMENT_IN_WORD = re.compile(r"(?<=\w)�(?=\w)")
_SHATTERED_ITALIC_LINE = re.compile(r"^\s*_[^_\n]{1,12}_\s*$", re.MULTILINE)
_SPACES = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_markdown(text: str) -> str:
    text = text.replace("~~", "")
    text = _PICTURE_MARKERS.sub("", text)
    text = _BREAKS.sub(" ", text)
    text = _TAGS.sub("", text)
    text = _SOFT_HYPHEN_SPLIT.sub(r"\1\2", text)
    text = _REPLACEMENT_IN_WORD.sub("'", text)
    text = text.replace("�", " ")
    text = _SHATTERED_ITALIC_LINE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean_title(title: str) -> str:
    """Clean a heading: markup removed, emphasis characters stripped."""
    return clean_markdown(title).strip().strip("*_` ").strip()
