"""
ingest.py
Loads the GigaCorp FAQ text file, splits it into paragraph-level chunks while
tracking the original line numbers and section headers of each chunk (for
accurate citations later), embeds the chunks with a local HuggingFace
sentence-transformer model, and persists a FAISS vector store to disk.

Run this once (or whenever data/gigacorp_faq.txt changes):
    python ingest.py
"""

import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = os.path.join("data", "gigacorp_faq.txt")
INDEX_PATH = "faiss_index"
SOURCE_NAME = "gigacorp_faq.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks_with_line_numbers(path: str) -> list[Document]:
    """
    Splits the FAQ file into paragraph chunks (separated by blank lines),
    tagging each chunk with:
      - the source file name
      - the 1-indexed start/end line numbers in the original file
      - the section header (## ...) it falls under
    This lets the assistant cite an exact source + line range later.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    documents: list[Document] = []
    current_section = "General"
    buffer_lines: list[str] = []
    buffer_start = None

    def flush_buffer(end_line: int):
        if buffer_lines and any(l.strip() for l in buffer_lines):
            text = "".join(buffer_lines).strip()
            if text and not text.startswith("##") and not text.startswith("#"):
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": SOURCE_NAME,
                            "start_line": buffer_start,
                            "end_line": end_line,
                            "section": current_section,
                        },
                    )
                )

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if stripped.startswith("## "):
            # Flush whatever paragraph we were building, then start new section
            flush_buffer(idx - 1)
            buffer_lines = []
            buffer_start = None
            current_section = stripped.replace("## ", "").strip()
            continue

        if stripped.startswith("# "):
            # Top-level title line, skip
            continue

        if stripped == "":
            # Blank line = paragraph boundary
            flush_buffer(idx - 1)
            buffer_lines = []
            buffer_start = None
            continue

        if buffer_start is None:
            buffer_start = idx
        buffer_lines.append(raw_line)

    # Flush any trailing paragraph
    flush_buffer(len(lines))

    return documents


def main():
    print(f"Loading and chunking {DATA_PATH} ...")
    docs = load_chunks_with_line_numbers(DATA_PATH)
    print(f"Created {len(docs)} chunks:")
    for d in docs:
        print(
            f"  [{d.metadata['section']}] lines {d.metadata['start_line']}-{d.metadata['end_line']}: "
            f"{d.page_content[:60].replace(chr(10), ' ')}..."
        )

    print(f"\nLoading embedding model '{EMBEDDING_MODEL}' (downloads on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(INDEX_PATH)
    print(f"\nSaved FAISS index to ./{INDEX_PATH}/")
    print("Ingestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
