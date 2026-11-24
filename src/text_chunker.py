
def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):

    chunks = []
    chunk_id = 1

    # Validate settings
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    # Process every PDF page
    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"].strip()

        if not text:
            continue

        start = 0

        # Divide page text into chunks
        while start < len(text):

            end = min(start + chunk_size, len(text))

            # Try to avoid cutting words in half
            if end < len(text):
                space_position = text.rfind(" ", start, end)

                if space_position > start + int(chunk_size * 0.7):
                    end = space_position

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

            # Move forward while maintaining overlap
            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks