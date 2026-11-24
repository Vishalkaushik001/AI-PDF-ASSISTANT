
from pypdf import PdfReader


def extract_pdf_text(pdf_path):

    # Open the PDF document
    reader = PdfReader(pdf_path)

    # Store extracted text
    extracted_pages = []

    # Read every page
    for page_number, page in enumerate(reader.pages, start=1):

        # Extract text from current page
        text = page.extract_text()

        if text:
            extracted_pages.append({
                "page": page_number,
                "text": text
            })

    return extracted_pages
