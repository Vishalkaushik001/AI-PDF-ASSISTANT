
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env
load_dotenv()

# Get API key
API_KEY = os.getenv("AQ.Ab8RN6LmU4-rRKD2SH1ksTH_hc1tz60OLc1VXh5vGXpOlJ-YxQ")

# Initialize Gemini client
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)

# Select the Gemini model
MODEL_NAME = "gemini-3.5-flash-lite"


def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "I could not find relevant information in the PDF."

    # Combine all retrieved document passages
    context_parts = []

    for result in retrieved_chunks:

        chunk = result["chunk"]

        context_parts.append(
            f"""
            SOURCE PAGE: {chunk['page']}
            CHUNK ID: {chunk['chunk_id']}
            CONTENT:
            {chunk['text']}
            """
        )

    context = "\n".join(context_parts)

    prompt = f"""
        You are an accurate and detailed PDF Question-Answering Assistant.

        Your job is to answer the user's question using the supplied PDF passages.

        USER QUESTION:
        {question}

        DOCUMENT PASSAGES:
        {context}

        IMPORTANT ANSWERING INSTRUCTIONS:

        1. Read ALL the supplied passages carefully before answering.

        2. If the user asks for "all", "every", "types", "categories",
        "features", "advantages", "disadvantages", "steps" or "methods",
        provide a COMPLETE list of the relevant items found in the passages.

        3. Do not stop after identifying only one or two items.
        Check every supplied passage for additional relevant information.

        4. If multiple passages describe different items belonging
        to the same topic, combine them into one organized answer.

        5. Explain each identified item separately with:
        - Name
        - Definition
        - Explanation
        - Example, if available in the PDF

        6. Use headings, numbered lists and bullet points.

        7. Provide detailed answers in simple, student-friendly language.

        8. Include the relevant PDF page number for each item wherever possible.

        9. Do not repeat the same item unnecessarily.

        10. Do not invent information or add unsupported items from
            your general knowledge.

        11. If the passages contain only some of the requested items,
            list the available items and clearly mention that the
            retrieved passages may not contain the complete list.

        12. If the answer is not available in the passages,
            clearly say that it was not found in the retrieved PDF content.

        Write a detailed and well-organized answer.

        ANSWER:
"""

    response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt,
    config=types.GenerateContentConfig(
        max_output_tokens=4096
    )
)

    if not response.text:
        return "The AI model returned an empty response. Please try again."

    return response.text
