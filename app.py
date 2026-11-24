
import streamlit as st

from src.chatbot import generate_answer
from src.pdf_reader import extract_pdf_text
from src.text_chunker import chunk_pages
from src.embeddings import generate_embeddings
from src.vector_store import create_vector_store, search_vector_store


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Question-Answering Assistant")

st.write(
    "Upload a PDF, process its content, "
    "and ask questions based on the document."
)


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "index" not in st.session_state:
    st.session_state.index = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "pages" not in st.session_state:
    st.session_state.pages = []


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    # Check whether a new PDF has been uploaded
    if st.session_state.processed_file != pdf_bytes:

        # Reset previous document data
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.pages = []

        try:

            with open("uploaded_document.pdf", "wb") as file:
                file.write(pdf_bytes)

            with st.spinner("Extracting PDF text..."):

                pages = extract_pdf_text("uploaded_document.pdf")

            if not pages:
                st.error(
                    "No readable text found in this PDF. "
                    "It may be a scanned document."
                )

            else:

                st.session_state.pages = pages
                st.session_state.processed_file = pdf_bytes

                st.success("PDF processed successfully!")

        except Exception as error:

            st.error(f"PDF processing failed: {error}")

    # Read pages from session state
    pages = st.session_state.pages

    if pages:

        st.success("PDF is ready!")

        st.write("Total pages:", len(pages))


        # --------------------------------------------------
        # CHUNK CONFIGURATION
        # --------------------------------------------------

        st.subheader("Text Chunking Configuration")

        col1, col2 = st.columns(2)

        with col1:

            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100
            )

        with col2:

            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=int(chunk_size) - 1,
                value=min(200, int(chunk_size) - 1),
                step=50
            )


        # --------------------------------------------------
        # GENERATE CHUNKS
        # --------------------------------------------------

        if st.button("Generate Chunks"):

            try:

                with st.spinner("Generating text chunks..."):

                    st.session_state.chunks = chunk_pages(
                        pages,
                        int(chunk_size),
                        int(chunk_overlap)
                    )

                # Reset old vector database
                st.session_state.embeddings = None
                st.session_state.index = None

                st.success("Chunks generated successfully!")

            except Exception as error:

                st.error(f"Chunk generation failed: {error}")


        chunks = st.session_state.chunks


        # --------------------------------------------------
        # DISPLAY CHUNKS
        # --------------------------------------------------

        if chunks:

            st.write("Total chunks:", len(chunks))

            with st.expander("View PDF Chunks"):

                for chunk in chunks:

                    st.write(
                        f"Chunk {chunk['chunk_id']} | "
                        f"Page {chunk['page']}"
                    )

                    st.write(chunk["text"])

                    st.divider()


            # --------------------------------------------------
            # CREATE FAISS VECTOR DATABASE
            # --------------------------------------------------

            if st.button("Create FAISS Vector Database"):

                try:

                    with st.spinner(
                        "Generating embeddings and building index..."
                    ):

                        embeddings = generate_embeddings(chunks)

                        index = create_vector_store(embeddings)

                        st.session_state.embeddings = embeddings
                        st.session_state.index = index

                    st.success(
                        "FAISS vector database created successfully!"
                    )

                except Exception as error:

                    st.error(
                        f"Vector database creation failed: {error}"
                    )


# --------------------------------------------------
# VECTOR DATABASE INFORMATION
# --------------------------------------------------

index = st.session_state.index
chunks = st.session_state.chunks

if index is not None:

    st.divider()

    st.subheader("Vector Database Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Stored Vectors", index.ntotal)

    with col2:
        st.metric("Vector Dimensions", index.d)


# --------------------------------------------------
# CHAT WITH YOUR PDF
# --------------------------------------------------

st.divider()

st.subheader("💬 Chat With Your PDF")


# Show chat only when the vector database is ready
if index is not None and index.ntotal > 0 and chunks:

    question = st.text_input(
        "Ask a question about your PDF",
        placeholder="What is machine learning?"
    )

    top_k = st.slider(                  
        "Number of relevant chunks",
        min_value=1,
        max_value=min(10, index.ntotal),
        value=min(7, index.ntotal)
    )


    if st.button("Generate AI Answer"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:

                # ------------------------------------------
                # SEMANTIC SEARCH
                # ------------------------------------------

                with st.spinner("Searching your PDF..."):

                    query_embedding = generate_embeddings([
                        {"text": question}
                    ])

                    results = search_vector_store(
                        index,
                        query_embedding[0],
                        chunks,
                        top_k
                    )


                # ------------------------------------------
                # GENERATE AI ANSWER
                # ------------------------------------------

                if not results:

                    st.warning(
                        "No relevant information was retrieved."
                    )

                else:

                    with st.spinner(
                        "Gemini is generating your answer..."
                    ):

                        answer = generate_answer(
                            question,
                            results
                        )

                    st.subheader("🤖 AI Assistant Answer")

                    st.markdown(answer)

                    st.divider()


                    # --------------------------------------
                    # DISPLAY RETRIEVED SOURCES
                    # --------------------------------------

                    st.subheader("📚 Retrieved Sources")

                    for result in results:

                        chunk = result["chunk"]

                        with st.expander(
                            f"Page {chunk['page']} | "
                            f"Similarity: {result['score']:.4f}"
                        ):

                            st.write(chunk["text"])


            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )

else:

    st.info(
        "Please upload a PDF, generate chunks, "
        "and create the FAISS vector database "
        "to start chatting."
    )
