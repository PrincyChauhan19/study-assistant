# Study Assistant

An AI study assistant that answers questions from your own notes and quizzes you on them. Upload your lecture notes as PDFs, ask anything, and get answers with the exact pieces of your notes they came from.

Built with Python, Streamlit, sentence-transformers, and the Gemini API.

## Features

- **Chat with your notes:** ask a question and get an answer based only on your PDFs.
- **Sources shown:** every answer lists the pieces of your notes it used, so you can check it.
- **Quiz generator:** creates multiple-choice questions from your notes (all topics or one topic), then scores you and explains each answer.
- **Handles busy servers:** retries automatically when the AI service is temporarily overloaded.

## Screenshots

   ![Chat screen](Screenshot%202026-11-12%20053129.png)
   ![Quiz screen](Screenshot%202026-11-12%20053438.png)

## How it works (RAG)

1. **Read and split:** text is extracted from the PDFs and cut into small overlapping chunks.
2. **Embed:** each chunk is turned into a vector with the `all-MiniLM-L6-v2` model, which runs locally.
3. **Retrieve:** your question is embedded the same way, and the most similar chunks are found with cosine similarity.
4. **Generate:** only those chunks are sent to Gemini, which is told to answer from them alone.

## Tech stack

| Part | Tool |
| --- | --- |
| Frontend | Streamlit |
| Backend logic | Python |
| Embeddings | sentence-transformers |
| Language model | Gemini API |
| PDF reading | pypdf |

## Run it yourself

1. Install Python 3.10 or newer.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Get a free API key from Google AI Studio and set it (Windows PowerShell):
   ```
   $env:GEMINI_API_KEY="your-key"
   ```
4. Start the app:
   ```
   python -m streamlit run app.py
   ```

## Limitations and next steps

- Notes are kept in memory, so they must be uploaded again after a restart. Next: a vector database (ChromaDB).
- Scanned or image-only PDFs cannot be read yet. Next: OCR support.

## Author

Princy Chauhan, B.Tech Computer Science Engineering, Banasthali Vidyapith.
