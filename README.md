# mini-rag

A small document question-answering tool in Python, built from scratch to
understand how retrieval-augmented generation (RAG) works.

## What it does
1. Splits a text file into overlapping chunks (size 300, overlap 50).
2. Embeds the chunks and the question with Gemini (`gemini-embedding-001`).
3. Scores every chunk against the question with cosine similarity (written by hand).
4. Returns the best chunk, or "NOT FOUND" if the best score is below a threshold (0.6).

## Results
Test document: a made-up refund policy (4 chunks).

| Question | Best score | Result |
|---|---|---|
| How do I return socks? | 0.806 | chunk 3 (correct) |
| What is the smell fee? | 0.676 | chunk 1 (correct) |
| How are refunds paid? | 0.626 | chunk 2 (correct) |
| What is the capital of the UAE? | 0.476 | not found (correct) |
| Do you sell shoes? | 0.594 | not found (correct) |

The gap between the lowest answerable score (0.626) and the highest
unanswerable one (0.594) is small, so a single threshold is fragile.

## How to run

Requires Python 3.10+ and a Gemini API key from Google AI Studio.

1. Install the library:
   ```
   python -m pip install google-genai
   ```
2. Set your API key as an environment variable (Windows PowerShell), then reopen the terminal:
   ```
   setx GEMINI_API_KEY "your-key-here"
   ```
   On macOS/Linux: `export GEMINI_API_KEY="your-key-here"`
3. Run it:
   ```
   python embed.py
   ```
   It reads `notes.txt`, chunks and embeds it, then prints the best chunk (or NOT FOUND) for the sample questions in the file.

`chunking.py` is a standalone demo of the chunker. It asks for a chunk size and overlap.

Never commit your API key. This project reads it from the environment so it never appears in the code.

## What I learned
- Overlap stops a fact from being cut in half at a chunk boundary.
- Embeddings compare meaning, not exact words.
- Retrieval always returns a "best" chunk, so a threshold is needed.

## Limitations
- Only 4 chunks and 5 test questions.
- Chunking is character-based, so chunks can start or end mid-word.
- No answer generation yet.
- Text is sent to Google to be embedded.

## Next
- Answer with Gemini, citing `[Source: chunk n]`.
- A 15-question test set.
- Compare with a local model through Ollama.
