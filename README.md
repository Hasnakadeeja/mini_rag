# mini_rag

A small document question-answering tool in Python, built from scratch to understand
how retrieval-augmented generation (RAG) works.

I wrote the chunking, similarity search and relevance threshold myself.
Gemini is used for two things only: creating embeddings, and writing the final answer.

## How it works
1. **Chunk:** split a text file into overlapping pieces (size 300, overlap 50).
2. **Embed:** turn each chunk and the question into vectors with `gemini-embedding-001`.
3. **Retrieve:** score every chunk against the question with cosine similarity (hand-written) and take the best one.
4. **Check relevance:** if the best score is below a threshold (0.6), stop. No model call is made.
5. **Generate:** otherwise, send the chunk and the question to Gemini (`gemini-flash-latest`)
   with an instruction to answer only from that text and cite `[Source: chunk n]`.
6. **Retry:** if the API is overloaded (HTTP 503), wait 1, 2, then 4 seconds and try again.

## Results
Test document: a made-up refund policy (`notes.txt`, 4 chunks).

**Retrieval and threshold**

| Question | Best score | Result |
|---|---|---|
| How do I return socks? | 0.806 | chunk 3 (correct) |
| What is the smell fee? | 0.676 | chunk 1 (correct) |
| How are refunds paid? | 0.626 | chunk 2 (correct) |
| What is the capital of the UAE? | 0.476 | blocked (correct) |
| Do you sell shoes? | 0.594 | blocked (correct) |

The gap between the lowest answerable score (0.626) and the highest unanswerable
one (0.594) is small, so a single threshold is fragile.

**Two layers of protection against made-up answers**

| Layer | Where it runs | What I saw |
|---|---|---|
| Relevance threshold (0.6) | My code, before any model call | Blocked the UAE and shipping-cost questions |
| Prompt rule: "answer only from the text" | Gemini | With the threshold set to 0, both questions still got "I can't find that in the document" |

This is a small test (a few questions on one document), not proof that the model never makes things up.

**Example answers**
- *How do I return socks?* gives the return instructions followed by `[Source: chunk 3]`.
- *What is the smell fee?* gives the fee followed by `[Source: chunk 1]`.

## Files
- `embed.py`: the full pipeline (run this one)
- `chunking.py`: a standalone demo of the chunker
- `notes.txt`: the sample document

## How to run
Tested on Python 3.11. You need a Gemini API key from Google AI Studio.

1. Install the library:
   ```
   python -m pip install google-genai
   ```
2. Set your key as an environment variable, then reopen the terminal (Windows PowerShell):
   ```
   setx GEMINI_API_KEY "your-key-here"
   ```
   On macOS/Linux: `export GEMINI_API_KEY="your-key-here"`
3. Run:
   ```
   python embed.py
   ```

Never commit your API key. This project reads it from the environment so it never appears in the code.

## What I learned
- Overlap stops a fact from being cut in half at a chunk boundary.
- Embeddings compare meaning, not exact words.
- Retrieval always returns a "best" chunk, even for unrelated questions, so a relevance check is needed.
- Scores are a rough signal. Related and unrelated questions can land close together.
- Real APIs fail sometimes, so calls need retries and clear error handling.

## Limitations
- One short document (4 chunks) and a handful of test questions.
- Chunking is character-based, so chunks can start or end mid-word.
- Only the single best chunk is used, so an answer split across chunks would be incomplete.
- The threshold was chosen from five questions and is not tuned.
- No automated tests yet.
- Text is sent to Google for embedding and answering, so it is not suitable for private documents.

## Next
- Use the top 3 chunks instead of one.
- A 15-question test set with a results table.
- Compare Gemini with a local model through Ollama.
