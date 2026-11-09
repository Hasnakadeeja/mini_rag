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
7. **[Citation check, if you added it: warn if the model cites a chunk it was not given.]**


## Evaluation
15 questions on one short sample document (`notes.txt`, 4 chunks). I wrote the questions
and chose the expected chunks by reading the text. A question passes if a correct chunk is
among the chunks kept above the threshold (0.6), or if nothing is kept for a question the
document cannot answer.

**Result: [13]/15**

**The two failures** are related questions the document cannot answer:
- *"Can I exchange socks for a different size?"* kept chunks `[3, 0, 2]`
- *"Which polite poem should I recite?"* kept chunk `[3]` (the text mentions a poem but never names one)

A similarity score measures how close the topic is, not whether the answer exists, so the
threshold cannot catch these.

**Score examples** (best score per question)

| Question | Best score | Result |
|---|---|---|
| How do I return socks? | 0.806 | correct chunk |
| What is the smell fee? | 0.676 | correct chunk |
| How are refunds paid? | 0.626 | correct chunk |
| Do you sell shoes? | 0.594 | blocked |
| What is the capital of the UAE? | 0.476 | blocked |

The gap between the lowest answerable score (0.626) and the highest blocked one (0.594) is small.

**Two layers of protection against made-up answers**

| Layer | Where it runs | What I saw |
|---|---|---|
| Relevance threshold (0.6) | My code, before any model call | Blocked unrelated questions and shoes/shipping-cost, but not the two failures above |
| Prompt rule: "answer only from the text" | Gemini | With the threshold at 0, the UAE and shipping questions got "I can't find that in the document". On the two failures above: [paste what Gemini said] |

This is a small test (15 questions, one document), not proof that the model never makes things up.

**Example answers**
- *How do I return socks?* gives the return instructions followed by `[Source: chunk 3]`.
- *What is the smell fee?* gives the fee followed by `[Source: chunk 1, chunk 2]`.

## Files
- `embed.py`: the full pipeline (run this one)
- `chunking.py`: a standalone demo of the chunker
- `notes.txt`: the sample document
- `vectors.json`: generated cache (ignored by Git)


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
- A threshold separates unrelated questions from related ones, but not related questions
  the document cannot answer.

## Limitations
- One short document, 4 chunks, 15 questions. Too small to say how it would perform on real documents.
- I wrote the expected answers myself, so the labels reflect my judgment.
- The pass rule checks that a correct chunk was found. It does not penalise extra, irrelevant
  chunks (for example, "How do I return socks?" kept `[3, 2, 0]`).
- Chunking is character-based, so chunks can start or end mid-word, and the overlap puts some
  sentences in two chunks.
- The cache compares chunk text only. If you change the embedding model, delete `vectors.json`.
- The threshold was chosen from a small number of tests, not tuned.
- Text is sent to Google for embedding and answering, so it is not suitable for private documents.

## Next
- Compare Gemini with a local model through Ollama.
- A larger, more varied test set.
- Measure how much irrelevant text reaches the model.
