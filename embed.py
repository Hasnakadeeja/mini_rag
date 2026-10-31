import time
from google import genai
import math
from google.genai import errors
THRESHOLD = 0.6

def chunk_text(text, size, overlap):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    
    chunks = []
    
    step = size - overlap
    for start in range(0, len(text), step):
        chunks.append(text[start:(start+size)])
        if (start+size) >= len(text):
            break
    return chunks

client = genai.Client()

text = open("notes.txt", encoding="utf-8").read()
chunks = chunk_text(text, 300, 50)

result = client.models.embed_content(
    model="gemini-embedding-001", 
    contents=chunks,
)

vectors = []
for e in result.embeddings:
    vectors.append(e.values)

print(len(chunks))
print(len(vectors))
print(len(vectors[0]))

""""
start_time = time.perf_counter()
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks,
)
print("seconds:", time.perf_counter() - start_time)
"""

def cosine_similarity(a, b):
    dot = 0
    for x, y in zip(a, b):
        dot += x*y
    len_a = math.sqrt(sum(x * x for x in a))
    len_b = math.sqrt(sum(y * y for y in b))
    return dot / (len_a * len_b)

def best_match(question):
    q_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
    )
    q_vector = q_result.embeddings[0].values
    scores = []
    for i in range(len(vectors)):
        scores.append(cosine_similarity(q_vector, vectors[i]))

    best = scores.index(max(scores))
    return best, scores[best]

def generate_with_retry(prompt, tries=3):
    for attempt in range(tries):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
            )
            return response.text
        except errors.ServerError:
            wait = 2 ** attempt        # waits 1, 2, then 4 seconds
            print("Server busy, retrying in", wait, "seconds...")
            time.sleep(wait)
    return "The model is busy right now. Please try again later."



def answer(question):
    matches = top_matches(question)

    kept = []
    for score, i in matches:
        if score >= THRESHOLD:
            kept.append(i)

    if len(kept) == 0:
        return "Not found in the document."

    context = ""
    for i in kept:
        context += f"[chunk {i}]\n{chunks[i]}\n\n"

    prompt = f"""Answer the question using ONLY the text below.
If the text does not contain the answer, say "I can't find that in the document."
Cite the chunk numbers you used, like [Source: chunk 3].

Text:
{context}
Question: {question}
"""
    return generate_with_retry(prompt)

"""
questions = [
    "How do I return socks?",
    "What is the smell fee?",
    "How are refunds paid?",
    "What is the capital of the UAE?",
    "Do you sell shoes?",
]

for q in questions:
    best, score = best_match(q)
    if score < THRESHOLD:
        print(q, "-> NOT FOUND", round(score, 4))
    else:
        print(q, "-> chunk", best, round(score, 4))
"""
def top_matches(question, k=3):
    q_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
    )
    q_vector = q_result.embeddings[0].values

    pairs = []
    for i in range(len(vectors)):
        score = cosine_similarity(q_vector, vectors[i])
        pairs.append((score, i))

    pairs.sort(reverse=True)
    return pairs[:k]

questions = [
    "How do I return socks?",
    "What is the smell fee?",
    "What is the capital of the UAE?",
    "How much does shipping cost?",
]

for q in questions:
    print("Q:", q)
    print("A:", answer(q))
    print()

