import time
from google import genai
import math
from google.genai import errors
import re
import json, os

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




CACHE = "vectors.json"

if os.path.exists(CACHE):
    saved = json.load(open(CACHE, encoding="utf-8"))
    if saved["chunks"] == chunks:
        vectors = saved["vectors"]
        print("loaded from cache")
    else:
        vectors = None
else:
    vectors = None

if vectors is None:
    # your existing embed_content call and the loop that builds vectors go here


    result = client.models.embed_content(
        model="gemini-embedding-001", 
        contents=chunks,
    )

    vectors = []
    for e in result.embeddings:
        vectors.append(e.values)


    print("embedded fresh")
    json.dump({"chunks": chunks, "vectors": vectors}, open(CACHE, "w", encoding="utf-8"))

for i, c in enumerate(chunks):
    print(f"--- chunk {i} ---")
    print(c)

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

def generate_with_retry(prompt, tries=5):
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

def check_citations(answer_text, kept):
    cited = re.findall(r"chunk (\d+)", answer_text)
    bad = []
    for n in cited:
        if int(n) not in kept:
            bad.append(n)
    return bad
"""
test to check working of check_citations
print(check_citations("Fee is $3.50 [Source: chunk 1, chunk 2]", [1, 2, 3]))
print(check_citations("Fee is $3.50 [Source: chunk 9]", [1, 2, 3]))
print(check_citations("I can't find that in the document.", [1, 2, 3]))
"""

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
    reply = generate_with_retry(prompt)
    bad = check_citations(reply, kept)
    if bad:
        reply += f"\n[Warning: cited chunk(s) {bad} were not given to the model]"
    return reply

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


tests = [
("How are refunds paid?", {2}),
("How do I return socks?", {2, 3}),
("What is the capital of the UAE?", None),
("When can I request a refund?", {0}),
("How many seconds do I need to refrain myself from looking at the socks so that I can return it?", {0}),
("Do you sell shoes?", None),
("How much does shipping cost?", None),
("Can I exchange socks for a different size?", None),
("How do I send socks back?", {2, 3}),
("What do I owe if my socks smell?", {1, 2}),
("How do I reset my password?", None),
("What time does the shop open?", None),
("Explain The Smell Clause.", {1, 2}),
("How much is emotional damage fee", {1, 2}),
("Which polite poem should I recite?", None)
# ... 12 more, which you write after reading notes.txt
]

passed = 0
for question, expected in tests:
    matches = top_matches(question)
    kept = [i for score, i in matches if score >= THRESHOLD]
    if expected is None:
        ok = (len(kept) == 0)
    else:
        ok = bool(expected & set(kept))
    print("PASS" if ok else "FAIL", "|", question, "|", kept)
    if ok:
        passed += 1

print(passed, "/", len(tests))

"""
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

"""



