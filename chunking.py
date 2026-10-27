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




def main():
    text = open("notes.txt", encoding="utf-8").read()
    size = int(input("Enter size: "))
    overlap = int(input("Enter the overlap count: "))
    chunk = chunk_text(text, size, overlap)
    print(chunk)
    print(len(chunk))

main()