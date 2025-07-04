import os
import glob
import faiss
import numpy as np
import PyPDF2
import nltk
import time
import ollama
nltk.download('punkt')
nltk.download('punkt_tab')

CORPUS_DIR = "corpus"          
# INDEX_FILE = "faiss_index.bin" 
# CHUNKS_FILE = "chunks.npy"
INDEX_FILE = os.environ["FAISS_INDEX_FILE"]
CHUNKS_FILE = os.environ["CHUNKS_FILE"]
EMBED_DIM = 768 # OpenAI -> 1536 


def split_text(text, max_length=200):
    if len(text) <= max_length:
        return [text]
    
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def build_faiss_index(corpus_dir: str = CORPUS_DIR,
                      index_file: str = INDEX_FILE,
                      chunks_file: str = CHUNKS_FILE) -> None:
    """
    1. Učitava PDF-ove iz direktorija.
    2. Izvlači tekst sa svake stranice i tokenizira ga na rečenice.
    3. Ako je rečenica duža od 200 znakova, dijeli je na manje chunkove.
    4. Za svaki chunk dohvaća embedding preko OpenAI API-ja.
    5. Ako embedding ne uspije dohvatiti, taj chunk se odmah preskače.
    6. Gradi FAISS indeks (dot-product) i sprema ga zajedno s chunkovima.
    """
    pdf_paths = glob.glob(os.path.join(corpus_dir, "*.pdf"))
    if not pdf_paths:
        print(f"Nema PDF datoteka u direktoriju '{corpus_dir}'.")
        return

    svi_chunkovi = []
    svi_embeddingi = []

    for pdf_file in pdf_paths:
        print(f"Obrada datoteke: {pdf_file}")
        pages_text = extract_text_from_pdf(pdf_file)
        for page in pages_text:
            recenice = nltk.tokenize.sent_tokenize(page)
            for rec in recenice:
                rec = rec.strip()
                if rec:
                    if len(rec) > 200:
                        podchunkovi = split_text(rec, max_length=200)
                        for podchunk in podchunkovi:
                            emb = get_embedding(podchunk)
                            if emb is not None:
                                svi_chunkovi.append(podchunk)
                                svi_embeddingi.append(emb)
                            else:
                                print("Preskačem chunk:")
                                print(podchunk)
                    else:
                        emb = get_embedding(rec)
                        if emb is not None:
                            svi_chunkovi.append(rec)
                            svi_embeddingi.append(emb)
                        else:
                            print("Preskačem chunk:")
                            print(rec)
    if not svi_chunkovi:
        print("Nije pronađen nikakav tekst za kreiranje FAISS indeksa.")
        return

    embedding_array = np.array(svi_embeddingi, dtype=np.float32)
    index = faiss.IndexFlatIP(EMBED_DIM)
    
    print(f"embedding_array.shape = {embedding_array.shape}")
    print(f"embedding_array.dtype = {embedding_array.dtype}")

    index.add(embedding_array)
    print(f"FAISS indeks kreiran sa {len(svi_chunkovi)} chunkova.")

    faiss.write_index(index, index_file)
    np.save(chunks_file, np.array(svi_chunkovi, dtype=object))
    print(f"Indeks spremljen u '{index_file}', a chunkovi u '{chunks_file}'.")

def extract_text_from_pdf(pdf_path: str):
    """
    Čita PDF datoteku i vraća listu tekstova za svaku stranicu.
    """
    texts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts

def get_embedding(text: str):
    if not text.strip():
        return None
    
    try:
        response = ollama.embed(
                model='nomic-embed-text:latest',
                input=text)
        emb = response["embeddings"][0]

        return np.array(emb, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding: {e}. Skipping this chunk:")
        print(text)
        return None

if __name__ == "__main__":
    build_faiss_index()
