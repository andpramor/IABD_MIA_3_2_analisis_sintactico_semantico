import spacy
from spacytextblob.spacytextblob import (
    SpacyTextBlob,
)  # Importación necesaria para registrar el componente
import pandas as pd
from collections import Counter
# Los resultados de análisis de sentimiento son malísimos, así que vamos a pasar por traducir a inglés antes del análisis.
from deep_translator import GoogleTranslator

# --- PASO 1: Preparación ---
try:
    nlp = spacy.load("es_core_news_lg")
except OSError:
    import es_core_news_lg

    nlp = es_core_news_lg.load()

# Necesitamos importar la clase SpacyTextBlob aunque no la usemos directamente, ya que su importación registra el componente en spaCy. Luego, verificamos si el componente ya está registrado para evitar errores:
if "spacytextblob" not in nlp.pipe_names:
    nlp.add_pipe("spacytextblob")

# Lectura del archivo
with open("data.txt", "r", encoding="utf-8") as f:
    comments = [line.strip() for line in f if line.strip()]

# --- PASO 2: Análisis Morfosintáctico (Ejemplo de una frase) ---
print("--- ANÁLISIS MORFOSINTÁCTICO ---")
example_doc = nlp(comments[0])
for token in list(example_doc.sents)[0]:  # Analizamos la primera frase
    print(f"{token.text:<12} | POS: {token.pos_:<6} | DEP: {token.dep_}")

# --- PASO 3: Extracción de Entidades (NER) ---
print("\n--- ENTIDADES NOMBRADAS ---")
extracted_entities = []
for text in comments:
    doc = nlp(text)
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.label_))
print(pd.DataFrame(extracted_entities, columns=["Texto", "Tipo"]).drop_duplicates())

# --- PASO 4: Análisis de Sentimiento ---
print("\n--- ANÁLISIS DE SENTIMIENTO ---")
translator = GoogleTranslator(source='es', target='en')

for i, text in enumerate(comments):
    # Traducimos antes de procesar con spaCy
    text_en = translator.translate(text)
    doc_en = nlp(text_en) # Cuidado: nlp debería ser un modelo inglés si quieres ser riguroso
    
    polarity = doc_en._.blob.polarity
    sentiment = "Positivo" if polarity > 0.1 else "Negativo" if polarity < -0.1 else "Neutro"
    print(f"Comentario {i+1}: {sentiment} (Score: {polarity:.2f})")

# --- PASO 5: Tópicos (Estrategia A: Sustantivos) ---
print("\n--- TÓPICOS PRINCIPALES ---")
nouns = []
for text in comments:
    doc = nlp(text)
    # Filtramos por sustantivos que no sean palabras comunes (stopwords)
    nouns.extend([t.lemma_.lower() for t in doc if t.pos_ == "NOUN" and not t.is_stop])

top_5 = Counter(nouns).most_common(5)
for topic, count in top_5:
    print(f"Tema: {topic.capitalize()} ({count} menciones)")
