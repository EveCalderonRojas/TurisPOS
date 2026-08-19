"""
rag_utils.py
============
Funciones del pipeline RAG para el Chatbot Turístico de Costa Rica.
Incluye chunking, generación de embeddings, indexación FAISS y búsqueda semántica.
"""

import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Configuración
MODELO_EMBEDDINGS  = 'intfloat/multilingual-e5-small'
CACHE_EMBEDDINGS   = '../../data/processed/embeddings_cache.pkl'
CACHE_FAISS        = '../../data/processed/faiss_index.bin'
CACHE_CHUNKS       = '../../data/processed/chunks_cache.pkl'
TOP_K              = 5

# Modelo de embeddings
def cargar_modelo_embeddings():
    """
    Carga el modelo de embeddings multilingual-e5-small.
    Se descarga la primera vez y queda cacheado por sentence-transformers.
    """
    print(f'Cargando modelo de embeddings: {MODELO_EMBEDDINGS}')
    return SentenceTransformer(MODELO_EMBEDDINGS)


# Chunking
def chunking_por_resena(df, col_texto='comentarios_espanol'):
    """
    Estrategia: cada reseña completa es un chunk.
    Preserva todos los metadatos por chunk.
    Retorna una lista de dicts con texto y metadatos.
    """
    chunks = []
    for _, fila in df.iterrows():
        texto = fila[col_texto]
        if pd.isna(texto) or str(texto).strip() == '':
            continue
        chunks.append({
            'texto'     : str(texto).strip(),
            'lugar'     : fila['lugar'],
            'categoria' : fila['categoria'],
            'calificacion': fila['calificacion'],
            'polaridad' : fila['polaridad'],
            'fuente'    : fila['fuente'],
            'estrategia': 'por_resena'
        })
    return chunks


def chunking_por_oracion(df, col_texto='comentarios_espanol'):
    """
    Estrategia: divide cada reseña en oraciones individuales.
    Cada oración es un chunk con los mismos metadatos de su reseña.
    Retorna una lista de dicts con texto y metadatos.
    """
    chunks = []
    for _, fila in df.iterrows():
        texto = fila[col_texto]
        if pd.isna(texto) or str(texto).strip() == '':
            continue
        oraciones = [o.strip() for o in str(texto).split('.') if len(o.strip()) > 10]
        for oracion in oraciones:
            chunks.append({
                'texto'     : oracion,
                'lugar'     : fila['lugar'],
                'categoria' : fila['categoria'],
                'calificacion': fila['calificacion'],
                'polaridad' : fila['polaridad'],
                'fuente'    : fila['fuente'],
                'estrategia': 'por_oracion'
            })
    return chunks


# Embeddings
def generar_embeddings(chunks, modelo, cache_path=CACHE_EMBEDDINGS):

    if os.path.exists(cache_path):
        print(f'Cargando embeddings desde cache: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f'Generando embeddings para {len(chunks)} chunks...')
    textos = [c['texto'] for c in chunks]

    # multilingual-e5-small requiere prefijo "query: " o "passage: "
    textos_con_prefijo = [f'passage: {t}' for t in textos]
    embeddings = modelo.encode(textos_con_prefijo, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype('float32')

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f'Embeddings guardados en: {cache_path}')
    return embeddings


# ── Índice FAISS ──────────────────────────────────────────────────────────────
def construir_indice_faiss(embeddings, index_path=CACHE_FAISS):
    """
    Construye un índice FAISS con búsqueda por similitud coseno.
    Normaliza los vectores antes de indexar para usar IndexFlatIP como coseno.
    Guarda el índice en disco.
    Retorna el índice FAISS.
    """
    if os.path.exists(index_path):
        print(f'Cargando índice FAISS desde: {index_path}')
        return faiss.read_index(index_path)

    print('Construyendo índice FAISS...')
    embeddings_norm = embeddings.copy()
    faiss.normalize_L2(embeddings_norm)

    dimension = embeddings_norm.shape[1]
    indice = faiss.IndexFlatIP(dimension)
    indice.add(embeddings_norm)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(indice, index_path)
    print(f'Índice FAISS guardado en: {index_path}')
    return indice


def guardar_chunks(chunks, cache_path=CACHE_CHUNKS):

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f'Chunks guardados en: {cache_path}')


def cargar_chunks(cache_path=CACHE_CHUNKS):

    with open(cache_path, 'rb') as f:
        return pickle.load(f)


# Búsqueda semántica
def buscar_chunks(pregunta, modelo, indice, chunks, top_k=TOP_K,
                  filtro_categoria=None, filtro_polaridad=None):

    # Embedding de la pregunta con prefijo de query
    embedding_pregunta = modelo.encode([f'query: {pregunta}'])
    embedding_pregunta = np.array(embedding_pregunta).astype('float32')
    faiss.normalize_L2(embedding_pregunta)

    # Buscar más resultados de los necesarios si hay filtros
    k_busqueda = top_k * 5 if (filtro_categoria or filtro_polaridad) else top_k
    scores, indices = indice.search(embedding_pregunta, k_busqueda)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk['score'] = float(score)

        # Aplicar filtros
        if filtro_categoria and chunk['categoria'] != filtro_categoria:
            continue
        if filtro_polaridad and chunk['polaridad'] != filtro_polaridad:
            continue

        resultados.append(chunk)
        if len(resultados) >= top_k:
            break

    return resultados


# Construcción del prompt
PROMPT_SISTEMA = """Eres TuriGuía, un experto en turismo de Costa Rica especializado en 
reseñas de viajeros. Tu conocimiento proviene exclusivamente de reseñas reales de Google Maps 
sobre parques nacionales, restaurantes y alojamientos costarricenses.

Reglas que debes seguir:
- Responde SIEMPRE en español, de forma amigable y natural.
- Fundamenta tus respuestas en las reseñas del contexto proporcionado.
- Si no encontrás información relevante en el contexto, decilo claramente.
- Citá el lugar y su tipo cuando menciones una reseña específica.
- No inventes información que no esté en el contexto.
- Podés hacer preguntas de seguimiento para entender mejor lo que busca el usuario.
"""

def construir_prompt(pregunta, chunks_recuperados, historial):

    # Contexto de chunks
    contexto = ''
    for i, chunk in enumerate(chunks_recuperados, 1):
        contexto += f'\n[{i}] Lugar: {chunk["lugar"]} ({chunk["categoria"]}) | '
        contexto += f'Calificación: {chunk["calificacion"]}⭐ | Polaridad: {chunk["polaridad"]}\n'
        contexto += f'Reseña: {chunk["texto"]}\n'

    # Historial de conversación (últimos 5 turnos)
    historial_texto = ''
    for turno in historial[-10:]:
        rol = 'Usuario' if turno['role'] == 'user' else 'TuriGuía'
        historial_texto += f'{rol}: {turno["content"]}\n'

    prompt = f"""{PROMPT_SISTEMA}

=== RESEÑAS RELEVANTES DEL CORPUS ===
{contexto}

=== HISTORIAL DE CONVERSACIÓN ===
{historial_texto}

=== PREGUNTA ACTUAL ===
Usuario: {pregunta}

TuriGuía:"""

    return prompt


# Generador con Gemini
def configurar_gemini(api_key):
    """
    Configura la API de Gemini con la clave proporcionada.
    """
    from google import genai
    client = genai.Client(api_key="XXXXXXXXXXXXXXXXXXXXXXX")
    return client


def generar_respuesta(prompt, modelo_gemini):
    try:
        respuesta = modelo_gemini.models.generate_content(
            model='gemini-3.5-flash',  # ← cambiá esto
            contents=prompt
        )
        return respuesta.text
    except Exception as e:
        return f'Lo siento, ocurrió un error al generar la respuesta: {str(e)}'


"""
    Ejecuta el pipeline RAG completo:
    1. Busca chunks relevantes en FAISS
    2. Construye el prompt con contexto e historial
    3. Genera la respuesta con Gemini
    Retorna la respuesta y los chunks recuperados.
    """

def pipeline_rag(pregunta, modelo_emb, indice, chunks, modelo_gemini, historial,
                 filtro_categoria=None, filtro_polaridad=None, top_k=TOP_K):

    chunks_recuperados = buscar_chunks(
        pregunta, modelo_emb, indice, chunks,
        top_k=top_k,
        filtro_categoria=filtro_categoria,
        filtro_polaridad=filtro_polaridad
    )

    prompt = construir_prompt(pregunta, chunks_recuperados, historial)
    respuesta = generar_respuesta(prompt, modelo_gemini)

    return respuesta, chunks_recuperados






