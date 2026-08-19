"""
Integra RAG + Clasificador Fine-tuneado + Memoria conversacional.
"""

import os
import pickle
import numpy as np
import faiss
from google import genai
from sentence_transformers import SentenceTransformer
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification
)
import torch

# Configuración
MODELO_EMBEDDINGS = 'intfloat/multilingual-e5-small'
MAX_HISTORIAL     = 5
TOP_K             = 5
MAX_LENGTH        = 128

# Etiquetas del clasificador
ETIQUETAS = [
    'parque_positivo', 'parque_neutro', 'parque_negativo',
    'restaurante_positivo', 'restaurante_neutro', 'restaurante_negativo',
    'alojamiento_positivo', 'alojamiento_neutro', 'alojamiento_negativo',
]
ID2LABEL = {i: label for i, label in enumerate(ETIQUETAS)}

#  Prompt de sistema
PROMPT_SISTEMA = """Eres TuriSito 🌿, un asistente conversacional EXCLUSIVAMENTE especializado
en turismo de Costa Rica. Tu conocimiento proviene únicamente de reseñas reales de viajeros
sobre parques nacionales, restaurantes y alojamientos costarricenses.

ROL:
- Sos un guía turístico virtual amigable, entusiasta y conocedor de Costa Rica.
- Respondés siempre en español, con un tono cálido y natural.
- Citás el nombre del lugar y su tipo cuando mencionás una reseña específica.

TAREAS QUE PODÉS HACER:
- Recomendar lugares turísticos según lo que busca el usuario.
- Resumir lo que dicen las reseñas sobre un lugar específico.
- Comparar lugares entre sí según las experiencias de los viajeros.
- Informar sobre aspectos como servicio, comida, naturaleza, instalaciones.
- Responder preguntas de seguimiento recordando el contexto de la conversación.

LIMITACIONES ESTRICTAS:
- Solo respondés preguntas relacionadas con turismo en Costa Rica.
- Únicamente usás la información presente en las reseñas del contexto proporcionado.
- Si no encontrás información relevante en el contexto, lo decís claramente:
  "No tengo información sobre eso en mis reseñas, pero puedo ayudarte con otros aspectos del turismo en Costa Rica."
- Si el usuario pregunta algo fuera de tu dominio (política, deportes, recetas generales,
  noticias, matemáticas, programación, etc.), respondés amablemente:
  "Soy TuriSito y solo puedo ayudarte con consultas sobre turismo en Costa Rica.
   ¿Hay algún destino o lugar que te gustaría conocer?"
- Nunca inventás información que no esté en el contexto.
- No recomendás lugares que no estén en tu corpus de reseñas.
"""


class TuriSito:


    def __init__(self, api_key_gemini):
        self.historial = []
        self._cargar_componentes(api_key_gemini)

    def _cargar_componentes(self, api_key_gemini):

        print('Cargando TuriSito...')

        # Rutas absolutas basadas en la ubicación de este archivo
        base_dir         = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        dir_clasificador = os.path.join(base_dir, 'models', 'clasificador_turismo')
        cache_faiss      = os.path.join(base_dir, 'data', 'processed', 'faiss_index.bin')
        cache_chunks     = os.path.join(base_dir, 'data', 'processed', 'chunks_cache.pkl')

        print(f'  → Base del proyecto : {base_dir}')
        print(f'  → Clasificador      : {dir_clasificador}')
        print(f'  → FAISS             : {cache_faiss}')

        # Modelo de embeddings
        print('  → Modelo de embeddings...')
        self.modelo_emb = SentenceTransformer(MODELO_EMBEDDINGS)

        # Índice FAISS y chunks
        print('  → Índice FAISS y chunks...')
        self.indice = faiss.read_index(cache_faiss)
        with open(cache_chunks, 'rb') as f:
            self.chunks = pickle.load(f)

        # Clasificador fine-tuneado
        print('  → Clasificador fine-tuneado...')
        self.tokenizer_clf = BertTokenizer.from_pretrained(dir_clasificador)
        self.modelo_clf    = AutoModelForSequenceClassification.from_pretrained(
            dir_clasificador,
            ignore_mismatched_sizes=True
        )
        self.modelo_clf.eval()

        # Gemini
        print('  → Conectando con Gemini...')
        self.cliente_gemini = genai.Client(api_key=api_key_gemini)

        print('TuriSito listo para conversar 🌿')

    #Clasificador
    def _clasificar_pregunta(self, texto):

        inputs = self.tokenizer_clf(
            texto, truncation=True, padding='max_length',
            max_length=MAX_LENGTH, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.modelo_clf(**inputs)

        probs    = torch.softmax(outputs.logits, dim=-1).squeeze()
        pred_id  = probs.argmax().item()
        label    = ID2LABEL[pred_id]
        confianza = probs[pred_id].item()

        partes    = label.split('_')
        return partes[0], partes[1], confianza

    #Búsqueda RAG
    def _buscar_chunks(self, pregunta, filtro_categoria=None, filtro_polaridad=None):

        embedding = self.modelo_emb.encode([f'query: {pregunta}'])
        embedding = np.array(embedding).astype('float32')
        faiss.normalize_L2(embedding)

        k_busqueda = TOP_K * 5 if (filtro_categoria or filtro_polaridad) else TOP_K
        scores, indices = self.indice.search(embedding, k_busqueda)

        resultados = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(score)

            if filtro_categoria and chunk['categoria'] != filtro_categoria:
                continue
            if filtro_polaridad and chunk['polaridad'] != filtro_polaridad:
                continue

            resultados.append(chunk)
            if len(resultados) >= TOP_K:
                break

        return resultados

    #  Construcción del prompt
    def _construir_prompt(self, pregunta, chunks_recuperados):
        """
        Construye el prompt completo con sistema, contexto
        de reseñas e historial conversacional.
        """
        contexto = ''
        for i, chunk in enumerate(chunks_recuperados, 1):
            contexto += f'\n[{i}] Lugar: {chunk["lugar"]} ({chunk["categoria"]}) | '
            contexto += f'Calificación: {chunk["calificacion"]}⭐ | '
            contexto += f'Polaridad: {chunk["polaridad"]}\n'
            contexto += f'Reseña: {chunk["texto"]}\n'

        historial_texto = ''
        for turno in self.historial[-(MAX_HISTORIAL * 2):]:
            rol = 'Usuario' if turno['role'] == 'user' else 'TuriSito'
            historial_texto += f'{rol}: {turno["content"]}\n'

        prompt = f"""{PROMPT_SISTEMA}

=== RESEÑAS RELEVANTES DEL CORPUS ===
{contexto if contexto else 'No se encontraron reseñas relevantes para esta consulta.'}

=== HISTORIAL DE CONVERSACIÓN ===
{historial_texto if historial_texto else 'Esta es la primera pregunta del usuario.'}

=== PREGUNTA ACTUAL ===
Usuario: {pregunta}

TuriSito:"""

        return prompt

    #Responder
    def responder(self, pregunta):
        """
        Procesa la pregunta del usuario y genera una respuesta.
        Flujo: Clasificar → Buscar chunks → Construir prompt → Gemini → Respuesta
        """
        # 1. Clasificar la pregunta
        categoria, polaridad, confianza = self._clasificar_pregunta(pregunta)

        # Aplicar filtro de categoría solo si la confianza es alta
        filtro_cat = categoria if confianza > 0.6 else None

        # 2. Buscar chunks relevantes
        chunks_recuperados = self._buscar_chunks(
            pregunta, filtro_categoria=filtro_cat
        )

        # 3. Construir prompt
        prompt = self._construir_prompt(pregunta, chunks_recuperados)

        # 4. Generar respuesta con Gemini
        try:
            respuesta_gemini = self.cliente_gemini.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt
            )
            respuesta = respuesta_gemini.text
        except Exception as e:
            respuesta = f'Lo siento, tuve un problema técnico. ¿Podés repetir tu pregunta? ({str(e)})'

        # 5. Actualizar historial
        self.historial.append({'role': 'user',      'content': pregunta})
        self.historial.append({'role': 'assistant',  'content': respuesta})

        return {
            'respuesta'          : respuesta,
            'categoria_detectada': categoria,
            'polaridad_detectada': polaridad,
            'confianza_clf'      : round(confianza, 4),
            'chunks_usados'      : len(chunks_recuperados),
            'chunks'             : chunks_recuperados
        }

    def limpiar_historial(self):
        """Reinicia la memoria conversacional."""
        self.historial = []
        print('Historial limpiado.')

    def get_historial(self):
        """Retorna el historial de conversación."""
        return self.historial