"""
Funciones para fine-tuning de DistilBETO sobre el corpus de reseñas turísticas.
Clasificador mixto: tipo de lugar (A) + polaridad (B).
Modelo base: dccuchile/distilbert-base-spanish-uncased
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# Configuración
MODELO_BASE      = 'dccuchile/distilbert-base-spanish-uncased'
DIR_MODELO       = 'models/clasificador_turismo'
SEED             = 42
MAX_LENGTH       = 128
BATCH_SIZE       = 16
EPOCHS           = 3

# Etiquetas
# Clasificación combinada: categoria + polaridad
ETIQUETAS = [
    'parque_positivo',
    'parque_neutro',
    'parque_negativo',
    'restaurante_positivo',
    'restaurante_neutro',
    'restaurante_negativo',
    'alojamiento_positivo',
    'alojamiento_neutro',
    'alojamiento_negativo',
]

LABEL2ID = {label: i for i, label in enumerate(ETIQUETAS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# Preparación del dataset
def preparar_dataset(df, col_texto='comentarios_espanol',
                     col_categoria='categoria', col_polaridad='polaridad'):
    """
    Prepara el dataset para el fine-tuning.
    Crea la etiqueta combinada categoria_polaridad.
    Filtra filas sin texto o con etiqueta inválida.
    Retorna un DataFrame limpio con columnas 'texto' y 'label'.
    """
    df = df.copy()

    # Crear etiqueta combinada
    df['label_str'] = df[col_categoria] + '_' + df[col_polaridad]

    # Filtrar etiquetas válidas y textos vacíos
    df = df[df['label_str'].isin(ETIQUETAS)]
    df = df.dropna(subset=[col_texto])
    df = df[df[col_texto].str.strip() != '']

    df['label'] = df['label_str'].map(LABEL2ID)
    df['texto'] = df[col_texto].str.strip()

    print(f'Total muestras válidas: {len(df)}')
    print('\nDistribución de etiquetas:')
    print(df['label_str'].value_counts())

    return df[['texto', 'label', 'label_str']].reset_index(drop=True)


def dividir_dataset(df, test_size=0.15, val_size=0.15):
    """
    Divide el dataset en train / validation / test
    Proporción: 70/15/15
    Retorna tres DataFrames.
    """
    # Primero separar test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df['label']
    )
    # Luego separar val del resto
    val_relativo = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relativo, random_state=SEED, stratify=train_val['label']
    )

    print(f'Train : {len(train)} muestras')
    print(f'Val   : {len(val)} muestras')
    print(f'Test  : {len(test)} muestras')

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# Dataset de PyTorch
class ResenasDataset(Dataset):

    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.textos = df['texto'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.textos[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids'     : encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels'        : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Métricas
def calcular_metricas(eval_pred):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'f1_macro': f1}


# Entrenamiento
def entrenar_modelo(train_dataset, val_dataset, num_labels=len(ETIQUETAS),
                    output_dir=DIR_MODELO):
    """
    Carga DistilBETO y lo entrena con el Trainer de HuggingFace.
    Retorna el modelo entrenado y el tokenizer.
    """
    print(f'Cargando tokenizer y modelo base: {MODELO_BASE}')
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        report_to='none',
    )

    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=calcular_metricas,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print('Iniciando entrenamiento...')
    trainer.train()
    print('Entrenamiento completado.')

    return modelo, tokenizer, trainer


# Evaluación
def evaluar_modelo(trainer, test_dataset, test_df, output_dir='src/resultados'):
    """
    Evalúa el modelo sobre el conjunto de test.
    Genera accuracy, F1 macro, reporte de clasificación y matriz de confusión.
    Guarda los resultados en JSON.
    """
    print('Evaluando sobre conjunto de test...')
    predicciones = trainer.predict(test_dataset)
    preds = np.argmax(predicciones.predictions, axis=-1)
    labels = predicciones.label_ids

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    reporte = classification_report(labels, preds,
                                    target_names=ETIQUETAS,
                                    zero_division=0)
    matriz = confusion_matrix(labels, preds)

    print(f'\nAccuracy : {acc:.4f}')
    print(f'F1 Macro : {f1:.4f}')
    print(f'\nReporte de clasificación:\n{reporte}')

    # Guardar métricas en JSON
    os.makedirs(output_dir, exist_ok=True)
    metricas = {
        'accuracy' : round(acc, 4),
        'f1_macro' : round(f1, 4),
        'matriz_confusion': matriz.tolist(),
        'reporte'  : reporte,
    }
    with open(f'{output_dir}/metricas_finetuning.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f'\nMétricas guardadas en {output_dir}/metricas_finetuning.json')

    return metricas, preds


# Baseline zero-shot
def evaluar_baseline_zeroshot(test_df):
    """
    Baseline sin fine-tuning: predice siempre la clase más frecuente del train.
    Sirve para comparar la ganancia del fine-tuning.
    """
    clase_mas_frecuente = test_df['label'].mode()[0]
    preds_baseline = [clase_mas_frecuente] * len(test_df)
    labels = test_df['label'].tolist()

    acc = accuracy_score(labels, preds_baseline)
    f1  = f1_score(labels, preds_baseline, average='macro', zero_division=0)

    print('=== BASELINE ZERO-SHOT (clase más frecuente) ===')
    print(f'Accuracy : {acc:.4f}')
    print(f'F1 Macro : {f1:.4f}')
    print(f'Clase predicha siempre: {ID2LABEL[clase_mas_frecuente]}')

    return {'accuracy': acc, 'f1_macro': f1}


# Inferencia
def cargar_modelo_entrenado(model_dir=DIR_MODELO):

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    modelo = AutoModelForSequenceClassification.from_pretrained(model_dir)
    modelo.eval()
    return modelo, tokenizer


def clasificar_texto(texto, modelo, tokenizer, max_length=MAX_LENGTH):

    inputs = tokenizer(
        texto,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = modelo(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id = probs.argmax().item()
    pred_label = ID2LABEL[pred_id]
    confianza = probs[pred_id].item()

    # Extraer categoria y polaridad por separado
    partes = pred_label.split('_')
    categoria = partes[0]
    polaridad = partes[1]

    return {
        'label'    : pred_label,
        'categoria': categoria,
        'polaridad': polaridad,
        'confianza': round(confianza, 4),
        'scores'   : {ETIQUETAS[i]: round(p.item(), 4) for i, p in enumerate(probs)}
    }
