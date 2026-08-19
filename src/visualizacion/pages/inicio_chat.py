import pandas as pd
from pathlib import Path
from dash import html
import dash

VERDE_OSCURO = '#1B4332'
VERDE_MEDIO  = '#2D6A4F'
VERDE_CLARO  = '#52B788'
DORADO       = '#D4A017'
DORADO_CLARO = '#F4D03F'
CREMA        = '#F9F5EC'
GRIS_SUAVE   = '#F0EDE6'
BLANCO       = '#FFFFFF'

BASE_DIR = Path(__file__).resolve().parents[3]
df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'corpus_final.csv')
muestras = df.dropna(subset=['comentarios_espanol']).sample(6, random_state=42)

layout = html.Div(children=[

    html.Section(style={
        'background': f'linear-gradient(135deg, {VERDE_OSCURO} 0%, {VERDE_MEDIO} 60%, {VERDE_CLARO} 100%)',
        'padding': '80px 60px 60px',
        'textAlign': 'center'
    }, children=[
        html.P('CHATBOT TURISTICO COSTA RICA', style={
            'color': DORADO_CLARO, 'fontSize': '13px',
            'letterSpacing': '4px', 'marginBottom': '16px',
            'fontFamily': 'Arial, sans-serif'
        }),
        html.H1('TuriSito', style={
            'color': BLANCO, 'fontSize': '60px',
            'lineHeight': '1.2', 'margin': '0 0 16px',
        }),
        html.P('Tu guia conversacional basado en reseñas reales de viajeros', style={
            'color': DORADO_CLARO, 'fontSize': '20px',
            'margin': '0 0 24px', 'fontFamily': 'Arial, sans-serif'
        }),
        html.P(
            'Pregunta sobre parques nacionales, restaurantes y alojamientos de Costa Rica. '
            'TuriSito responde fundamentado en mas de 5,000 resenas reales de Google Maps, '
            'usando RAG y un clasificador fine-tuneado sobre el corpus turístico.',
            style={'color': '#D5E8D4', 'fontSize': '17px', 'maxWidth': '680px',
                   'margin': '0 auto 48px', 'lineHeight': '1.7', 'fontFamily': 'Arial, sans-serif'}
        ),
        html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '48px', 'flexWrap': 'wrap'}, children=[
            html.Div([
                html.Span(f'{len(df):,}', style={'color': DORADO_CLARO, 'fontSize': '42px', 'fontWeight': 'bold', 'display': 'block'}),
                html.Span('reseñas encontradas', style={'color': '#D5E8D4', 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif'})
            ]),
            html.Div([
                html.Span(f'{df["lugar"].nunique()}', style={'color': DORADO_CLARO, 'fontSize': '42px', 'fontWeight': 'bold', 'display': 'block'}),
                html.Span('destinos turisticos', style={'color': '#D5E8D4', 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif'})
            ]),
            html.Div([
                html.Span(f'{df["categoria"].nunique()}', style={'color': DORADO_CLARO, 'fontSize': '42px', 'fontWeight': 'bold', 'display': 'block'}),
                html.Span('categorias de lugar', style={'color': '#D5E8D4', 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif'})
            ]),
        ])
    ]),

    html.Section(style={'padding': '60px', 'maxWidth': '900px', 'margin': '0 auto'}, children=[
        html.H2('Como funciona TuriSito?', style={
            'color': VERDE_OSCURO, 'fontSize': '28px',
            'borderBottom': f'3px solid {DORADO}', 'paddingBottom': '12px', 'marginBottom': '28px'
        }),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '24px'}, children=[
            html.Div(style={
                'backgroundColor': BLANCO, 'borderRadius': '12px', 'padding': '24px',
                'boxShadow': '0 2px 12px rgba(0,0,0,0.07)', 'borderTop': f'4px solid {VERDE_CLARO}'
            }, children=[
                html.H3('RAG', style={'color': VERDE_OSCURO, 'fontSize': '18px', 'margin': '0 0 8px'}),
                html.P('Busca semanticamente en el corpus las resenas mas relevantes para tu pregunta usando FAISS y embeddings multilingues.',
                       style={'color': '#555', 'fontSize': '14px', 'lineHeight': '1.6', 'fontFamily': 'Arial, sans-serif', 'margin': '0'})
            ]),
            html.Div(style={
                'backgroundColor': BLANCO, 'borderRadius': '12px', 'padding': '24px',
                'boxShadow': '0 2px 12px rgba(0,0,0,0.07)', 'borderTop': f'4px solid {DORADO}'
            }, children=[
                html.H3('Fine-Tuning', style={'color': VERDE_OSCURO, 'fontSize': '18px', 'margin': '0 0 8px'}),
                html.P('Un clasificador DistilBETO entrenado sobre el corpus detecta el tipo de lugar y la polaridad para filtrar resultados con mayor precision.',
                       style={'color': '#555', 'fontSize': '14px', 'lineHeight': '1.6', 'fontFamily': 'Arial, sans-serif', 'margin': '0'})
            ]),
            html.Div(style={
                'backgroundColor': BLANCO, 'borderRadius': '12px', 'padding': '24px',
                'boxShadow': '0 2px 12px rgba(0,0,0,0.07)', 'borderTop': f'4px solid {VERDE_MEDIO}'
            }, children=[
                html.H3('Gemini', style={'color': VERDE_OSCURO, 'fontSize': '18px', 'margin': '0 0 8px'}),
                html.P('Genera respuestas naturales y contextualizadas basandose exclusivamente en las resenas recuperadas del corpus costarricense.',
                       style={'color': '#555', 'fontSize': '14px', 'lineHeight': '1.6', 'fontFamily': 'Arial, sans-serif', 'margin': '0'})
            ]),
        ])
    ]),

    html.Section(style={'backgroundColor': GRIS_SUAVE, 'padding': '60px'}, children=[
        html.H2('Reseñas del corpus', style={
            'color': VERDE_OSCURO, 'fontSize': '28px', 'textAlign': 'center',
            'borderBottom': f'3px solid {DORADO}', 'paddingBottom': '12px',
            'maxWidth': '900px', 'margin': '0 auto 40px'
        }),
        html.Div(style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))',
            'gap': '24px', 'maxWidth': '1100px', 'margin': '0 auto'
        }, children=[
            html.Div(style={
                'backgroundColor': BLANCO, 'borderRadius': '12px', 'padding': '24px',
                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
                'borderTop': f'4px solid {VERDE_CLARO}'
            }, children=[
                html.P(str(int(row['calificacion'])) + '/5', style={
                    'margin': '0 0 8px', 'fontSize': '14px',
                    'color': DORADO, 'fontFamily': 'Arial, sans-serif', 'fontWeight': 'bold'
                }),
                html.P(str(row['lugar']), style={
                    'color': VERDE_MEDIO, 'fontSize': '13px', 'margin': '0 0 6px',
                    'fontFamily': 'Arial, sans-serif', 'fontWeight': 'bold'
                }),
                html.P(str(row['categoria']).capitalize(), style={
                    'color': BLANCO, 'fontSize': '11px', 'fontFamily': 'Arial, sans-serif',
                    'backgroundColor': VERDE_MEDIO, 'display': 'inline-block',
                    'padding': '2px 10px', 'borderRadius': '10px', 'margin': '0 0 12px'
                }),
                html.P(
                    str(row['comentarios_espanol'])[:200] + '...'
                    if len(str(row['comentarios_espanol'])) > 200
                    else str(row['comentarios_espanol']),
                    style={'color': '#444', 'fontSize': '14px', 'lineHeight': '1.6',
                           'margin': '0', 'fontStyle': 'italic', 'fontFamily': 'Arial, sans-serif'}
                )
            ]) for _, row in muestras.iterrows()
        ])
    ]),
])

dash.register_page(__name__, path='/', name='Inicio', layout=layout)
