import sys
import os
from pathlib import Path
from dash import html, dcc, Input, Output, State, callback
import dash

RAIZ = Path(__file__).resolve().parents[3]
MODELOS_DIR = str(RAIZ / 'src' / 'modelos')
if MODELOS_DIR not in sys.path:
    sys.path.insert(0, MODELOS_DIR)

from src.modelos.chatbot_engine import TuriSito

VERDE_OSCURO = '#1B4332'
VERDE_MEDIO  = '#2D6A4F'
VERDE_CLARO  = '#52B788'
DORADO       = '#D4A017'
CREMA        = '#F9F5EC'
GRIS_SUAVE   = '#F0EDE6'
BLANCO       = '#FFFFFF'

API_KEY = os.getenv('GEMINI_API_KEY', 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('Iniciando TuriSito...')
bot = TuriSito(api_key_gemini=API_KEY)
print('TuriSito listo.')

SUGERENCIAS = [
    'Que parques recomendas?',
    'Donde comer bien?',
    'Mejores alojamientos?',
    'Lugares con malas resenas?'
]

PLACEHOLDER = html.P(
    'Hola! Soy TuriSito. Preguntame sobre turismo en Costa Rica.',
    style={
        'color': '#888', 'textAlign': 'center',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '15px', 'margin': 'auto', 'padding': '20px'
    }
)

def burbuja(autor, texto):
    es_usuario = autor == 'usuario'
    return html.Div(style={
        'display': 'flex',
        'justifyContent': 'flex-end' if es_usuario else 'flex-start',
        'margin': '8px 0',
        'flexShrink': '0',
    }, children=[
        dcc.Markdown(texto, style={
            'backgroundColor': VERDE_MEDIO if es_usuario else BLANCO,
            'color': BLANCO if es_usuario else '#333',
            'padding': '12px 18px',
            'borderRadius': '18px 18px 4px 18px' if es_usuario else '18px 18px 18px 4px',
            'maxWidth': '70%',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '15px',
            'lineHeight': '1.6',
            'margin': '0',
        })
    ])

def item_historial(chat_id, nombre, activo=False):
    return html.Div(
        nombre,
        id={'type': 'item-chat', 'index': chat_id},
        n_clicks=0,
        style={
            'padding': '10px 14px',
            'borderRadius': '8px',
            'cursor': 'pointer',
            'fontSize': '13px',
            'fontFamily': 'Arial, sans-serif',
            'backgroundColor': VERDE_MEDIO if activo else 'transparent',
            'color': BLANCO if activo else '#ddd',
            'marginBottom': '4px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'whiteSpace': 'nowrap',
        }
    )

layout = html.Div(style={
    'display': 'flex',
    'height': 'calc(100vh - 56px)',
    'overflow': 'hidden',
    'backgroundColor': CREMA,
}, children=[

    # ── Sidebar ───────────────────────────────────────────────────────────────
    html.Div(style={
        'width': '220px',
        'minWidth': '220px',
        'backgroundColor': VERDE_OSCURO,
        'display': 'flex',
        'flexDirection': 'column',
        'padding': '16px 12px',
        'overflowY': 'auto',
        'flexShrink': '0',
    }, children=[
        html.Button('+ Nuevo chat', id='btn-nuevo-chat', n_clicks=0, style={
            'width': '100%',
            'padding': '12px',
            'backgroundColor': VERDE_MEDIO,
            'color': BLANCO,
            'border': f'1px solid {VERDE_CLARO}',
            'borderRadius': '8px',
            'cursor': 'pointer',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px',
            'fontWeight': 'bold',
            'marginBottom': '20px',
        }),
        html.P('CONVERSACIONES', style={
            'color': '#aaa',
            'fontSize': '11px',
            'fontFamily': 'Arial, sans-serif',
            'letterSpacing': '1px',
            'marginBottom': '8px',
        }),
        html.Div(id='lista-chats'),
    ]),

    # ── Area principal ────────────────────────────────────────────────────────
    html.Div(style={
        'flex': '1',
        'display': 'flex',
        'flexDirection': 'column',
        'overflow': 'hidden',
        'minWidth': '0',
    }, children=[

        # Sugerencias
        html.Div(style={
            'display': 'flex',
            'gap': '10px',
            'flexWrap': 'wrap',
            'justifyContent': 'center',
            'padding': '16px 24px 8px',
            'flexShrink': '0',
        }, children=[
            html.Button(texto, id=f'sug-{i}', n_clicks=0, style={
                'padding': '8px 16px',
                'border': f'1px solid {VERDE_MEDIO}',
                'borderRadius': '20px',
                'backgroundColor': BLANCO,
                'color': VERDE_MEDIO,
                'cursor': 'pointer',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '13px',
            }) for i, texto in enumerate(SUGERENCIAS)
        ]),

        # Mensajes — este es el unico elemento que hace scroll
        # Mensajes
        html.Div(
            id='chat-contenedor',
            children=[PLACEHOLDER],
            style={
                'flex': '1',
                'minHeight': '0',
                'overflowY': 'auto',
                'display': 'flex',
                'flexDirection': 'column',
                'padding': '16px 24px',
                'backgroundColor': GRIS_SUAVE,
                'margin': '0 24px',
                'borderRadius': '16px',
                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
            }
        ),

        # Input
        html.Div(style={
            'display': 'flex',
            'gap': '10px',
            'alignItems': 'center',
            'padding': '16px 24px',
            'flexShrink': '0',
        }, children=[
            dcc.Input(
                id='chat-input', type='text',
                placeholder='Escribi tu pregunta sobre turismo en Costa Rica...',
                debounce=False,
                style={
                    'flex': '1',
                    'padding': '16px 22px',
                    'borderRadius': '24px',
                    'border': f'2px solid {VERDE_MEDIO}',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '15px',
                    'outline': 'none',
                    'height': '52px',
                }
            ),
            html.Button('>', id='btn-enviar', n_clicks=0, style={
                'backgroundColor': VERDE_MEDIO,
                'color': BLANCO,
                'border': 'none',
                'borderRadius': '50%',
                'width': '52px',
                'height': '52px',
                'fontSize': '20px',
                'cursor': 'pointer',
                'flexShrink': '0',
            }),
        ]),

        # Info clasificador
        html.Div(id='info-clf', style={
            'padding': '0 24px 12px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '12px',
            'color': '#999',
            'display': 'none',
            'flexShrink': '0',
        }),
    ]),

    # Stores con localStorage
    dcc.Store(id='todos-los-chats', storage_type='local', data={}),
    dcc.Store(id='chat-activo-id',  storage_type='local', data=None),
    dcc.Store(id='contador-chats',  storage_type='local', data=0),
])

dash.register_page(__name__, path='/hablemos', name='Hablemos', layout=layout)


@callback(
    Output('todos-los-chats', 'data',  allow_duplicate=True),
    Output('chat-activo-id',  'data',  allow_duplicate=True),
    Output('contador-chats',  'data',  allow_duplicate=True),
    Input('btn-nuevo-chat',   'n_clicks'),
    State('todos-los-chats',  'data'),
    State('contador-chats',   'data'),
    prevent_initial_call=True,
)
def nuevo_chat(n, todos, contador):
    if not n:
        raise dash.exceptions.PreventUpdate
    nuevo_id  = f'chat_{contador + 1}'
    nuevo_num = contador + 1
    todos[nuevo_id] = {'nombre': f'Chat {nuevo_num}', 'mensajes': []}
    bot.limpiar_historial()
    return todos, nuevo_id, nuevo_num


@callback(
    Output('chat-activo-id', 'data', allow_duplicate=True),
    Input({'type': 'item-chat', 'index': dash.ALL}, 'n_clicks'),
    State({'type': 'item-chat', 'index': dash.ALL}, 'id'),
    prevent_initial_call=True,
)
def seleccionar_chat(n_clicks_list, ids):
    from dash import ctx
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered_id
    if triggered:
        bot.limpiar_historial()
        return triggered['index']
    raise dash.exceptions.PreventUpdate


@callback(
    Output('lista-chats',     'children'),
    Output('chat-contenedor', 'children', allow_duplicate=True),
    Input('todos-los-chats',  'data'),
    Input('chat-activo-id',   'data'),
    prevent_initial_call='initial_duplicate',
)
def renderizar_sidebar(todos, activo_id):
    if not todos:
        return [], [PLACEHOLDER]
    items = [
        item_historial(cid, cdata['nombre'], activo=(cid == activo_id))
        for cid, cdata in reversed(list(todos.items()))
    ]
    if activo_id and activo_id in todos:
        mensajes = todos[activo_id]['mensajes']
        burbujas = [burbuja(m['autor'], m['texto']) for m in mensajes] if mensajes else [PLACEHOLDER]
    else:
        burbujas = [PLACEHOLDER]
    return items, burbujas


@callback(
    Output('todos-los-chats', 'data',     allow_duplicate=True),
    Output('chat-activo-id',  'data',     allow_duplicate=True),
    Output('contador-chats',  'data',     allow_duplicate=True),
    Output('chat-input',      'value'),
    Output('chat-contenedor', 'children', allow_duplicate=True),
    Output('info-clf',        'children'),
    Output('info-clf',        'style'),
    Input('btn-enviar',       'n_clicks'),
    Input('chat-input',       'n_submit'),
    *[Input(f'sug-{i}',       'n_clicks') for i in range(len(SUGERENCIAS))],
    State('chat-input',       'value'),
    State('todos-los-chats',  'data'),
    State('chat-activo-id',   'data'),
    State('contador-chats',   'data'),
    prevent_initial_call=True,
)
def enviar_mensaje(n_env, n_sub, *args):
    from dash import ctx
    pregunta  = args[len(SUGERENCIAS)]
    todos     = args[len(SUGERENCIAS) + 1]
    activo_id = args[len(SUGERENCIAS) + 2]
    contador  = args[len(SUGERENCIAS) + 3]

    estilo_visible = {
        'padding': '0 24px 12px',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '12px', 'color': '#999', 'display': 'block', 'flexShrink': '0',
    }
    estilo_oculto = {**estilo_visible, 'display': 'none'}

    for i, sug in enumerate(SUGERENCIAS):
        if ctx.triggered_id == f'sug-{i}':
            pregunta = sug
            break

    if not pregunta or not pregunta.strip():
        raise dash.exceptions.PreventUpdate

    if not activo_id or activo_id not in (todos or {}):
        contador  = (contador or 0) + 1
        activo_id = f'chat_{contador}'
        todos     = todos or {}
        todos[activo_id] = {'nombre': f'Chat {contador}', 'mensajes': []}
        bot.limpiar_historial()

    resultado = bot.responder(pregunta)
    respuesta = resultado['respuesta']

    todos[activo_id]['mensajes'].append({'autor': 'usuario', 'texto': pregunta})
    todos[activo_id]['mensajes'].append({'autor': 'bot',     'texto': respuesta})

    burbujas = [burbuja(m['autor'], m['texto']) for m in todos[activo_id]['mensajes']]

    info = (f"Clasificador: {resultado['categoria_detectada']} / {resultado['polaridad_detectada']} "
            f"(confianza: {resultado['confianza_clf']:.0%}) — {resultado['chunks_usados']} resenas recuperadas")

    return todos, activo_id, contador, '', burbujas, info, estilo_visible


dash.clientside_callback(
    """
    function(hijos) {
        var c = document.getElementById('chat-contenedor');
        if (c) c.scrollTop = c.scrollHeight;
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-contenedor', 'title'),
    Input('chat-contenedor',  'children'),
)