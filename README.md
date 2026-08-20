# Chatbot de reseñas - TuriSito

## TurisPOS


### ✏️ Descripción 

En esta parte del proyecto de TurisPOS, iremos por el rumbo de los agentes conversacionales de Inteligencia Artificial, ya que se implementará un pequeño agente con el que vamos a poder hablar para pedirle opiniones acerca de lugares turísticos de Costa Rica.

El corpus trabajado en este proyecto es el mismo que se trabajó para la primera parte de TurisPOS, con la diferencia de que en esta ocasión, se enriqueció mucho más gracias a la incorporación de más comentarios

### 🛠️ Herramientas 

Python:
- Entrenamiento de modelos, tanto en RAG como en Fine Tuning.
- Guardado de información en .csv de la información limpia para los modelos.
- Uso del modelo de DistilBETO.
- Visualización de información.

Claude como asistente de IA para entendimiento y optimización de código.

Gemini API para trabajar la parte de LLM en el modelo y que responda de forma adecuada siguiendo un prompt determinado.

Plotly Dash:
- Parte visual del proyecto.
- Interfaz web en la que se puede chatear con el agente de IA. También se puede consultar el historial de conversaciones.

### 📁 Organización

✅ data
- 🗁 processed: Datos a los que se les aplicó limpieza, análisis y traducción.
- 🗁 raw: corpus de los proyectos anteriores.

✅ src
- 🗁 limpieza: unión de los corpus del proyecto 2 y que lleva las nuevas reseñas para enriquecer más nuestra fuente de datos.
- 🗁 modelos: sección en donde se desarrolla la arquitectura que seguirán los modelos para entrenamientos, pruebas y validaciones.
- 🗁 resultados: muestra en notebooks del comportamiento de los modelos y también, del entrenamiento con el Fine Tuning.
- 🗁 visualizacion: resultados visuales utilizando Plotly Dash.

#### 👩🏻‍💻 Elaborado por:

Evelin Calderón Rojas 

Estudiante de Big Data 

Curso: Minería de Textos
