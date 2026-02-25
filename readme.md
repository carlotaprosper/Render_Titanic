# 🚢 Predicción de Supervivencia del Titanic... ¡con Visión Artificial! 👁️

Esta es una aplicación web interactiva construida con **Flask**. A diferencia de los modelos tradicionales donde el usuario introduce sus datos manualmente, esta app recibe la **URL de una imagen**. Utiliza el modelo de visión multimodal de **Cohere** para analizar la foto, estimar las características de la persona (sexo, edad y estatus/clase) y, a continuación, pasa esos datos a un modelo de Machine Learning local para predecir si sobreviviría al desastre del Titanic o si "palmaría".

## ✨ Características Principales

* **Extracción de Features con IA**: Utiliza el modelo `command-a-vision-07-2025` de Cohere para interpretar imágenes y devolver un JSON estructurado de forma estricta.
* **Modelo Predictivo Local**: Usa un modelo clásico pre-entrenado (`model.pkl`) creado a partir de un entorno de experimentación (`training.ipynb`).
* **Interfaz Web**: Formularios HTML dinámicos usando el motor de plantillas Jinja2 de Flask.
* **Seguridad**: Gestión segura de la API Key de Cohere mediante variables de entorno (`.env`).

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**
* **Flask**: Framework para el backend web y renderizado de plantillas.
* **Cohere SDK**: Para la comunicación con el modelo de visión artificial.
* **Pickle & Scikit-learn**: Para la carga y ejecución del modelo predictivo pre-entrenado.
* **Python-dotenv**: Para la carga de variables de entorno.

## 📂 Estructura del Proyecto

Para que la aplicación funcione correctamente, tu directorio debe verse más o menos así:

```text
📁 tu_proyecto/
├── 📄 app.py                 # El script principal de la aplicación Flask
├── 📄 model.pkl              # El modelo predictivo serializado
├── 📄 training.ipynb         # Notebook de Jupyter donde se entrenó el modelo
├── 📄 .env                   # Variables de entorno (¡No subir a Git!)
├── 📄 .gitignore             # Archivos a ignorar en el control de versiones
└── 📁 templates/             # CARPETA OBLIGATORIA PARA FLASK
    ├── 📄 index.html         # Página de inicio con el formulario
    └── 📄 index2.html        # Página de resultados
----------------------------
### 🚀 Instalación y Configuración
#### 1. Instalar dependencias Asegúrate de tener instaladas las librerías necesarias ejecutando:

Bash:

pip install Flask cohere requests pandas scikit-learn python-dotenv
#### 2. Configurar la API Key de Cohere Crea un archivo llamado exactamente .env en la raíz de tu proyecto y añade tu clave de API de Cohere:

Fragmento de código:

COHERE_API_KEY=tu_clave_secreta_de_cohere_aqui
#### 3. Ejecutar la aplicación Arranca el servidor de desarrollo de Flask ejecutando:

Bash:

python app.py
La aplicación estará disponible en http://localhost:5000/.

### 🛣️ Flujo de la Aplicación (Rutas)
GET /: Carga la página inicial (index.html) donde el usuario puede introducir la URL de una imagen.

POST /inicio:

Recibe la URL de la imagen.

Envía un prompt hiper-estricto a Cohere para analizar la imagen y extraer age, sex y class en formato JSON.

Procesa el JSON y se lo pasa a model.predict().

Renderiza index2.html mostrando la imagen original, las características detectadas y el veredicto final.

### 🧠 Notas sobre el Entrenamiento (training.ipynb)
El archivo model.pkl utilizado en esta API fue generado en el notebook training.ipynb. Si deseas modificar el algoritmo (por ejemplo, cambiar un Random Forest por una Regresión Logística), debes re-entrenar el modelo en ese notebook y sobrescribir el archivo .pkl.
