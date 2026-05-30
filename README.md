````md
# 🎾 SmartBallTracking — Padel Vision Analytics Light

Sistema de análisis de partidos de pádel mediante visión por computador y aprendizaje automático enfocado en la detección y seguimiento de la pelota en videos de cancha.

---

# 🚀 Características

✅ Auto-labeling con YOLOv8 + SAHI  
✅ Generación automática de datasets YOLO  
✅ Human-in-the-loop review  
✅ Compatible con Roboflow  
✅ Escaneo recursivo de frames  
✅ Debug visual automático  
✅ Sistema incremental (`processed.txt`)  
✅ Optimizado para datasets grandes  
✅ Compatible con GPU CUDA  
✅ Pipeline completo de entrenamiento para detección de pelota de pádel

---

# 🧠 Tecnologías

- Python 3.11+
- YOLOv8
- SAHI
- OpenCV
- Roboflow
- NumPy
- YAML
- Jupyter Notebook
- UV Package Manager

---

# 📂 Estructura del proyecto

```text
SmartBallTracking_padel-vision-analytics-light/
│
├── Assets/
│   ├── frames_ball/
│   │   ├── partido_1/
│   │   ├── partido_2/
│   │   └── ...
│   │
│   └── videos/
│
├── dataset/
│   ├── images/
│   ├── labels/
│   ├── debug/
│   ├── review/
│   ├── data.yaml
│   └── processed.txt
│
├── upload/
│
├── src/
│   ├── review_tool.py
│   └── validate_annotations.py
│
├── auto-labeling-final.ipynb
├── best.pt
├── pyproject.toml
├── .env
├── .gitignore
└── README.md
````

---

# ⚡ Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/TU-USUARIO/SmartBallTracking_padel-vision-analytics-light.git

cd SmartBallTracking_padel-vision-analytics-light
```

---

## 2. Crear entorno virtual

### Con UV (recomendado)

```bash
uv venv
```

Activar:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

---

## 3. Instalar dependencias

```bash
uv sync
```

o:

```bash
pip install -e .
```

---

# 🔑 Variables de entorno

Crear archivo `.env`:

```env
ROBOFLOW_API_KEY=tu_api_key
```

---

# 🎥 Agregar frames

Los frames deben ir en:

```text
Assets/frames_ball/
```

Puedes usar subcarpetas:

```text
Assets/frames_ball/
├── partido_1/
├── partido_2/
└── entrenamiento/
```

El sistema escanea automáticamente todas las imágenes de forma recursiva.

---

# 📓 Notebook principal

Todo el pipeline principal se encuentra en:

```text
auto-labeling-final.ipynb
```

El notebook incluye:

✅ Auto-labeling con YOLOv8 + SAHI
✅ Generación de labels YOLO
✅ Human review system
✅ Generación de debug images
✅ Sistema incremental (`processed.txt`)
✅ Preparación de dataset
✅ Upload rápido a Roboflow
✅ Validación automática del dataset

---

# 🤖 Auto-labeling con SAHI

El notebook utiliza slicing inteligente para detectar objetos pequeños:

```python
SLICE_HEIGHT = 640
SLICE_WIDTH = 640

OVERLAP_HEIGHT = 0.20
OVERLAP_WIDTH = 0.20
```

Ideal para:

* pelotas pequeñas
* cámaras lejanas
* partidos completos
* tracking deportivo

---

# 🧪 Sistema de Review

Los casos ambiguos o con múltiples detecciones son enviados automáticamente a:

```text
dataset/review/
```

Cada caso contiene:

* imagen debug
* YAML editable
* detecciones detectadas
* referencia al label original

---

# 📦 Dataset generado

Formato YOLO:

```text
dataset/
├── images/
├── labels/
└── data.yaml
```

Compatible con:

* YOLOv8
* Roboflow
* Ultralytics
* CVAT
* Supervisely

---

# ☁️ Upload a Roboflow

El notebook incluye un uploader optimizado que:

✅ limpia automáticamente el dataset
✅ ignora debug/review
✅ verifica labels válidos
✅ sube usando workers paralelos

---

# 🖼️ Sistema de Debug

El sistema genera automáticamente overlays visuales en:

```text
dataset/debug/
```

para validar rápidamente las detecciones generadas.

---

# 🔄 Sistema incremental

Las imágenes procesadas se almacenan en:

```text
dataset/processed.txt
```

Esto permite:

* continuar procesos interrumpidos
* evitar reprocesar imágenes
* agregar nuevos frames dinámicamente
* trabajar con datasets grandes

---

# 🎯 Objetivo del proyecto

Este proyecto busca construir un sistema de analítica deportiva para pádel basado en inteligencia artificial, enfocado inicialmente en:

* detección de pelota
* tracking automático
* análisis de trayectorias
* eventos del partido
* estadísticas avanzadas

---

# 📈 Roadmap

* [x] Auto-labeling
* [x] SAHI integration
* [x] Human review system
* [x] Roboflow uploader
* [ ] Ball tracking temporal
* [ ] Heatmaps
* [ ] Trayectorias
* [ ] Detección de golpes
* [ ] Detección de rebotes
* [ ] Dashboard analítico
* [ ] IA táctica

---

# 👨‍💻 Autor

**Santiago Peña Beltran**
Ingeniería Informática — Universidad de La Sabana

---

# 📄 Licencia

MIT License

---

# ⭐ Inspiración

Inspirado en sistemas modernos de sports analytics y computer vision aplicados a deportes de raqueta como:

* Pádel
* Tennis Analytics
* Hawk-Eye
* Computer Vision Sports Tracking

---

# 🚀 SmartBallTracking

> “Turning padel matches into data.”

```
```
