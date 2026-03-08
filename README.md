# 🔊 Water Pipe Leak Detection with Vision Transformers

> **Tesis de Licenciatura · Jorge Ramírez Hernández**  
> Detección de fugas en tuberías de agua mediante representaciones tiempo-frecuencia y Vision Transformers (ViT).

---

## Overview

Pipeline completo para la tesis *"Detección de Fugas en Tuberías de Agua Mediante Transformadores de Visión"*.

La idea central es convertir **señales acústicas 1-D** en **imágenes tiempo-frecuencia 2-D**
(escalogramas o espectrogramas) y clasificarlas con un **Vision Transformer (ViT)**.

```
Señal acústica cruda
        │
        ▼
  Wavelet Denoising  (db4, nivel 3)
        │
        ├──── CWT  →  Escalograma   (50 × 2048)
        │
        └──── STFT →  Log-Espectrograma  (272 × 112)
                             │
                             ▼
                     Vision Transformer
                     ┌───────────────────┐
                     │  Patch Embedding  │
                     │  Positional Enc.  │
                     │  Transformer ×10  │
                     │  CLS → MLP Head   │
                     └───────────────────┘
                             │
                  ┌──────────┴──────────┐
              Binario               Multiclase
          (fuga / no-fuga)   (no-fuga + 4 tipos de fuga)
```

---

## Dataset

Señales acústicas grabadas en dos topologías de tubería a **25 600 Hz**.

| Topología | Descripción |
|-----------|-------------|
| **Branched** | Red de tuberías con bifurcaciones en T |
| **Looped** | Red de tuberías en circuito cerrado |

**Clases (tarea multiclase):**

| Label | Carpeta en el dataset |
|-------|-----------------------|
| 0 | Circumferential Crack |
| 1 | Gasket Leak |
| 2 | Longitudinal Crack |
| 3 | No-leak |
| 4 | Orifice Leak |

Fuente: [Mendeley Data — Water Pipe Leakage Detection](https://data.mendeley.com)

---

## Matriz de experimentos

16 experimentos cubriendo todas las combinaciones de:

| Eje | Opciones |
|-----|----------|
| Transformada tiempo-frecuencia | CWT · STFT |
| Topología de tubería | Branched · Looped |
| Tarea de clasificación | Binaria · Multiclase |
| Inicialización del modelo | Scratch · Preentrenado ImageNet (ViT-B/16) |

---

## Estructura del repositorio

```
├── train.py                    # Punto de entrada único para los 16 experimentos
├── test_pipeline.py            # Tests con datos sintéticos (sin dataset real)
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # WaveletDenoising, denoise_signal_batch
│   │   ├── transforms.py      # Escalogramas CWT + espectrogramas STFT
│   │   └── dataset.py         # Dataset, DataLoader factory, label maps
│   ├── models/
│   │   └── vit.py             # VisionTransformer (scratch) + PretrainedViT + build_model()
│   ├── training/
│   │   └── trainer.py         # Loop de entrenamiento, early stopping, checkpoints
│   └── utils/
│       └── metrics.py         # Confusion matrix, ROC, PR, classification report
│
├── configs/
│   └── default.yaml           # Hiperparámetros por defecto
│
├── docs/
│   └── guia_colab.md          # Guía paso a paso para correr en Google Colab
│
├── requirements.txt
└── .gitignore
```

---

## Pipeline de datos

El pipeline está diseñado para no cargar el dataset completo en RAM en ningún momento,
lo que lo hace estable en entornos con memoria limitada como Google Colab.

```
CSVs del dataset Mendeley
        │
        ▼
load_signals_from_csv()          # Lee frames de 1 segundo, balancea clases
        │
        ▼
denoise_signal_batch()           # Wavelet denoising señal por señal
        │
        ▼
CWT / STFT                       # Una imagen a la vez
        │
        ▼
cache .h5 en Drive               # Escrita imagen por imagen (sin pico de RAM)
        │
        ▼
HDF5Dataset (lazy)               # Lee del .h5 bajo demanda, batch por batch
        │
        ▼
DataLoader → ViT → train()
```

El archivo `.h5` se calcula una vez y se reutiliza en todas las ejecuciones
siguientes de la misma combinación `transform + topology + task`.
Los experimentos scratch y pretrained comparten el mismo cache.

---

## Quick start

### 1 · Instalar dependencias

```bash
sudo apt-get install libfftw3-single3   # requerido por fCWT
pip install -r requirements.txt
pip install fCWT
```

### 2 · Verificar instalación

```bash
python test_pipeline.py --full
# Debe imprimir: 9 pasados | 0 fallidos
```

### 3 · Correr un experimento

`train.py` es el único punto de entrada para los 16 experimentos:

```bash
# ViT desde cero | STFT | Branched | Binario
python train.py --transform stft --topology branched --task binary \
    --data_dir /ruta/al/Dataset --cache_dir /ruta/al/cache

# ViT desde cero | CWT | Looped | Multiclase
python train.py --transform cwt --topology looped --task multiclass \
    --data_dir /ruta/al/Dataset --cache_dir /ruta/al/cache

# ViT preentrenado (ImageNet) | STFT | Branched | Binario
python train.py --transform stft --topology branched --task binary --pretrained \
    --data_dir /ruta/al/Dataset --cache_dir /ruta/al/cache

# Sobrescribir hiperparámetros
python train.py --transform cwt --topology looped --task multiclass \
    --lr 5e-5 --epochs 100 --batch_size 16
```

Outputs guardados en `checkpoints/{transform}_{topology}_{task}_{model}/`:

```
checkpoints/stft_branched_binary_scratch/
├── best_model.pt
├── training_history.png
├── confusion_matrix.png
├── roc_curve.png
└── pr_curve.png
```

### 4 · Google Colab

Ver [`docs/guia_colab.md`](docs/guia_colab.md) para instrucciones paso a paso,
incluyendo cómo activar GPU, manejo del cache en Drive y solución de problemas.

---

## Arquitectura del modelo

### ViT desde cero

| Hiperparámetro | Valor |
|----------------|-------|
| Embedding dim | 768 |
| Profundidad (bloques) | 10 |
| Attention heads | 8 |
| MLP ratio | 4× |
| Dropout | 0.15 |
| Attention dropout | 0.05 |
| Optimizador | AdamW |
| LR scheduler | ReduceLROnPlateau |

### ViT preentrenado

- Backbone: **ViT-Base/16** (pesos ImageNet-21k via `timm`)
- Las imágenes se redimensionan a 224 × 224 con interpolación bilineal
- El canal único (escala de grises) se replica a 3 canales (RGB) antes del backbone
- Solo la cabeza de clasificación se reemplaza y entrena desde cero

### Shapes de entrada por transformada

| Transformada | Shape imagen | Patch size | Nº patches |
|-------------|-------------|------------|-----------|
| CWT | 50 × 2048 | 5 × 16 | 10 × 128 = 1280 |
| STFT | 272 × 112 | 8 × 8 | 34 × 14 = 476 |

---

## Dependencias

| Librería | Propósito |
|---------|---------|
| `torch` / `torchvision` | Deep learning |
| `timm` | Backbone ViT-B/16 preentrenado |
| `PyWavelets` | Wavelet denoising + CWT fallback |
| `fCWT` | CWT rápida (GPU/CPU) |
| `scipy` | STFT |
| `h5py` | Cache de imágenes en HDF5 |
| `scikit-learn` | Métricas, splits estratificados |
| `seaborn` | Visualización de confusion matrix |

---

## Citación

```bibtex
@thesis{ramirez2024leak,
  author  = {Ramírez Hernández, Jorge},
  title   = {Detección de Fugas en Tuberías de Agua Mediante Transformadores de Visión},
  school  = {[Universidad]},
  year    = {2024},
  type    = {Tesis de Licenciatura}
}
```

---

## Licencia

MIT — ver [LICENSE](LICENSE).
