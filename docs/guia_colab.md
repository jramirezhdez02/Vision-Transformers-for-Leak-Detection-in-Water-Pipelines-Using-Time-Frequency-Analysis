# Guía: Correr el pipeline en Google Colab

## Requisitos previos

El dataset de Mendeley en tu Google Drive con esta estructura exacta:

```
MyDrive/
└── Dataset/
    ├── Branched/
    │   ├── Circumferential Crack/   ← archivos .csv
    │   ├── Gasket Leak/
    │   ├── Longitudinal Crack/
    │   ├── No-leak/
    │   └── Orifice Leak/
    └── Looped/
        └── (mismas 5 carpetas)
```

El zip `leak_detection_vit_repo_v8.zip` subido a tu Drive.

---

## ⚠️ Paso 0 — Activar GPU ANTES de hacer nada

Esto es lo primero al abrir el notebook. Si lo olvidas, CWT en CPU tarda horas.

```
Runtime → Change runtime type → T4 GPU → Save
```

Para confirmar:

```python
import torch
print(torch.cuda.is_available())      # debe imprimir True
print(torch.cuda.get_device_name(0))  # debe decir Tesla T4
```

---

## Paso 1 — Setup

```python
from google.colab import drive
drive.mount('/content/drive')

# Descomprimir el repo
!unzip -q /content/drive/MyDrive/leak_detection_vit_repo_v8.zip -d /content/
%cd /content/repo

# Dependencias Python
!pip install PyWavelets timm h5py scikit-learn tqdm scipy seaborn -q

# Librería del sistema requerida por fCWT
!apt-get install -y -q libfftw3-single3

# fCWT (CWT rápida por GPU/CPU)
!pip install fCWT -q
```

> Si `fCWT` falla al instalarse, el pipeline usa `PyWavelets` automáticamente
> como fallback. No bloquea nada.

---

## Paso 2 — Verificar instalación con datos sintéticos

Antes de usar los datos reales, verifica que todos los módulos funcionan:

```python
!python test_pipeline.py --full
```

Deberías ver **9 tests con ✓** y `9 pasados | 0 fallidos`.

Si alguno falla, **no sigas adelante** — lee el traceback y corrígelo antes de continuar.

---

## Paso 3 — Primer experimento real

Empieza con **STFT + Branched + Binario** — es la combinación más rápida.

```python
!python train.py \
    --transform  stft \
    --topology   branched \
    --task       binary \
    --data_dir   /content/drive/MyDrive/Dataset \
    --cache_dir  /content/drive/MyDrive/cache \
    --epochs     50 \
    --patience   10
```

Lo que ocurre en orden:

1. Lee todos los `.csv` de `Dataset/Branched/` (5 carpetas de clases)
2. Aplica wavelet denoising señal por señal
3. Calcula los espectrogramas STFT y los guarda **uno por uno** en
   `/content/drive/MyDrive/cache/stft_branched_binary.h5`
4. Entrena el ViT leyendo del `.h5` en batches — **nunca carga todo en RAM**
5. Guarda el mejor checkpoint en `checkpoints/stft_branched_binary_scratch/best_model.pt`
6. Imprime el classification report y guarda las 4 gráficas

> **El cache se guarda en Drive, no en `/content/`.**  
> Si Colab se desconecta y reconectas, el paso 3 se salta automáticamente
> y el entrenamiento arranca directo desde el `.h5` ya calculado.

---

## Paso 4 — Ver los resultados

```python
import os
from IPython.display import Image, display

# Listar experimentos corridos
for exp in sorted(os.listdir('checkpoints')):
    path = f'checkpoints/{exp}'
    if not os.path.isdir(path): continue
    print(f'\n── {exp}')
    for f in sorted(os.listdir(path)):
        size = os.path.getsize(f'{path}/{f}') / 1e6
        print(f'   {f:<35} {size:.1f} MB')
```

```python
# Ver gráficas de un experimento
exp = 'stft_branched_binary_scratch'  # ← cambia según el experimento

for plot in ['training_history.png', 'confusion_matrix.png', 'roc_curve.png', 'pr_curve.png']:
    fpath = f'checkpoints/{exp}/{plot}'
    if os.path.exists(fpath):
        print(f'\n── {plot}')
        display(Image(fpath))
```

---

## Paso 5 — Los 16 experimentos completos

El orden recomendado es de más rápido a más lento.

```python
# ── STFT scratch ───────────────────────────────────────────────────────────────
for topology in ['branched', 'looped']:
    for task in ['binary', 'multiclass']:
        !python train.py --transform stft --topology {topology} --task {task} \
            --data_dir /content/drive/MyDrive/Dataset \
            --cache_dir /content/drive/MyDrive/cache

# ── STFT pretrained ────────────────────────────────────────────────────────────
for topology in ['branched', 'looped']:
    for task in ['binary', 'multiclass']:
        !python train.py --transform stft --topology {topology} --task {task} --pretrained \
            --data_dir /content/drive/MyDrive/Dataset \
            --cache_dir /content/drive/MyDrive/cache

# ── CWT scratch  (~20 min calcular imágenes la primera vez) ───────────────────
for topology in ['branched', 'looped']:
    for task in ['binary', 'multiclass']:
        !python train.py --transform cwt --topology {topology} --task {task} \
            --data_dir /content/drive/MyDrive/Dataset \
            --cache_dir /content/drive/MyDrive/cache

# ── CWT pretrained ─────────────────────────────────────────────────────────────
for topology in ['branched', 'looped']:
    for task in ['binary', 'multiclass']:
        !python train.py --transform cwt --topology {topology} --task {task} --pretrained \
            --data_dir /content/drive/MyDrive/Dataset \
            --cache_dir /content/drive/MyDrive/cache
```

> **El cache se comparte entre scratch y pretrained.**  
> `stft_branched_binary.h5` lo usan tanto el experimento scratch como el pretrained,
> así que solo hay 8 archivos `.h5` en total para los 16 experimentos.

---

## Copiar checkpoints a Drive

Los checkpoints viven en `/content/repo/checkpoints/` — memoria temporal de Colab.
Cópialos a Drive al terminar la sesión:

```python
import shutil

shutil.copytree(
    '/content/repo/checkpoints',
    '/content/drive/MyDrive/checkpoints_leak_vit',
    dirs_exist_ok=True
)
print('Checkpoints copiados a Drive ✓')
```

---

## Referencia de argumentos

| Argumento | Opciones | Default | Descripción |
|-----------|----------|---------|-------------|
| `--transform` | `cwt` · `stft` | `cwt` | Representación tiempo-frecuencia |
| `--topology` | `branched` · `looped` | `branched` | Topología del dataset |
| `--task` | `binary` · `multiclass` | `binary` | Tipo de clasificación |
| `--pretrained` | flag | scratch | ViT-B/16 con pesos ImageNet |
| `--data_dir` | ruta | `data/raw` | Carpeta raíz del dataset Mendeley |
| `--cache_dir` | ruta | `data/processed` | Dónde guardar los `.h5` |
| `--no_cache` | flag | — | Recalcular imágenes aunque exista cache |
| `--epochs` | int | 50 | Epochs máximos |
| `--patience` | int | 10 | Early stopping |
| `--lr` | float | 1e-4 | Learning rate inicial |
| `--weight_decay` | float | 1e-4 | Regularización L2 |
| `--batch_size` | int | 32 | Tamaño del batch |
| `--val_split` | float | 0.10 | Fracción de train usada para validación |
| `--num_workers` | int | 2 | Workers del DataLoader |
| `--seed` | int | 42 | Semilla de aleatoriedad |

---

## Tiempos estimados en T4 GPU

| Experimento | Calcular imágenes (1ª vez) | Entrenamiento (50 epochs) |
|-------------|--------------------------|--------------------------|
| STFT binary | ~5 min | ~15 min |
| STFT multiclass | ~5 min | ~20 min |
| CWT binary | ~20 min | ~15 min |
| CWT multiclass | ~20 min | ~20 min |

La segunda vez que corres la misma combinación `transform+topology+task`,
el cálculo de imágenes se salta por completo.

---

## Solución de problemas

**`^C` aparece solo / proceso muere sin error visible**  
Era un OOM (Out of Memory). Resuelto en v8: las imágenes se leen del `.h5`
una por una, sin pico de RAM. Si sigue pasando, reduce: `--batch_size 16`.

**`fCWT` no se instala o falla**  
No pasa nada — el pipeline usa `PyWavelets` como fallback automático.
Verás `CWT (pywt)` en el progress bar en lugar de `CWT`.

**Colab se desconecta a mitad del cálculo de CWT**  
Verifica si el `.h5` está completo antes de reconectar:
```python
import h5py
with h5py.File('/content/drive/MyDrive/cache/cwt_branched_binary.h5', 'r') as f:
    print({k: f[k].shape for k in f.keys()})
# Correcto: {'x_train': (N, 1, 50, 2048), 'x_test': ..., 'y_train': ..., 'y_test': ...}
```
Si faltan keys o la shape es incorrecta, borra el archivo y recalcula con `--no_cache`.

**`CUDA out of memory`**  
Reduce el batch size: `--batch_size 16` o `--batch_size 8`.

**`OSError: Unable to open file` en el `.h5`**  
Archivo corrupto. Bórralo de Drive y recalcula con `--no_cache`.
