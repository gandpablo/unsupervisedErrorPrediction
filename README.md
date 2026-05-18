# GUÍA RÁPIDA DE USO DE ESTIMATOR

## 1. Versiones

Las versiones soportadas para ejecutar la herramienta son:

* Python soportado: 3.9 – 3.13
* Python recomendado: 3.12

Dependencias según versión:

* Python 3.9:

  * NumPy 2.0.1
  * SciPy 1.13.1
  * Matplotlib 3.9.4

* Python 3.10 – 3.13:

  * NumPy 2.2.6
  * SciPy 1.15.2
  * Matplotlib 3.10.8

Se instalarán solas al usar install.sh, o se indicará al usuario
como instalar

---

## 2. Instalación

Desde la carpeta del repositorio:

```bash
chmod +x install.sh toolkit.sh
./install.sh
```

El instalador:

* comprueba que exista una versión de Python compatible
* comprueba que el módulo `venv` esté disponible
* crea el entorno virtual `.toolkit2venv`
* instala las dependencias necesarias
* instala el comando `estimator` en `~/.local/bin`

### PATH

Si `~/.local/bin` no está en el `PATH`, debes añadir:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Después, puedes comprobar la instalación usando
desde una terminal nueva:

```bash
estimator --help
```

---

## 3. Funcionamiento

El comando `estimator` ejecuta tres tipos de cálculo:

* `lm`: ajuste de la curva `f(x)=a*x^b` mediante Levenberg-Marquardt

* `sx`: ajuste de la misma curva mediante Simplex (Nelder-Mead)

* `ts`: calibración mediante temperature scaling sobre probabilidades

En todos los casos:

* se leen ficheros de calibración y evaluación
* se agrupan los datos en bloques
* se calcula la calibración
* se guardan resultados estructurados en un directorio

---

## 4. Uso

### 4.1 Sintaxis general

Para `lm` y `sx`:

```bash
estimator lm|sx archivo_calibracion archivo_evaluacion outdir [K] [--remain] \
  [--trim:VALOR]
```

Para `ts`:

```bash
estimator ts archivo_calibracion archivo_evaluacion outdir [K] [--remain] \
  [ce|dE|DE]
```

Para ficheros que ya vienen agrupados:

```bash
estimator lm|sx archivo_calibracion_agrupada archivo_evaluacion_agrupada \
  outdir K --grouped [--remain] [--trim:VALOR]

estimator ts archivo_calibracion_agrupada archivo_evaluacion_agrupada \
  outdir K --grouped [--remain]
```

---

### 4.2 Parámetros

* `lm | sx | ts`: tipo de cálculo
* `archivo_calibracion`: fichero para ajustar el modelo (.txt, .out, ...)
* `archivo_evaluacion`: fichero para evaluar el modelo (.txt, .out, ...)
* `outdir`: directorio raíz de salida
* `K`: número de grupos (por defecto `100` si no se usa `--grouped`).
  Con `--grouped` es obligatorio, aunque sólo se usa para nombrar la
  carpeta del experimento

Opcionales:

* `--remain`: el resto se guarda como último bloque independiente
* (sin flag) el resto se fusiona en el último bloque y la ruta usa `t0`
* `--grouped`: los ficheros de entrada ya están agrupados; en este modo
  `K` y `t` no se usan para calcular, sólo para que las rutas de salida
  sean comprensibles
* si detecta exactamente el mismo experimento en el mismo `outdir`,
  avisa y pide confirmación antes de recalcular y sobrescribir

Solo para `lm` y `sx`:

* `--trim:VALOR`: activa trimming de outliers

Solo para `ts`:

* `ce | dE | DE`: criterio de optimización (por defecto `ce`)

---

## 5. Formato de los ficheros (input)

### 5.1 Para `lm` y `sx`

Cada línea debe contener al menos dos valores:

```text
error_empirico error_estimado
```

Donde:

* `error_empirico`:

  * `0` si la predicción es correcta
  * `1` si es incorrecta

* `error_estimado`:

  * típicamente `1 - max(P(c|x))`
  * donde `P(c|x)` es la probabilidad softmax de la clase más probable

---

### 5.2 Para `ts`

Cada línea debe contener al menos cinco columnas:

```text
E  1-Pmax  y_true  y_pred  P0  P1 ... Pn
```

Donde:

* `E`: error empírico (0 o 1)
* `1-Pmax`: error estimado = `1 - max(P(c|x))`
* `y_true`: índice de la clase correcta
* `y_pred`: índice de la clase predicha
* `P0 ... Pn`: probabilidades por clase

Importante:

* Las probabilidades deben ser **salidas softmax (no logits)**
* Deben sumar 1

---

### 5.3 Para `--grouped`

Cuando los ficheros de calibración y evaluación ya están agrupados,
deben tener el mismo formato que genera el toolkit en `grouped/`:

```text
m  media_estimada  media_empirica
```

En este modo:

* se pasa `K`
* si aparece `--remain`, la ruta usa `t1`; si no aparece, usa `t0`
* `K` y `t` no se usan para reagrupar ni para recalcular los datos; sólo
  se guardan en `metadata.json` y en el nombre de la carpeta del
  experimento
* no se genera la carpeta superior `grouped/` con una nueva agrupación
  de calibración/evaluación
* `lm` y `sx` ajustan el modelo con la calibración agrupada y evalúan
  sobre la evaluación agrupada
* `ts` calcula las métricas directamente sobre la evaluación agrupada;
  no optimiza `T`, porque el fichero agrupado ya no contiene las
  probabilidades por clase necesarias para recalibrar temperatura

Ejemplos:

```bash
estimator lm grouped/calibrationK100 grouped/evaluationK100 results 100 \
  --grouped
estimator sx grouped/calibrationK100 grouped/evaluationK100 results 100 \
  --grouped \
  --trim:1.5
estimator ts grouped/calibrationK100 grouped/evaluationK100 results 100 \
  --grouped
```

---

## 6. Cálculos que realiza

### 6.1 Agrupación

Los datos se ordenan por el valor estimado y se dividen en `K` bloques.

Sea `n` el número total de muestras:

```text
M = n // K
r = n % K
```

Para cada bloque se calcula:

* tamaño (`m_i`)
* media estimada (`x_i`)
* media empírica (`y_i`)

Tratamiento del resto:

* con `--remain`: bloque final de tamaño `r`
* sin `--remain`: último bloque de tamaño `M + r`

---

### 6.2 Ajuste `lm`

Se ajusta:

```text
f(x) = a * x^b
```

Minimizando:

```text
sum_i m_i * (y_i - a*x_i^b)^2
```

Usando `least_squares` (método LM).

---

### 6.3 Ajuste `sx`

Se ajusta:

```text
f(x) = a * x^b
```

Minimizando:

```text
sum_i m_i * |y_i - a*x_i^b|
```

Usando `minimize` (Nelder-Mead).

---

### 6.4 Trimming de outliers

Solo en `lm` y `sx`.

Proceso:

1. Ajuste inicial
2. Cálculo de residuos ponderados
3. Eliminación de outliers fuera de:

```text
[media - lim*std, media + lim*std]
```

4. Reajuste del modelo

---

### 6.5 Temperature scaling

Se aplica temperatura `T`:

```text
p_i' = p_i^(1/T) / sum_j p_j^(1/T)
```

El error calibrado por muestra tras aplicar `T` es:

```text
1 - max(p_i)
```

Optimización de `T`:

* `ce`: entropía cruzada
* `dE`: error absoluto medio
* `DE`: diferencia entre medias

---

### 6.6 Métricas finales (output)

Calculadas sobre la evaluación agrupada:

* `Ecal`: error calibrado (%)
* `Ee`: error estimado de entrada (%), media de `1-Pmax`
* `E`: error empírico (%)
* `DE`: |Ecal - E| (%)
* `dE`: error absoluto medio (%)
* `rDE`: DE / E
* `rdE`: dE / E

Además:

* `lm`, `sx`: guardan `a`, `b`
* `ts`: guarda `T`

---

## 7. Salidas generadas

Cada ejecución crea una carpeta dentro de `outdir`.

### 7.1 `lm`

```text
outdir/
└── lm_K{K}_t{0|1}_tr{0|1}_lim{valor}/
```

### 7.2 `sx`

```text
outdir/
└── simplex_K{K}_t{0|1}_tr{0|1}_lim{valor}/
```

### 7.3 `ts`

```text
outdir/
└── ts_K{K}_t{0|1}_{ce|dE|DE}/
```

Con `--grouped`:

```text
outdir/
├── lm_grouped_K{K}_t{0|1}_tr{0|1}_lim{valor}/
├── simplex_grouped_K{K}_t{0|1}_tr{0|1}_lim{valor}/
└── ts_grouped_K{K}_t{0|1}/
```

Contenido común:

* `metadata.json`
* `predictions/`
* `plots/`
* `scripts/`

Además, cuando no se usa `--grouped`, se genera:

* `grouped/`

Opcional:

* `trim/` (si se activa trimming)

---

### 7.4 `metadata.json`

Incluye:

* inputs del experimento
* parámetros usados
* métricas finales (`Ecal`, `Ee`, `E`, `DE`, `dE`, `rDE`, `rdE`)
* parámetros del modelo (`a,b` o `T`)
* resultados de trimming (si aplica)

---

## 8. Ejemplos de uso

En base ficheros de `examples/`

### 8.1 `lm`

```bash
estimator lm examples/calibration.txt examples/evaluation.txt \
  examples/results_lm 3 --remain --trim:1.5
```

### 8.2 `sx`

```bash
estimator sx examples/calibration.txt examples/evaluation.txt \
  examples/results_sx 3 --remain
```

### 8.3 `ts`

```bash
estimator ts examples/calibration_ts.txt examples/evaluation_ts.txt \
  examples/results_ts 3 dE --remain
```

### 8.4 Ficheros ya agrupados

```bash
estimator lm examples/results_lm/lm_K3_t1_tr1_lim1.5/grouped/calibrationK3 \
  examples/results_lm/lm_K3_t1_tr1_lim1.5/grouped/evaluationK3 \
  examples/results_lm_grouped 3 --grouped --remain --trim:1.5
```

### 8.5 Ejecución de código para generación de gráficos

En los experimentos `lm` y `sx`, dentro de `scripts/` se genera un
script `*.py` por gráfico y otro `*.sh` con el mismo nombre base.

El archivo que se debe editar para cambiar colores, ejes, tamaños o
cualquier otro detalle del gráfico es el `*.py` generado.

El archivo `.sh` se genera solo como ayuda para ejecutar ese `*.py`
usando el Python del entorno creado con `install.sh`, así que no hace
falta lanzar el script Python manualmente.

Al ejecutarlo, se sobrescribe el PDF por defecto que ya existe en la
carpeta `plots/` del experimento.

Ejemplo completo usando los ficheros de `examples/`:

```bash
estimator lm examples/calibration.txt examples/evaluation.txt \
  examples/results_lm 3 --remain --trim:1.5

./examples/results_lm/lm_K3_t1_tr1_lim1.5/scripts/Calib_plot_scriptK3.sh
./examples/results_lm/lm_K3_t1_tr1_lim1.5/scripts/Eval_plot_scriptK3.sh
```
