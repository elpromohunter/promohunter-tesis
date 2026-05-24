# El Promo Hunter — Dataset para Simulación de Embudo de Conversión

Dataset completo del canal de Telegram **@ElPromoHunter** (~5500 suscriptores) cruzado con datos reales de **Amazon Associates**, para calibrar un modelo de simulación de eventos discretos.

**Período:** 1 enero → 23 mayo 2026 (143 días)  
**Fuentes:** API de Telegram (Telethon) + Amazon Associates (11 tracking IDs)

---

## El embudo modelado

```
[1] PUBLICACIÓN ──→ [2] VISTA ──────→ [3] CLIC ────→ [4] COMPRA
    146.5 posts/día    LogNormal          p = 1.46%      p = 6.25%
    Gamma inter-arr.   μ=5.726 σ=0.201   1/68 vistas    1/16 clics
                       media = 313 vis.                  $16.09/compra
```

Todos los parámetros de arriba están calibrados con datos reales. El archivo `sim_embudo_completo.json` los tiene listos para usar en el simulador.

---

## Mapa de datos — qué hay y cómo se relaciona

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FUENTE: TELEGRAM                             │
│                                                                     │
│  mensajes_telegram.csv          ←── archivo principal               │
│  │  21,097 filas, una por post                                      │
│  │  clave: message_id + fecha_col                                   │
│  │                                                                  │
│  ├─→ resumen_extraccion.json    agrega por hora/día/categoría       │
│  ├─→ sim_tabla_horaria.csv      λ y vistas por hora Colombia        │
│  ├─→ sim_tabla_dia_semana.csv   λ y vistas por día de semana        │
│  ├─→ sim_tabla_categoria.csv    métricas por categoría producto     │
│  ├─→ sim_lambda_matriz_dia_hora.csv  matriz λ[día × hora]           │
│  ├─→ sim_llegada_por_hora.csv   posts esperados por hora            │
│  └─→ sim_llegada_por_dia.csv    posts esperados por día             │
│                                                                     │
│  sim_crecimiento_suscriptores.csv                                   │
│     93 días de suscriptores totales + altas + bajas diarias         │
│     clave: fecha  (cruza con mensajes_telegram por fecha_col)       │
│                                                                     │
│  estadisticas_canal.json        respuesta cruda de la API Telegram  │
│  mensajes_completos.json        igual que el CSV + texto del post   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     FUENTE: AMAZON ASSOCIATES                       │
│                                                                     │
│  sim_amazon_por_tracking_y_periodo.csv                              │
│     22 filas (11 tracking IDs × 2 períodos)                        │
│     clave: tracking_id                                              │
│                                                                     │
│  sim_amazon_por_tracking_combinado.csv                              │
│     11 filas, un tracking ID por fila, P1 y P2 lado a lado         │
│     clave: tracking_id                                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  ARCHIVO INTEGRADOR (usa todo lo anterior)          │
│                                                                     │
│  sim_embudo_completo.json  ←── empieza aquí para la simulación      │
│  sim_parametros_maestro.json   distribuciones ajustadas completas   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Cruces posibles entre datasets

### Cruce 1 — Posts vs. suscriptores del día
**Pregunta:** ¿Los días con más posts tienen más crecimiento de suscriptores?  
**Clave de unión:** `fecha_col` (mensajes) ↔ `fecha` (suscriptores)

```python
import pandas as pd

msgs = pd.read_csv("data/mensajes_telegram.csv")
subs = pd.read_csv("data/sim_crecimiento_suscriptores.csv")

# Agregar posts por día
posts_dia = msgs.groupby(msgs["fecha_col"].str[:10]).agg(
    n_posts=("message_id", "count"),
    vistas_dia=("views", "sum"),
    vistas_media=("views", "mean"),
).reset_index().rename(columns={"fecha_col": "fecha"})

df = posts_dia.merge(subs, on="fecha", how="inner")
# df tiene: n_posts, vistas_dia, nuevos_suscriptores, bajas, total_suscriptores
```

---

### Cruce 2 — Vistas por hora vs. clics Amazon estimados
**Pregunta:** ¿Qué hora del día genera más clics y revenue?  
**Clave de unión:** `hora_col` → escalar CTR por hora

```python
import pandas as pd, json

msgs  = pd.read_csv("data/mensajes_telegram.csv")
tabla = pd.read_csv("data/sim_tabla_horaria.csv")

# tabla ya tiene: hora_col, vistas_media, fwd_rate
# Añadimos CTR global de Amazon (1.46%) escalado por factor de hora
with open("data/sim_parametros_maestro.json") as f:
    params = json.load(f)

factor_hora = params["vistas_por_post"]["factor_multiplicador_hora"]
CTR_GLOBAL  = 0.01459   # vista → clic (Amazon Associates)
CONV_GLOBAL = 0.0625    # clic → compra

tabla["ctr_estimado"]     = tabla["hora_col"].astype(str).map(factor_hora) * CTR_GLOBAL
tabla["clics_esperados"]  = tabla["vistas_media"] * tabla["ctr_estimado"]
tabla["compras_esperadas"] = tabla["clics_esperados"] * CONV_GLOBAL
tabla["revenue_esperado"]  = tabla["compras_esperadas"] * 16.09  # USD por compra
```

---

### Cruce 3 — Categoría de producto vs. conversión
**Pregunta:** ¿Las publicaciones de tecnología convierten mejor que hogar?  
**Clave de unión:** `categoria` (mensajes) → escalar con datos Amazon por tracking ID

```python
import pandas as pd

msgs = pd.read_csv("data/mensajes_telegram.csv")
cat  = pd.read_csv("data/sim_tabla_categoria.csv")

# cat ya tiene: categoria, vistas_media, pct_descuento_medio, pct_con_cupon, fwd_rate
# Cruzar con métricas post a post
por_cat = msgs[msgs["es_producto"]].groupby("categoria").agg(
    n_posts     = ("message_id", "count"),
    vistas_med  = ("views", "mean"),
    desc_medio  = ("pct_descuento", "mean"),
    rating_med  = ("rating", "mean"),
    pct_cupon   = ("tiene_cupon", "mean"),
).reset_index()

# Nota: la conversión real por categoría no está disponible en Amazon Associates
# (la API no desglosa por categoría de link). Se asume CTR_GLOBAL para todas.
```

---

### Cruce 4 — Cupón vs. vistas vs. compras estimadas
**Pregunta:** ¿Cuánto impacto tiene el cupón en el embudo completo?

```python
import pandas as pd

msgs = pd.read_csv("data/mensajes_telegram.csv")
prod = msgs[msgs["es_producto"]]

CTR   = 0.01459
CONV  = 0.0625
REV   = 16.09

resumen = prod.groupby("tiene_cupon").agg(
    n_posts      = ("message_id", "count"),
    vistas_media = ("views", "mean"),
).assign(
    clics_esperados   = lambda d: d["vistas_media"] * CTR,
    compras_esperadas = lambda d: d["vistas_media"] * CTR * CONV,
    revenue_esperado  = lambda d: d["vistas_media"] * CTR * CONV * REV,
)
print(resumen)
```

---

### Cruce 5 — Crecimiento de suscriptores vs. revenue estimado
**Pregunta:** ¿A más suscriptores, más vistas y más revenue?  
**Clave de unión:** `fecha`

```python
import pandas as pd

subs  = pd.read_csv("data/sim_crecimiento_suscriptores.csv")
msgs  = pd.read_csv("data/mensajes_telegram.csv")

posts_dia = msgs.groupby(msgs["fecha_col"].str[:10]).agg(
    vistas_dia=("views", "sum"),
    n_posts=("message_id", "count"),
).reset_index().rename(columns={"fecha_col": "fecha"})

df = subs.merge(posts_dia, on="fecha", how="inner")
df["revenue_dia_est"] = df["vistas_dia"] * 0.01459 * 0.0625 * 16.09
df["revenue_por_suscriptor"] = df["revenue_dia_est"] / df["total_suscriptores"]
```

---

## Parámetros listos para el simulador

```python
import json

with open("data/sim_embudo_completo.json") as f:
    p = json.load(f)

# Etapa 1 — llegada de posts
lambda_por_hora = p["telegram"]["posts_por_dia"] / 24          # promedio
# O usar la matriz hora×día para Poisson no homogéneo:
import pandas as pd
lambda_matriz = pd.read_csv("data/sim_lambda_matriz_dia_hora.csv", index_col=0)

# Etapa 2 — vistas por post
import numpy as np
mu    = p["embudo_etapas"]["etapa1_publicacion_a_vista"]["distribucion"].split("mu=")[1].split(",")[0]
sigma = p["embudo_etapas"]["etapa1_publicacion_a_vista"]["distribucion"].split("sigma=")[1].rstrip(")")
vistas_simuladas = np.random.lognormal(float(mu), float(sigma), size=1000)

# Etapa 3 — clics
p_clic   = p["embudo_etapas"]["etapa2_vista_a_clic"]["tasa_decimal"]      # 0.01459
clics    = np.random.binomial(n=vistas_simuladas.astype(int), p=p_clic)

# Etapa 4 — compras
p_compra = p["embudo_etapas"]["etapa3_clic_a_compra"]["tasa_decimal"]     # 0.0625
compras  = np.random.binomial(n=clics, p=p_compra)

# Revenue
ingreso_medio = p["embudo_etapas"]["etapa4_compra"]["ingreso_medio_por_compra_usd"]  # 16.09
revenue = compras * ingreso_medio
```

---

## Inventario completo de archivos

| Archivo | Filas | Granularidad | Descripción |
|---|---|---|---|
| `mensajes_telegram.csv` | 21,097 | 1 fila = 1 post | Dataset principal. Views, forwards, precio, rating, categoría, cupón, link |
| `mensajes_completos.json` | 21,097 | 1 objeto = 1 post | Igual + texto completo del post |
| `sim_crecimiento_suscriptores.csv` | 93 | 1 fila = 1 día | Total suscriptores, altas, bajas, neto diario |
| `sim_tabla_horaria.csv` | 24 | 1 fila = 1 hora | λ, vistas, CTR proxy por hora Colombia |
| `sim_tabla_dia_semana.csv` | 7 | 1 fila = 1 día | λ, vistas por día de semana |
| `sim_tabla_categoria.csv` | 7 | 1 fila = 1 categoría | Métricas por categoría de producto |
| `sim_lambda_matriz_dia_hora.csv` | 7×24 | celda = λ[día,hora] | Posts esperados por combinación día+hora |
| `sim_llegada_por_hora.csv` | 24 | 1 fila = 1 hora | Posts totales observados y λ por hora |
| `sim_llegada_por_dia.csv` | 7 | 1 fila = 1 día | Posts totales observados y λ por día |
| `sim_amazon_por_tracking_y_periodo.csv` | 22 | tracking × período | Clics, compras, conversión por ID y período |
| `sim_amazon_por_tracking_combinado.csv` | 11 | 1 fila = 1 tracking ID | P1 y P2 lado a lado para comparar |
| `sim_embudo_completo.json` | — | canal completo | **Archivo principal del embudo** — todos los parámetros integrados |
| `sim_parametros_maestro.json` | — | canal completo | Distribuciones ajustadas, factores, percentiles |
| `estadisticas_canal.json` | — | canal completo | Respuesta cruda de la API de Telegram |
| `resumen_extraccion.json` | — | canal completo | Vistas/hora/día/categoría agregadas |

---

## Campos del dataset principal (`mensajes_telegram.csv`)

| Campo | Tipo | Descripción |
|---|---|---|
| `message_id` | int | ID único del post en Telegram |
| `fecha_utc` | datetime | Timestamp UTC |
| `fecha_col` | datetime | Timestamp hora Colombia (UTC−5) |
| `hora_col` | int | Hora 0–23 hora Colombia |
| `dia_semana` | str | Lunes … Domingo |
| `dia_num` | int | 0=Lunes … 6=Domingo |
| `views` | int | Vistas acumuladas |
| `forwards` | int | Reenvíos (proxy de alta intención) |
| `tiene_media` | bool | `True` si tiene imagen o video |
| `rating` | float | Calificación del producto (ej: 4.5) |
| `precio_original` | float | Precio original en COP |
| `precio_descuento` | float | Precio con descuento en COP |
| `pct_descuento` | float | % de descuento calculado |
| `tiene_cupon` | bool | `True` si tiene código cupón |
| `codigo_cupon` | str | Código del cupón (si aplica) |
| `link_amazon` | str | URL `amzn.to` del post |
| `categoria` | str | tecnología / hogar / salud-belleza / ropa / juguetes / deportes / otra |
| `es_producto` | bool | `True` si es post de producto (vs. mensaje informativo) |

> **Precios en COP:** el punto es separador de miles. `$105.407` = 105,407 pesos ≈ $26 USD.

---

## KPIs globales del canal (143 días)

| KPI | Valor |
|---|---|
| Posts publicados | 21,097 |
| Vistas totales | 6,613,127 |
| Clics en Amazon (estimado) | 96,507 |
| Compras generadas | 6,027 |
| Ingresos totales | $96,974 USD |
| Ganancias afiliado | $3,149 USD |
| Suscriptores: inicio → fin | 4,202 → 5,690 (+35.4%) |
| Ganancia por post publicado | $0.15 USD |

---

## Reproducir todo desde cero

```bash
pip install telethon pandas openpyxl scipy matplotlib

# 1. Extraer mensajes de Telegram
python scripts/extractor.py
# Pide api_id, api_hash → obtener en https://my.telegram.org/apps

# 2. Re-parsear el texto de los posts con regex ajustados al canal
python scripts/reparser.py

# 3. Calibrar distribuciones para simulación
python scripts/calibrar_simulacion.py
```
