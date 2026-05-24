# El Promo Hunter — Dataset para Simulación de Embudo de Conversión

Dataset completo y parámetros calibrados del canal de Telegram **El Promo Hunter** (@ElPromoHunter, ~5500 suscriptores) para un proyecto universitario de **Modelos y Simulación** (simulación de eventos discretos).

## Embudo modelado

```
Publicación ──→ Visualización ──→ Clic/Forward ──→ Compra
   (λ posts/h)    (LogNormal)      (Bernoulli p)   (literatura)
```

## Datos clave calibrados

| Parámetro | Valor |
|---|---|
| Período | 2026-01-01 → 2026-05-23 (143 días) |
| Posts totales | 21,097 |
| Posts/día promedio | **146.5** |
| Tiempo entre posts | **Gamma(shape=0.37, scale=13.4) min** |
| Distribución de vistas | **LogNormal(μ=5.726, σ=0.201)** |
| Vistas media/post | 313.3 |
| Vistas mediana/post | 302.0 |
| CTR proxy (forward rate) | **0.118%** |
| Boost por cupón | ×1.13 CTR |

---

## Estructura del repositorio

```
data/
│
│── Datos brutos
│   ├── mensajes_telegram.csv          # 21,097 filas — dataset completo por mensaje
│   ├── mensajes_completos.json        # Igual + texto crudo de cada post
│   ├── estadisticas_canal.json        # Stats nativas de Telegram (API)
│   └── resumen_extraccion.json        # Resumen agregado (vistas/hora/día/categoría)
│
└── Archivos para simulación (prefijo sim_)
    ├── sim_parametros_maestro.json    # ★ Archivo principal — todos los parámetros
    ├── sim_tabla_horaria.csv          # λ, vistas, CTR proxy por hora Colombia
    ├── sim_tabla_dia_semana.csv       # λ, vistas, CTR proxy por día de semana
    ├── sim_tabla_categoria.csv        # Parámetros por categoría de producto
    ├── sim_llegada_por_hora.csv       # Posts esperados por hora (proceso Poisson)
    ├── sim_llegada_por_dia.csv        # Posts esperados por día de semana
    └── sim_lambda_matriz_dia_hora.csv # Matriz λ[día × hora] para NHPP

scripts/
    ├── extractor.py                   # Extrae mensajes de Telegram vía Telethon
    ├── reparser.py                    # Re-parsea JSON con regex ajustados al canal
    └── calibrar_simulacion.py         # Ajusta distribuciones y genera archivos sim_*
```

---

## Parámetros de simulación

### Etapa 1 — Llegada de publicaciones (Poisson no homogéneo)

- **λ global:** 146.5 posts/día ≈ 6.1 posts/hora
- **Tiempo entre posts:** Gamma(shape=0.37, scale=13.4) min  *(media=4.97 min)*
- **Variación horaria:** ver `sim_tabla_horaria.csv` y `sim_lambda_matriz_dia_hora.csv`
- **Días con más actividad:** Domingo y Lunes (~15% más posts que Jueves)

| Hora Colombia | λ posts/día | Vistas promedio |
|---|---|---|
| 8–9h | 5.3 | 339 |
| 10–15h | ~10 | 288–324 |
| 16–20h | ~12 | 292–386 |
| 21–23h | 0.7 | 510–599 |

### Etapa 2 — Distribución de vistas por post

- **Distribución ajustada:** LogNormal(μ=5.726, σ=0.201)  *(mejor KS=0.061)*
- **Media:** 313 vistas · **Mediana:** 302 vistas · **Std:** 101
- **Percentiles:** P5=171 · P25=251 · P75=367 · P95=495

**Factores multiplicadores sobre la media:**

| Variable | Efecto |
|---|---|
| Descuento 36-50% | ×1.02 |
| Descuento 51-65% | ×0.98 |
| Rating 4.8-5.0 | ×1.04 |
| Rating < 3.5 | ×0.88 |
| Con media/imagen | ×1.01 |
| Tecnología | ×1.013 |
| Salud/Belleza | ×0.979 |

### Etapa 3 — Clic en link (proxy forward rate)

> Telegram no provee clics en links. Se usa `forwards/views` como proxy de usuarios que compartieron el post (alta intención de compra).

- **Tasa global:** p = 0.00118 (Bernoulli)
- **Con cupón:** p = 0.00126 · **Sin cupón:** p = 0.00112
- **Distribución:** Bernoulli(p) por cada vista

| Categoría | Forward rate |
|---|---|
| Tecnología | 0.00124 |
| Hogar | 0.00118 |
| Ropa | 0.00112 |
| Salud/Belleza | 0.00109 |

### Etapa 4 — Compra

No observable desde Telegram. Fuentes recomendadas:
- **Amazon Associates** → Informes → Clics y pedidos por tag de afiliado
- **Literatura:** conversión afiliados Amazon ≈ 3–8% de clics resultan en compra
- Para calibrar: `compras / forwards` una vez se tenga data de Associates

---

## Campos del dataset (`mensajes_telegram.csv`)

| Campo | Descripción |
|---|---|
| `message_id` | ID único en Telegram |
| `fecha_utc` / `fecha_col` | Timestamp UTC y hora Colombia (UTC-5) |
| `hora_col` | Hora Colombia (0–23) |
| `dia_semana` | Lunes … Domingo |
| `views` | Vistas acumuladas del post |
| `forwards` | Reenvíos (proxy CTR) |
| `tiene_media` | `True` si tiene imagen/video |
| `rating` | Calificación del producto |
| `precio_original` | Precio original en COP |
| `precio_descuento` | Precio con descuento en COP |
| `pct_descuento` | % de descuento calculado |
| `tiene_cupon` | `True` si tiene código de cupón |
| `codigo_cupon` | Código del cupón |
| `link_amazon` | URL `amzn.to` |
| `categoria` | tecnología / hogar / salud-belleza / ropa / juguetes / deportes / otra |
| `es_producto` | `True` si es post de producto |

> **Nota de precios:** los valores están en **pesos colombianos (COP)**. El punto es separador de miles. Ejemplo: `$105.407` = 105,407 COP ≈ $26 USD.

---

## Cómo usar los archivos de simulación

```python
import json, pandas as pd

# Cargar todos los parámetros
with open("data/sim_parametros_maestro.json") as f:
    params = json.load(f)

# Distribución de vistas
mu    = params["vistas_por_post"]["lognormal_mu"]     # 5.726
sigma = params["vistas_por_post"]["lognormal_sigma"]  # 0.201
# numpy: np.random.lognormal(mu, sigma)

# Tasa de llegada por hora
lambda_hora = params["llegada_publicaciones"]["lambda_por_hora_global"]
# lambda_hora["10"] → posts esperados a las 10h

# CTR proxy
p_ctr = params["forwards_ctr_proxy"]["fwd_rate_global"]  # 0.00118

# Tabla horaria completa
df_hora = pd.read_csv("data/sim_tabla_horaria.csv")
```

---

## Reproducir la extracción

```bash
pip install telethon pandas openpyxl scipy matplotlib
python scripts/extractor.py        # Requiere api_id, api_hash de my.telegram.org/apps
python scripts/reparser.py         # Re-parsea sin reconectarse a Telegram
python scripts/calibrar_simulacion.py  # Regenera todos los archivos sim_*
```
