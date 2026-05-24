"""
Calibra distribuciones de probabilidad desde los datos del canal El Promo Hunter.
Genera todos los archivos necesarios para la simulación de eventos discretos.

Embudo modelado:
  Publicación → Visualización → Clic (forward como proxy) → Compra estimada
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import (lognorm, expon, gamma, norm, weibull_min,
                         kstest, chi2_contingency)

warnings.filterwarnings("ignore")

DATA  = Path("/home/ubuntu/datos_promohunter")
OUT   = Path("/home/ubuntu/promohunter-tesis/data")
DIAS  = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ── Carga ─────────────────────────────────────────────────────────────────
print("Cargando datos…")
df = pd.read_csv(DATA / "mensajes_telegram.csv", encoding="utf-8-sig")
df["fecha_col"] = pd.to_datetime(df["fecha_col"])
df["fecha_dia"] = df["fecha_col"].dt.date
prod = df[df["es_producto"] == True].copy()
print(f"  {len(df)} mensajes totales, {len(prod)} productos")


# ══════════════════════════════════════════════════════════════════════════
# 1. TASA DE LLEGADA DE PUBLICACIONES (proceso de Poisson no homogéneo)
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/6] Tasas de llegada de publicaciones…")

# Posts por hora por día → lambda_h_d = E[posts en esa hora ese día]
dias_unicos = df["fecha_dia"].nunique()
llegada_hora = (
    df.groupby("hora_col")["message_id"]
    .count()
    .rename("total_posts")
    .reset_index()
)
llegada_hora["lambda_por_dia"] = llegada_hora["total_posts"] / dias_unicos
llegada_hora["lambda_por_hora"] = llegada_hora["lambda_por_dia"]  # ya es por hora

llegada_dia = (
    df.groupby("dia_semana")["message_id"]
    .count()
    .rename("total_posts")
    .reset_index()
)
semanas = dias_unicos / 7
llegada_dia["lambda_por_semana"] = llegada_dia["total_posts"] / semanas
llegada_dia["lambda_por_dia"]    = llegada_dia["lambda_por_semana"] / 1

# Matriz hora × día de semana
matriz_llegada = (
    df.groupby(["dia_semana", "hora_col"])["message_id"]
    .count()
    .unstack(fill_value=0)
)
# Normalizar por número de semanas → posts esperados por celda
lambda_matriz = (matriz_llegada / semanas).round(3)

llegada_out = {
    "descripcion": "Proceso de Poisson no homogéneo. Lambda = posts esperados por hora.",
    "dias_observados": int(dias_unicos),
    "semanas_observadas": round(semanas, 1),
    "posts_por_dia_promedio": round(len(df) / dias_unicos, 1),
    "lambda_por_hora_global": llegada_hora.set_index("hora_col")["lambda_por_dia"].round(3).to_dict(),
    "lambda_por_dia_semana":  llegada_dia.set_index("dia_semana")["lambda_por_dia"].round(3).to_dict(),
    "lambda_matriz_dia_hora": {
        dia: lambda_matriz.loc[dia].to_dict() if dia in lambda_matriz.index else {}
        for dia in DIAS
    },
}

llegada_hora.to_csv(OUT / "sim_llegada_por_hora.csv", index=False)
llegada_dia.to_csv(OUT  / "sim_llegada_por_dia.csv", index=False)
lambda_matriz.round(3).to_csv(OUT / "sim_lambda_matriz_dia_hora.csv")
print(f"  Posts/día promedio: {len(df)/dias_unicos:.1f}")


# ══════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUCIÓN DE VISTAS POR POST
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/6] Ajustando distribución de vistas…")

vistas = prod["views"].dropna().astype(float)
vistas = vistas[vistas > 0]

candidatas = {
    "lognormal": lognorm,
    "gamma":     gamma,
    "exponential": expon,
    "normal":    norm,
    "weibull":   weibull_min,
}

resultados_fit = {}
mejor = None
mejor_ks = 1.0

for nombre, dist in candidatas.items():
    try:
        params = dist.fit(vistas)
        ks_stat, ks_p = kstest(vistas, dist.cdf, args=params)
        resultados_fit[nombre] = {
            "params": list(params),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue":    round(float(ks_p), 4),
        }
        if ks_stat < mejor_ks:
            mejor_ks = ks_stat
            mejor = nombre
    except Exception as e:
        resultados_fit[nombre] = {"error": str(e)}

# Parámetros de la mejor distribución con nombres legibles
params_lognorm = lognorm.fit(vistas)
mu_log  = float(np.log(params_lognorm[2]) + params_lognorm[0]**2/0 ) if False else float(np.mean(np.log(vistas)))
sigma_log = float(np.std(np.log(vistas)))

vistas_out = {
    "descripcion": "Distribución de vistas por publicación (solo posts de producto).",
    "n": int(len(vistas)),
    "media": round(float(vistas.mean()), 1),
    "mediana": round(float(vistas.median()), 1),
    "desv_std": round(float(vistas.std()), 1),
    "p5": round(float(np.percentile(vistas, 5)), 1),
    "p25": round(float(np.percentile(vistas, 25)), 1),
    "p75": round(float(np.percentile(vistas, 75)), 1),
    "p95": round(float(np.percentile(vistas, 95)), 1),
    "max": round(float(vistas.max()), 1),
    "mejor_distribucion": mejor,
    "lognormal_mu":    round(mu_log, 4),
    "lognormal_sigma": round(sigma_log, 4),
    "lognormal_nota":  "X ~ LogNormal(mu, sigma) donde mu=E[ln(X)], sigma=Std[ln(X)]",
    "fits": resultados_fit,
}

# Vistas por hora y por día (para escalar en simulación)
factor_hora = (
    prod.groupby("hora_col")["views"].mean() /
    prod["views"].mean()
).round(4).to_dict()

factor_dia = (
    prod.groupby("dia_semana")["views"].mean() /
    prod["views"].mean()
).round(4).to_dict()

vistas_out["factor_multiplicador_hora"] = factor_hora
vistas_out["factor_multiplicador_dia"]  = factor_dia

# Percentiles completos para tabla de simulación
pcts = np.percentile(vistas, np.arange(0, 101, 5))
vistas_out["percentiles_5pct"] = {f"p{int(p)}": round(float(v), 1) for p, v in zip(np.arange(0, 101, 5), pcts)}

print(f"  Mejor distribución: {mejor} (KS={mejor_ks:.4f})")
print(f"  LogNormal μ={mu_log:.3f}, σ={sigma_log:.3f}")
print(f"  Media={vistas.mean():.1f}, Mediana={vistas.median():.1f}")


# ══════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUCIÓN DE FORWARDS (proxy de CTR)
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/6] Calculando tasas de forward (proxy CTR)…")

prod2 = prod[prod["views"] > 0].copy()
prod2["fwd_rate"] = prod2["forwards"] / prod2["views"]

fwd_rate_global = float(prod2["fwd_rate"].mean())
fwd_rates_hora  = prod2.groupby("hora_col")["fwd_rate"].mean().round(5).to_dict()
fwd_rates_dia   = prod2.groupby("dia_semana")["fwd_rate"].mean().round(5).to_dict()
fwd_rates_cat   = prod2.groupby("categoria")["fwd_rate"].mean().round(5).to_dict()

# ¿El cupón aumenta el CTR?
con_cupon = prod2[prod2["tiene_cupon"] == True]["fwd_rate"].mean()
sin_cupon = prod2[prod2["tiene_cupon"] == False]["fwd_rate"].mean()

forwards_out = {
    "descripcion": "Tasa forward/vistas como proxy de CTR (click-through rate). "
                   "Telegram no provee clics reales en links.",
    "fwd_rate_global": round(fwd_rate_global, 5),
    "fwd_rate_con_cupon": round(float(con_cupon), 5),
    "fwd_rate_sin_cupon": round(float(sin_cupon), 5),
    "multiplicador_cupon": round(float(con_cupon / sin_cupon) if sin_cupon > 0 else 1.0, 3),
    "fwd_rate_por_hora": fwd_rates_hora,
    "fwd_rate_por_dia":  fwd_rates_dia,
    "fwd_rate_por_categoria": fwd_rates_cat,
}

print(f"  CTR proxy global: {fwd_rate_global*100:.3f}%")
print(f"  Con cupón: {con_cupon*100:.3f}%  |  Sin cupón: {sin_cupon*100:.3f}%")


# ══════════════════════════════════════════════════════════════════════════
# 4. EFECTO DE VARIABLES EN VISTAS (factores de la simulación)
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/6] Calculando factores de efecto sobre vistas…")

# Efecto del descuento (agrupado en bins)
prod2["bin_descuento"] = pd.cut(
    prod2["pct_descuento"].fillna(0),
    bins=[0, 20, 35, 50, 65, 100],
    labels=["0-20%", "21-35%", "36-50%", "51-65%", "66-100%"],
    right=True
)
efecto_descuento = prod2.groupby("bin_descuento", observed=True)["views"].agg(["mean", "count"]).round(1)

# Efecto del rating
prod2["bin_rating"] = pd.cut(
    prod2["rating"].fillna(0),
    bins=[0, 3.5, 4.0, 4.5, 4.8, 5.0],
    labels=["<3.5", "3.5-4.0", "4.0-4.5", "4.5-4.8", "4.8-5.0"],
    right=True
)
efecto_rating = prod2.groupby("bin_rating", observed=True)["views"].agg(["mean", "count"]).round(1)

# Efecto de tener media
efecto_media = df.groupby("tiene_media")["views"].agg(["mean", "count"]).round(1)

factores_out = {
    "descripcion": "Factores multiplicadores sobre vistas base para la simulación.",
    "base_vistas": round(float(vistas.mean()), 1),
    "efecto_cupon": {
        "con_cupon":  round(float(prod2[prod2["tiene_cupon"]==True]["views"].mean()), 1),
        "sin_cupon":  round(float(prod2[prod2["tiene_cupon"]==False]["views"].mean()), 1),
        "multiplicador": round(float(
            prod2[prod2["tiene_cupon"]==True]["views"].mean() /
            prod2[prod2["tiene_cupon"]==False]["views"].mean()
        ), 3),
    },
    "efecto_media": {
        str(k): {"vistas_media": row["mean"], "n": int(row["count"])}
        for k, row in efecto_media.iterrows()
    },
    "efecto_descuento_pct": {
        str(k): {"vistas_media": row["mean"], "n": int(row["count"])}
        for k, row in efecto_descuento.iterrows()
    },
    "efecto_rating": {
        str(k): {"vistas_media": row["mean"], "n": int(row["count"])}
        for k, row in efecto_rating.iterrows()
    },
    "efecto_categoria": {
        cat: {
            "vistas_media": round(float(prod2[prod2["categoria"]==cat]["views"].mean()), 1),
            "n": int((prod2["categoria"]==cat).sum()),
            "multiplicador": round(float(
                prod2[prod2["categoria"]==cat]["views"].mean() / vistas.mean()
            ), 3),
        }
        for cat in prod2["categoria"].unique() if pd.notna(cat)
    },
}

print(f"  Efecto cupón: ×{factores_out['efecto_cupon']['multiplicador']}")


# ══════════════════════════════════════════════════════════════════════════
# 5. DISTRIBUCIÓN DE TIEMPO ENTRE PUBLICACIONES (inter-arrival)
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/6] Distribución de tiempo entre publicaciones…")

df_sorted = df.sort_values("fecha_col")
df_sorted["interarrival_min"] = df_sorted["fecha_col"].diff().dt.total_seconds() / 60
ia = df_sorted["interarrival_min"].dropna()
ia = ia[(ia > 0) & (ia < 120)]  # filtrar gaps > 2h (pausa nocturna, no inter-arrival real)

ia_params_exp = expon.fit(ia, floc=0)
ia_params_gamma = gamma.fit(ia, floc=0)
ks_exp   = kstest(ia, expon.cdf,   args=ia_params_exp)[0]
ks_gamma = kstest(ia, gamma.cdf, args=ia_params_gamma)[0]

interarrival_out = {
    "descripcion": "Tiempo entre publicaciones consecutivas (minutos). "
                   "Filtrado: solo gaps ≤ 120 min (excluye pausa nocturna).",
    "n": int(len(ia)),
    "media_min": round(float(ia.mean()), 2),
    "mediana_min": round(float(ia.median()), 2),
    "desv_std_min": round(float(ia.std()), 2),
    "p5_min":  round(float(np.percentile(ia, 5)), 2),
    "p95_min": round(float(np.percentile(ia, 95)), 2),
    "exponential": {
        "scale": round(float(ia_params_exp[1]), 3),
        "ks_stat": round(float(ks_exp), 4),
        "nota": "T ~ Exp(scale). scale = 1/lambda. Lambda = posts/min.",
    },
    "gamma": {
        "shape": round(float(ia_params_gamma[0]), 3),
        "scale": round(float(ia_params_gamma[2]), 3),
        "ks_stat": round(float(ks_gamma), 4),
    },
    "mejor": "exponential" if ks_exp < ks_gamma else "gamma",
}
print(f"  Media inter-arrival: {ia.mean():.2f} min")
print(f"  Mejor fit: {'Exponencial' if ks_exp < ks_gamma else 'Gamma'}")


# ══════════════════════════════════════════════════════════════════════════
# 6. TABLA MAESTRA DE PARÁMETROS PARA SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/6] Generando tabla maestra y archivos finales…")

# Tabla horaria completa para simulación
tabla_horaria = []
for h in range(24):
    sub = prod[prod["hora_col"] == h]
    n = len(sub)
    tabla_horaria.append({
        "hora_col": h,
        "n_posts": n,
        "lambda_posts_por_dia": round(n / dias_unicos, 4),
        "vistas_media": round(float(sub["views"].mean()), 1) if n > 0 else 0,
        "vistas_mediana": round(float(sub["views"].median()), 1) if n > 0 else 0,
        "vistas_std": round(float(sub["views"].std()), 1) if n > 0 else 0,
        "fwd_rate": round(float((sub["forwards"] / sub["views"].replace(0,np.nan)).mean()), 5) if n > 0 else 0,
        "pct_con_cupon": round(float(sub["tiene_cupon"].mean()), 3) if n > 0 else 0,
    })

pd.DataFrame(tabla_horaria).to_csv(OUT / "sim_tabla_horaria.csv", index=False)

# Tabla por categoría
tabla_categoria = []
for cat in sorted(prod["categoria"].unique()):
    sub = prod[prod["categoria"] == cat]
    tabla_categoria.append({
        "categoria": cat,
        "n_posts": len(sub),
        "pct_total": round(len(sub)/len(prod)*100, 1),
        "vistas_media": round(float(sub["views"].mean()), 1),
        "vistas_mediana": round(float(sub["views"].median()), 1),
        "vistas_std": round(float(sub["views"].std()), 1),
        "pct_descuento_medio": round(float(sub["pct_descuento"].mean()), 1),
        "rating_medio": round(float(sub["rating"].mean()), 2),
        "pct_con_cupon": round(float(sub["tiene_cupon"].mean()), 3),
        "fwd_rate": round(float((sub["forwards"]/sub["views"].replace(0,np.nan)).mean()), 5),
    })

pd.DataFrame(tabla_categoria).to_csv(OUT / "sim_tabla_categoria.csv", index=False)

# Tabla por día de semana
tabla_dia = []
for dia in DIAS:
    sub = prod[prod["dia_semana"] == dia]
    tabla_dia.append({
        "dia_semana": dia,
        "n_posts": len(sub),
        "lambda_posts_por_semana": round(len(sub)/semanas, 3),
        "vistas_media": round(float(sub["views"].mean()), 1),
        "vistas_mediana": round(float(sub["views"].median()), 1),
        "vistas_std": round(float(sub["views"].std()), 1),
        "fwd_rate": round(float((sub["forwards"]/sub["views"].replace(0,np.nan)).mean()), 5),
    })

pd.DataFrame(tabla_dia).to_csv(OUT / "sim_tabla_dia_semana.csv", index=False)

# JSON maestro
maestro = {
    "canal": "El Promo Hunter (@ElPromoHunter)",
    "periodo": "2026-01-01 / 2026-05-23",
    "total_mensajes": len(df),
    "total_productos": len(prod),
    "dias_observados": int(dias_unicos),
    "semanas_observadas": round(semanas, 1),
    "llegada_publicaciones": llegada_out,
    "vistas_por_post": vistas_out,
    "forwards_ctr_proxy": forwards_out,
    "factores_sobre_vistas": factores_out,
    "tiempo_entre_publicaciones_min": interarrival_out,
    "embudo_estimado": {
        "descripcion": (
            "Parámetros del embudo de conversión. "
            "Etapa 1→2 usa distribución de vistas. "
            "Etapa 2→3 usa forward_rate como proxy CTR. "
            "Etapa 3→4 (compra) no observable desde Telegram — usar literatura o Amazon Associates."
        ),
        "etapa1_publicacion_a_vista": {
            "distribucion": "LogNormal",
            "mu": round(mu_log, 4),
            "sigma": round(sigma_log, 4),
            "media_esperada": round(float(vistas.mean()), 1),
        },
        "etapa2_vista_a_forward": {
            "tasa_global": round(fwd_rate_global, 5),
            "distribucion": "Bernoulli(p=fwd_rate)",
            "nota": "forward ≈ usuario que compartió el link (proxy de clic alto-intención)",
        },
        "etapa3_forward_a_compra": {
            "tasa_estimada": None,
            "fuente_recomendada": "Amazon Associates → Informes → Clics y pedidos",
            "literatura_referencia": "CTR afiliados Amazon típico: 3-8% de clics resultan en compra",
        },
    },
}

with open(OUT / "sim_parametros_maestro.json", "w", encoding="utf-8") as f:
    json.dump(maestro, f, ensure_ascii=False, indent=2, default=str)

print("\n── Archivos generados ──────────────────────────────────────")
for f in sorted(OUT.glob("sim_*")):
    size = f.stat().st_size
    print(f"  {f.name:<45} {size/1024:>7.1f} KB")

print("\n── Resumen para simulación ─────────────────────────────────")
print(f"  Posts/día promedio:        {len(df)/dias_unicos:.1f}")
print(f"  Inter-arrival medio:       {ia.mean():.1f} min")
print(f"  Vistas: LogNormal(μ={mu_log:.3f}, σ={sigma_log:.3f})")
print(f"  CTR proxy global:          {fwd_rate_global*100:.3f}%")
print(f"  Multiplicador cupón:       ×{factores_out['efecto_cupon']['multiplicador']}")
print("\n✓ Calibración completada.")
