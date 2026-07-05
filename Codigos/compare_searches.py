"""Análisis comparativo exhaustivo: Clásico vs Aer vs QRydDemo vs IBM (HW real).

Métricas calculadas por nivel de qubits:
  1. Iteraciones promedio / mediana / desv. estándar
  2. Tiempo de búsqueda promedio / mediana / desv. estándar
  3. Speedup cuántico vs clásico (iteraciones y tiempo)
  4. Ratio iteraciones requeridas / iteraciones teóricas de Grover
  5. Tasa de éxito (% corridas que encontraron la llave)
  6. Profundidad del circuito transpilado
  7. Número total de compuertas
  8. Compuertas por qubit (complejidad relativa del circuito)
  9. Evaluaciones de k realizadas (coste de barrido cuántico)
 10. Entropía de Shannon de la distribución de mediciones (solo QRydDemo)
 11. Ratio iteraciones clásicas vs esperado teórico (N/2)

Gráficos generados:
  Fig 1  – Iteraciones requeridas vs qubits  (escala log, + curvas O(N) y O(√N))
  Fig 2  – Tiempo de búsqueda vs qubits      (escala log)
  Fig 3  – Speedup cuántico vs clásico       (iteraciones y tiempo)
  Fig 4  – Profundidad del circuito vs qubits
  Fig 5  – Número de compuertas vs qubits
  Fig 6  – Tasa de éxito cuántica vs qubits
  Fig 7  – Eficiencia cuántica                (iter_requeridas / iter_teoricas)
  Fig 8  – Tabla comparativa resumen          (matplotlib table)
"""

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Configuración ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "results"
GRAFICOS_DIR = RESULTADOS_DIR / "graficos"
QUBITS = [2, 4, 6, 8, 10, 12]

COLOR_CLASICO = "#E74C3C"
COLOR_AER = "#2980B9"
COLOR_QRYD = "#27AE60"
COLOR_IBM = "#E67E22"
COLOR_TEORICO = "#95A5A6"

ESTILO_CLASICO = {"color": COLOR_CLASICO, "marker": "o", "linewidth": 2, "markersize": 7}
ESTILO_AER = {"color": COLOR_AER, "marker": "s", "linewidth": 2, "markersize": 7}
ESTILO_QRYD = {"color": COLOR_QRYD, "marker": "^", "linewidth": 2, "markersize": 7}
ESTILO_IBM = {"color": COLOR_IBM, "marker": "D", "linewidth": 2, "markersize": 7}
ESTILO_TEORICO = {"color": COLOR_TEORICO, "linestyle": "--", "linewidth": 1.5}


# ─── Lectura de CSV ───────────────────────────────────────────────────────────

def leer_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    filas = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for fila in csv.DictReader(fh):
            filas.append(fila)
    return filas


def _float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val, default=None):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _bool(val):
    return str(val).strip().lower() in ("true", "1", "yes")


# ─── Extracción de métricas por sistema ───────────────────────────────────────

def metricas_clasico(bits: int) -> dict | None:
    path = RESULTADOS_DIR / f"results_classic_{bits}q.csv"
    filas = leer_csv(path)
    if not filas:
        return None

    iters = [_int(f["iteraciones_requeridas"]) for f in filas if _int(f.get("iteraciones_requeridas")) is not None]
    tiempos = [_float(f["tiempo_busqueda_s"]) for f in filas if _float(f.get("tiempo_busqueda_s")) is not None]
    if not iters:
        return None

    n = 2 ** bits
    return {
        "n_corridas": len(iters),
        "iter_promedio": statistics.mean(iters),
        "iter_mediana": statistics.median(iters),
        "iter_std": statistics.stdev(iters) if len(iters) > 1 else 0,
        "iter_min": min(iters),
        "iter_max": max(iters),
        "tiempo_promedio": statistics.mean(tiempos) if tiempos else None,
        "tiempo_mediana": statistics.median(tiempos) if tiempos else None,
        "tiempo_std": statistics.stdev(tiempos) if len(tiempos) > 1 else 0,
        "iter_esperado_teorico": n / 2,
        "ratio_vs_esperado": statistics.mean(iters) / (n / 2),
    }


def metricas_aer(bits: int) -> dict | None:
    path = RESULTADOS_DIR / f"results_quantum_{bits}q.csv"
    filas = leer_csv(path)
    if not filas:
        return None

    iter_teorica_grover = math.floor((math.pi / 4) * math.sqrt(2 ** bits))

    iters, tiempos, profundidades, encontradas, evaluaciones = [], [], [], [], []
    for f in filas:
        v = _int(f.get("iteraciones_requeridas"))
        if v is not None:
            iters.append(v)
        t = _float(f.get("tiempo_busqueda_s"))
        if t is not None:
            tiempos.append(t)
        p = _int(f.get("profundidad"))
        if p is not None:
            profundidades.append(p)
        encontradas.append(_bool(f.get("encontrada", "False")))
        ev = _int(f.get("evaluaciones_realizadas"))
        if ev is not None:
            evaluaciones.append(ev)

    if not iters:
        return None

    shots = _int(filas[0].get("shots", 1024)) or 1024
    freq_llaves = [_int(f.get("frecuencia_llave", 0)) or 0 for f in filas]
    tasas = [fk / shots for fk in freq_llaves]

    return {
        "n_corridas": len(iters),
        "iter_promedio": statistics.mean(iters),
        "iter_mediana": statistics.median(iters),
        "iter_std": statistics.stdev(iters) if len(iters) > 1 else 0,
        "iter_min": min(iters),
        "iter_max": max(iters),
        "iter_teorica_grover": iter_teorica_grover,
        "ratio_vs_teorico": statistics.mean(iters) / iter_teorica_grover if iter_teorica_grover else None,
        "tiempo_promedio": statistics.mean(tiempos) if tiempos else None,
        "tiempo_mediana": statistics.median(tiempos) if tiempos else None,
        "tiempo_std": statistics.stdev(tiempos) if len(tiempos) > 1 else 0,
        "tasa_exito": statistics.mean(tasas),
        "tasa_exito_pct": statistics.mean(tasas) * 100,
        "n_encontradas": sum(encontradas),
        "profundidad_promedio": statistics.mean(profundidades) if profundidades else None,
        "evaluaciones_promedio": statistics.mean(evaluaciones) if evaluaciones else None,
        "shots": shots,
    }


def _metricas_extendidas(path: Path, bits: int) -> dict | None:
    """Métricas para backends con CSV extendido (QRydDemo e IBM son idénticos)."""
    filas = leer_csv(path)
    if not filas:
        return None

    iter_teorica_grover = math.floor((math.pi / 4) * math.sqrt(2 ** bits))

    iters, tiempos, profundidades, compuertas, encontradas, evaluaciones, tasas = [], [], [], [], [], [], []
    entropias = []

    for f in filas:
        v = _int(f.get("iteraciones_requeridas"))
        if v is not None:
            iters.append(v)
        t = _float(f.get("tiempo_busqueda_s"))
        if t is not None:
            tiempos.append(t)
        p = _int(f.get("profundidad_circuito"))
        if p is not None:
            profundidades.append(p)
        g = _int(f.get("num_compuertas"))
        if g is not None:
            compuertas.append(g)
        encontradas.append(_bool(f.get("encontrada", "False")))
        ev = _int(f.get("evaluaciones_realizadas"))
        if ev is not None:
            evaluaciones.append(ev)
        te = _float(f.get("tasa_exito"))
        if te is not None:
            tasas.append(te)
        # Entropía de Shannon de la distribución de mediciones
        dist_json = f.get("distribucion_mediciones", "")
        if dist_json:
            try:
                dist = json.loads(dist_json)
                total = sum(dist.values())
                if total > 0:
                    probs = [v / total for v in dist.values() if v > 0]
                    entropia = -sum(p * math.log2(p) for p in probs)
                    entropias.append(entropia)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    if not iters:
        return None

    return {
        "n_corridas": len(iters),
        "iter_promedio": statistics.mean(iters),
        "iter_mediana": statistics.median(iters),
        "iter_std": statistics.stdev(iters) if len(iters) > 1 else 0,
        "iter_min": min(iters),
        "iter_max": max(iters),
        "iter_teorica_grover": iter_teorica_grover,
        "ratio_vs_teorico": statistics.mean(iters) / iter_teorica_grover if iter_teorica_grover else None,
        "tiempo_promedio": statistics.mean(tiempos) if tiempos else None,
        "tiempo_mediana": statistics.median(tiempos) if tiempos else None,
        "tiempo_std": statistics.stdev(tiempos) if len(tiempos) > 1 else 0,
        "tasa_exito": statistics.mean(tasas) if tasas else None,
        "tasa_exito_pct": statistics.mean(tasas) * 100 if tasas else None,
        "n_encontradas": sum(encontradas),
        "profundidad_promedio": statistics.mean(profundidades) if profundidades else None,
        "compuertas_promedio": statistics.mean(compuertas) if compuertas else None,
        "compuertas_por_qubit": statistics.mean(compuertas) / bits if compuertas else None,
        "evaluaciones_promedio": statistics.mean(evaluaciones) if evaluaciones else None,
        "entropia_promedio": statistics.mean(entropias) if entropias else None,
    }


def metricas_qryd(bits: int) -> dict | None:
    return _metricas_extendidas(RESULTADOS_DIR / f"results_qryd_{bits}q.csv", bits)


def metricas_ibm(bits: int) -> dict | None:
    """Hardware real IBM: mismo CSV extendido que QRyd → mismas métricas."""
    return _metricas_extendidas(RESULTADOS_DIR / f"results_ibm_{bits}q.csv", bits)


# ─── Recolección de todos los datos ──────────────────────────────────────────

def recolectar_datos() -> dict:
    datos = {}
    for bits in QUBITS:
        datos[bits] = {
            "clasico": metricas_clasico(bits),
            "aer": metricas_aer(bits),
            "qryd": metricas_qryd(bits),
            "ibm": metricas_ibm(bits),
        }
    return datos


# ─── Helpers de graficación ───────────────────────────────────────────────────

def _bits_con_dato(datos, sistema, campo):
    xs, ys = [], []
    for bits in QUBITS:
        m = datos[bits][sistema]
        if m and m.get(campo) is not None:
            xs.append(bits)
            ys.append(m[campo])
    return xs, ys


def _guardar(fig, nombre):
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAFICOS_DIR / nombre
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")


# ─── Fig 1: Iteraciones vs qubits ─────────────────────────────────────────────

def fig_iteraciones(datos):
    fig, ax = plt.subplots(figsize=(10, 6))

    xs_c, ys_c = _bits_con_dato(datos, "clasico", "iter_promedio")
    xs_a, ys_a = _bits_con_dato(datos, "aer", "iter_promedio")
    xs_q, ys_q = _bits_con_dato(datos, "qryd", "iter_promedio")
    xs_i, ys_i = _bits_con_dato(datos, "ibm", "iter_promedio")

    if xs_c:
        ax.plot(xs_c, ys_c, label="Clásico (promedio)", **ESTILO_CLASICO)
        std_c = [datos[b]["clasico"]["iter_std"] for b in xs_c]
        ax.fill_between(xs_c, [y - s for y, s in zip(ys_c, std_c)],
                        [y + s for y, s in zip(ys_c, std_c)],
                        color=COLOR_CLASICO, alpha=0.15)
    if xs_a:
        ax.plot(xs_a, ys_a, label="Aer (promedio)", **ESTILO_AER)
    if xs_q:
        ax.plot(xs_q, ys_q, label="QRydDemo (promedio)", **ESTILO_QRYD)
    if xs_i:
        ax.plot(xs_i, ys_i, label="IBM (hardware real)", **ESTILO_IBM)

    # Curvas teóricas de referencia
    q_ref = np.linspace(2, 12, 100)
    n_ref = 2 ** q_ref
    ax.plot(q_ref, n_ref / 2, label="Teórico clásico O(N/2)", **ESTILO_TEORICO)
    ax.plot(q_ref, (math.pi / 4) * np.sqrt(n_ref),
            label="Teórico Grover O(√N·π/4)", color="#8E44AD", linestyle=":", linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel("Número de qubits / bits", fontsize=12)
    ax.set_ylabel("Iteraciones requeridas (escala log)", fontsize=12)
    ax.set_title("Iteraciones requeridas por búsqueda según número de qubits", fontsize=13, fontweight="bold")
    ax.set_xticks(QUBITS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    _guardar(fig, "fig1_iteraciones_vs_qubits.png")


# ─── Fig 2: Tiempo vs qubits ──────────────────────────────────────────────────

def fig_tiempo(datos):
    fig, ax = plt.subplots(figsize=(10, 6))

    xs_c, ys_c = _bits_con_dato(datos, "clasico", "tiempo_promedio")
    xs_a, ys_a = _bits_con_dato(datos, "aer", "tiempo_promedio")
    xs_q, ys_q = _bits_con_dato(datos, "qryd", "tiempo_promedio")
    xs_i, ys_i = _bits_con_dato(datos, "ibm", "tiempo_promedio")

    if xs_c:
        ax.plot(xs_c, ys_c, label="Clásico", **ESTILO_CLASICO)
        std_c = [datos[b]["clasico"]["tiempo_std"] for b in xs_c]
        ax.fill_between(xs_c, [max(1e-9, y - s) for y, s in zip(ys_c, std_c)],
                        [y + s for y, s in zip(ys_c, std_c)],
                        color=COLOR_CLASICO, alpha=0.15)
    if xs_a:
        ax.plot(xs_a, ys_a, label="Aer (simulador local)", **ESTILO_AER)
        std_a = [datos[b]["aer"]["tiempo_std"] for b in xs_a]
        ax.fill_between(xs_a, [max(1e-9, y - s) for y, s in zip(ys_a, std_a)],
                        [y + s for y, s in zip(ys_a, std_a)],
                        color=COLOR_AER, alpha=0.15)
    if xs_q:
        ax.plot(xs_q, ys_q, label="QRydDemo (emulador)", **ESTILO_QRYD)
    if xs_i:
        ax.plot(xs_i, ys_i, label="IBM (hardware real, incluye cola)", **ESTILO_IBM)

    ax.set_yscale("log")
    ax.set_xlabel("Número de qubits", fontsize=12)
    ax.set_ylabel("Tiempo de búsqueda (s, escala log)", fontsize=12)
    ax.set_title("Tiempo de búsqueda promedio por número de qubits", fontsize=13, fontweight="bold")
    ax.set_xticks(QUBITS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    _guardar(fig, "fig2_tiempo_vs_qubits.png")


# ─── Fig 3: Speedup cuántico ──────────────────────────────────────────────────

def fig_speedup(datos):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    xs_a_i, ys_a_i = [], []
    xs_q_i, ys_q_i = [], []
    xs_i_i, ys_i_i = [], []
    xs_a_t, ys_a_t = [], []
    xs_q_t, ys_q_t = [], []
    xs_i_t, ys_i_t = [], []

    for bits in QUBITS:
        cl = datos[bits]["clasico"]
        ae = datos[bits]["aer"]
        qr = datos[bits]["qryd"]
        ib = datos[bits]["ibm"]

        if cl and ae and cl["iter_promedio"] and ae["iter_promedio"]:
            xs_a_i.append(bits)
            ys_a_i.append(cl["iter_promedio"] / ae["iter_promedio"])
        if cl and qr and cl["iter_promedio"] and qr.get("iter_promedio"):
            xs_q_i.append(bits)
            ys_q_i.append(cl["iter_promedio"] / qr["iter_promedio"])
        if cl and ib and cl["iter_promedio"] and ib.get("iter_promedio"):
            xs_i_i.append(bits)
            ys_i_i.append(cl["iter_promedio"] / ib["iter_promedio"])

        if cl and ae and cl.get("tiempo_promedio") and ae.get("tiempo_promedio"):
            xs_a_t.append(bits)
            ys_a_t.append(cl["tiempo_promedio"] / ae["tiempo_promedio"])
        if cl and qr and cl.get("tiempo_promedio") and qr.get("tiempo_promedio"):
            xs_q_t.append(bits)
            ys_q_t.append(cl["tiempo_promedio"] / qr["tiempo_promedio"])
        if cl and ib and cl.get("tiempo_promedio") and ib.get("tiempo_promedio"):
            xs_i_t.append(bits)
            ys_i_t.append(cl["tiempo_promedio"] / ib["tiempo_promedio"])

    # Speedup teórico esperado = (N/2) / (π/4·√N) = 2√N/π
    q_ref = np.array(QUBITS, dtype=float)
    n_ref = 2.0 ** q_ref
    speedup_teorico = (n_ref / 2) / ((math.pi / 4) * np.sqrt(n_ref))

    ax1.plot(q_ref, speedup_teorico, label="Teórico esperado", **ESTILO_TEORICO)
    if xs_a_i:
        ax1.plot(xs_a_i, ys_a_i, label="Aer vs Clásico", **ESTILO_AER)
    if xs_q_i:
        ax1.plot(xs_q_i, ys_q_i, label="QRydDemo vs Clásico", **ESTILO_QRYD)
    if xs_i_i:
        ax1.plot(xs_i_i, ys_i_i, label="IBM vs Clásico", **ESTILO_IBM)
    ax1.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax1.set_yscale("log")
    ax1.set_xlabel("Número de qubits", fontsize=11)
    ax1.set_ylabel("Speedup (iter. clásicas / cuánticas, log)", fontsize=11)
    ax1.set_title("Speedup en iteraciones\n(cuántico vs clásico)", fontsize=12, fontweight="bold")
    ax1.set_xticks(QUBITS)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    if xs_a_t or xs_q_t or xs_i_t:
        if xs_a_t:
            ax2.plot(xs_a_t, ys_a_t, label="Aer vs Clásico", **ESTILO_AER)
        if xs_q_t:
            ax2.plot(xs_q_t, ys_q_t, label="QRydDemo vs Clásico", **ESTILO_QRYD)
        if xs_i_t:
            ax2.plot(xs_i_t, ys_i_t, label="IBM vs Clásico", **ESTILO_IBM)
        ax2.axhline(1, color="gray", linestyle=":", linewidth=1, label="Sin speedup")
        ax2.set_xlabel("Número de qubits", fontsize=11)
        ax2.set_ylabel("Speedup en tiempo (t_clásico / t_cuántico)", fontsize=11)
        ax2.set_title("Speedup en tiempo de ejecución\n(cuántico vs clásico)", fontsize=12, fontweight="bold")
        ax2.set_xticks(QUBITS)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Sin datos de tiempo\ndisponibles para comparar",
                 ha="center", va="center", fontsize=12, transform=ax2.transAxes)
        ax2.axis("off")

    fig.suptitle("Speedup cuántico vs búsqueda clásica", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _guardar(fig, "fig3_speedup_cuantico.png")


# ─── Fig 4: Profundidad del circuito ─────────────────────────────────────────

def fig_profundidad(datos):
    fig, ax = plt.subplots(figsize=(10, 6))
    xs_a, ys_a = _bits_con_dato(datos, "aer", "profundidad_promedio")
    xs_q, ys_q = _bits_con_dato(datos, "qryd", "profundidad_promedio")
    xs_i, ys_i = _bits_con_dato(datos, "ibm", "profundidad_promedio")

    if xs_a:
        ax.plot(xs_a, ys_a, label="Aer", **ESTILO_AER)
    if xs_q:
        ax.plot(xs_q, ys_q, label="QRydDemo", **ESTILO_QRYD)
    if xs_i:
        ax.plot(xs_i, ys_i, label="IBM (hardware real)", **ESTILO_IBM)

    if not xs_a and not xs_q and not xs_i:
        ax.text(0.5, 0.5, "Sin datos de profundidad disponibles",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
    else:
        ax.set_xlabel("Número de qubits", fontsize=12)
        ax.set_ylabel("Profundidad promedio del circuito transpilado", fontsize=12)
        ax.set_title("Profundidad del circuito de Grover vs qubits", fontsize=13, fontweight="bold")
        ax.set_xticks(QUBITS)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    _guardar(fig, "fig4_profundidad_circuito.png")


# ─── Fig 5: Número de compuertas ─────────────────────────────────────────────

def fig_compuertas(datos):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    xs_q, ys_q = _bits_con_dato(datos, "qryd", "compuertas_promedio")
    xs_i, ys_i = _bits_con_dato(datos, "ibm", "compuertas_promedio")
    xs_cpq_q, ys_cpq_q = _bits_con_dato(datos, "qryd", "compuertas_por_qubit")
    xs_cpq_i, ys_cpq_i = _bits_con_dato(datos, "ibm", "compuertas_por_qubit")

    if xs_q or xs_i:
        if xs_q:
            ax1.plot(xs_q, ys_q, label="QRydDemo", **ESTILO_QRYD)
        if xs_i:
            ax1.plot(xs_i, ys_i, label="IBM (hardware real)", **ESTILO_IBM)
        ax1.set_xlabel("Número de qubits", fontsize=11)
        ax1.set_ylabel("Número promedio de compuertas", fontsize=11)
        ax1.set_title("Compuertas totales vs qubits\n(circuito Grover transpilado)", fontsize=12, fontweight="bold")
        ax1.set_xticks(QUBITS)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Sin datos de compuertas disponibles\n(requiere datos QRydDemo o IBM)",
                 ha="center", va="center", fontsize=11, transform=ax1.transAxes)
        ax1.axis("off")

    if xs_cpq_q or xs_cpq_i:
        ancho = 0.35
        if xs_cpq_q:
            ax2.bar([x - ancho / 2 for x in xs_cpq_q], ys_cpq_q, width=ancho,
                    color=COLOR_QRYD, alpha=0.85, edgecolor="white", label="QRydDemo")
        if xs_cpq_i:
            ax2.bar([x + ancho / 2 for x in xs_cpq_i], ys_cpq_i, width=ancho,
                    color=COLOR_IBM, alpha=0.85, edgecolor="white", label="IBM")
        ax2.set_xlabel("Número de qubits", fontsize=11)
        ax2.set_ylabel("Compuertas por qubit", fontsize=11)
        ax2.set_title("Complejidad relativa del circuito\n(compuertas / qubits)", fontsize=12, fontweight="bold")
        ax2.set_xticks(sorted(set(xs_cpq_q) | set(xs_cpq_i)))
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "Sin datos disponibles",
                 ha="center", va="center", fontsize=11, transform=ax2.transAxes)
        ax2.axis("off")

    fig.suptitle("Complejidad del circuito cuántico", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _guardar(fig, "fig5_compuertas.png")


# ─── Fig 6: Tasa de éxito ─────────────────────────────────────────────────────

def fig_tasa_exito(datos):
    fig, ax = plt.subplots(figsize=(10, 6))
    xs_a, ys_a = _bits_con_dato(datos, "aer", "tasa_exito_pct")
    xs_q, ys_q = _bits_con_dato(datos, "qryd", "tasa_exito_pct")
    xs_i, ys_i = _bits_con_dato(datos, "ibm", "tasa_exito_pct")

    if xs_a:
        ax.plot(xs_a, ys_a, label="Aer", **ESTILO_AER)
    if xs_q:
        ax.plot(xs_q, ys_q, label="QRydDemo", **ESTILO_QRYD)
    if xs_i:
        ax.plot(xs_i, ys_i, label="IBM (hardware real)", **ESTILO_IBM)

    if xs_a or xs_q or xs_i:
        ax.axhline(90, color="orange", linestyle="--", linewidth=1.5, label="Umbral de éxito (90%)")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Número de qubits", fontsize=12)
        ax.set_ylabel("Tasa de éxito (%)", fontsize=12)
        ax.set_title("Tasa de éxito del algoritmo de Grover vs qubits", fontsize=13, fontweight="bold")
        ax.set_xticks(QUBITS)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Sin datos de tasa de éxito disponibles",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
    _guardar(fig, "fig6_tasa_exito.png")


# ─── Fig 7: Eficiencia cuántica ───────────────────────────────────────────────

def fig_eficiencia(datos):
    """Ratio iter_requeridas / iter_teoricas_Grover (1 = óptimo teórico)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    xs_a, ys_a, xs_q, ys_q, xs_i, ys_i = [], [], [], [], [], []
    for bits in QUBITS:
        ae = datos[bits]["aer"]
        qr = datos[bits]["qryd"]
        ib = datos[bits]["ibm"]
        if ae and ae.get("ratio_vs_teorico") is not None:
            xs_a.append(bits)
            ys_a.append(ae["ratio_vs_teorico"])
        if qr and qr.get("ratio_vs_teorico") is not None:
            xs_q.append(bits)
            ys_q.append(qr["ratio_vs_teorico"])
        if ib and ib.get("ratio_vs_teorico") is not None:
            xs_i.append(bits)
            ys_i.append(ib["ratio_vs_teorico"])

    if xs_a:
        ax.plot(xs_a, ys_a, label="Aer", **ESTILO_AER)
    if xs_q:
        ax.plot(xs_q, ys_q, label="QRydDemo", **ESTILO_QRYD)
    if xs_i:
        ax.plot(xs_i, ys_i, label="IBM (hardware real)", **ESTILO_IBM)

    if xs_a or xs_q or xs_i:
        ax.axhline(1, color="gray", linestyle="--", linewidth=1.5, label="Óptimo teórico (ratio = 1)")
        ax.set_xlabel("Número de qubits", fontsize=12)
        ax.set_ylabel("iter_requeridas / iter_teóricas_Grover", fontsize=12)
        ax.set_title("Eficiencia cuántica: distancia al óptimo teórico de Grover",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(QUBITS)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Sin datos de eficiencia cuántica disponibles",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
    _guardar(fig, "fig7_eficiencia_cuantica.png")


# ─── Fig 8: Tabla comparativa resumen ─────────────────────────────────────────

def fig_tabla(datos):
    columnas = ["Qubits", "N", "Clás. iter.\n(prom)", "Aer iter.\n(prom)",
                "QRyd iter.\n(prom)", "IBM iter.\n(prom)", "Grover\nteórico",
                "Speedup iter.\nAer/Clás",
                "Speedup iter.\nQRyd/Clás",
                "Speedup iter.\nIBM/Clás",
                "Tasa éxito\nAer (%)",
                "Tasa éxito\nQRyd (%)",
                "Tasa éxito\nIBM (%)"]

    filas_tabla = []
    for bits in QUBITS:
        n = 2 ** bits
        cl = datos[bits]["clasico"]
        ae = datos[bits]["aer"]
        qr = datos[bits]["qryd"]
        ib = datos[bits]["ibm"]
        teorico = math.floor((math.pi / 4) * math.sqrt(n))

        def fmt_iter(m): return f"{m['iter_promedio']:.1f}" if m else "—"
        def fmt_speedup(cl_m, q_m):
            if cl_m and q_m and q_m.get("iter_promedio"):
                return f"{cl_m['iter_promedio'] / q_m['iter_promedio']:.2f}×"
            return "—"
        def fmt_tasa(m, campo="tasa_exito_pct"):
            v = m.get(campo) if m else None
            return f"{v:.1f}" if v is not None else "—"

        filas_tabla.append([
            str(bits), f"{n:,}",
            fmt_iter(cl), fmt_iter(ae), fmt_iter(qr), fmt_iter(ib),
            str(teorico),
            fmt_speedup(cl, ae), fmt_speedup(cl, qr), fmt_speedup(cl, ib),
            fmt_tasa(ae), fmt_tasa(qr), fmt_tasa(ib),
        ])

    fig, ax = plt.subplots(figsize=(19, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=filas_tabla,
        colLabels=columnas,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECF0F1")
        else:
            cell.set_facecolor("white")

    ax.set_title("Tabla comparativa: Clásico vs Aer vs QRydDemo",
                 fontsize=14, fontweight="bold", pad=20, y=0.85)
    _guardar(fig, "fig8_tabla_comparativa.png")


# ─── Tabla CSV de resumen ─────────────────────────────────────────────────────

def exportar_tabla_csv(datos):
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAFICOS_DIR / "full_comparison_table.csv"
    campos = [
        "bits", "N",
        # Clásico
        "cl_n_corridas", "cl_iter_promedio", "cl_iter_mediana", "cl_iter_std",
        "cl_tiempo_promedio_s", "cl_tiempo_std_s",
        "cl_iter_esperado_N2", "cl_ratio_vs_esperado",
        # Aer
        "aer_n_corridas", "aer_iter_promedio", "aer_iter_mediana", "aer_iter_std",
        "aer_tiempo_promedio_s", "aer_tiempo_std_s",
        "aer_iter_teorica_grover", "aer_ratio_vs_teorico",
        "aer_tasa_exito_pct", "aer_profundidad_promedio",
        "aer_evaluaciones_promedio",
        "aer_speedup_iter_vs_clasico", "aer_speedup_tiempo_vs_clasico",
        # QRydDemo
        "qryd_n_corridas", "qryd_iter_promedio", "qryd_iter_mediana", "qryd_iter_std",
        "qryd_tiempo_promedio_s", "qryd_tiempo_std_s",
        "qryd_iter_teorica_grover", "qryd_ratio_vs_teorico",
        "qryd_tasa_exito_pct", "qryd_profundidad_promedio",
        "qryd_compuertas_promedio", "qryd_compuertas_por_qubit",
        "qryd_evaluaciones_promedio", "qryd_entropia_promedio",
        "qryd_speedup_iter_vs_clasico", "qryd_speedup_tiempo_vs_clasico",
        # IBM (hardware real)
        "ibm_n_corridas", "ibm_iter_promedio", "ibm_iter_mediana", "ibm_iter_std",
        "ibm_tiempo_promedio_s", "ibm_tiempo_std_s",
        "ibm_iter_teorica_grover", "ibm_ratio_vs_teorico",
        "ibm_tasa_exito_pct", "ibm_profundidad_promedio",
        "ibm_compuertas_promedio", "ibm_compuertas_por_qubit",
        "ibm_evaluaciones_promedio", "ibm_entropia_promedio",
        "ibm_speedup_iter_vs_clasico", "ibm_speedup_tiempo_vs_clasico",
    ]

    def val(m, k):
        if m is None:
            return ""
        v = m.get(k)
        if v is None:
            return ""
        if isinstance(v, float):
            return round(v, 6)
        return v

    def speedup_iter(cl, q):
        if cl and q and cl.get("iter_promedio") and q.get("iter_promedio"):
            return round(cl["iter_promedio"] / q["iter_promedio"], 4)
        return ""

    def speedup_tiempo(cl, q):
        if cl and q and cl.get("tiempo_promedio") and q.get("tiempo_promedio"):
            return round(cl["tiempo_promedio"] / q["tiempo_promedio"], 4)
        return ""

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=campos)
        writer.writeheader()
        for bits in QUBITS:
            n = 2 ** bits
            cl = datos[bits]["clasico"]
            ae = datos[bits]["aer"]
            qr = datos[bits]["qryd"]
            ib = datos[bits]["ibm"]
            writer.writerow({
                "bits": bits, "N": n,
                "cl_n_corridas": val(cl, "n_corridas"),
                "cl_iter_promedio": val(cl, "iter_promedio"),
                "cl_iter_mediana": val(cl, "iter_mediana"),
                "cl_iter_std": val(cl, "iter_std"),
                "cl_tiempo_promedio_s": val(cl, "tiempo_promedio"),
                "cl_tiempo_std_s": val(cl, "tiempo_std"),
                "cl_iter_esperado_N2": val(cl, "iter_esperado_teorico"),
                "cl_ratio_vs_esperado": val(cl, "ratio_vs_esperado"),
                "aer_n_corridas": val(ae, "n_corridas"),
                "aer_iter_promedio": val(ae, "iter_promedio"),
                "aer_iter_mediana": val(ae, "iter_mediana"),
                "aer_iter_std": val(ae, "iter_std"),
                "aer_tiempo_promedio_s": val(ae, "tiempo_promedio"),
                "aer_tiempo_std_s": val(ae, "tiempo_std"),
                "aer_iter_teorica_grover": val(ae, "iter_teorica_grover"),
                "aer_ratio_vs_teorico": val(ae, "ratio_vs_teorico"),
                "aer_tasa_exito_pct": val(ae, "tasa_exito_pct"),
                "aer_profundidad_promedio": val(ae, "profundidad_promedio"),
                "aer_evaluaciones_promedio": val(ae, "evaluaciones_promedio"),
                "aer_speedup_iter_vs_clasico": speedup_iter(cl, ae),
                "aer_speedup_tiempo_vs_clasico": speedup_tiempo(cl, ae),
                "qryd_n_corridas": val(qr, "n_corridas"),
                "qryd_iter_promedio": val(qr, "iter_promedio"),
                "qryd_iter_mediana": val(qr, "iter_mediana"),
                "qryd_iter_std": val(qr, "iter_std"),
                "qryd_tiempo_promedio_s": val(qr, "tiempo_promedio"),
                "qryd_tiempo_std_s": val(qr, "tiempo_std"),
                "qryd_iter_teorica_grover": val(qr, "iter_teorica_grover"),
                "qryd_ratio_vs_teorico": val(qr, "ratio_vs_teorico"),
                "qryd_tasa_exito_pct": val(qr, "tasa_exito_pct"),
                "qryd_profundidad_promedio": val(qr, "profundidad_promedio"),
                "qryd_compuertas_promedio": val(qr, "compuertas_promedio"),
                "qryd_compuertas_por_qubit": val(qr, "compuertas_por_qubit"),
                "qryd_evaluaciones_promedio": val(qr, "evaluaciones_promedio"),
                "qryd_entropia_promedio": val(qr, "entropia_promedio"),
                "qryd_speedup_iter_vs_clasico": speedup_iter(cl, qr),
                "qryd_speedup_tiempo_vs_clasico": speedup_tiempo(cl, qr),
                "ibm_n_corridas": val(ib, "n_corridas"),
                "ibm_iter_promedio": val(ib, "iter_promedio"),
                "ibm_iter_mediana": val(ib, "iter_mediana"),
                "ibm_iter_std": val(ib, "iter_std"),
                "ibm_tiempo_promedio_s": val(ib, "tiempo_promedio"),
                "ibm_tiempo_std_s": val(ib, "tiempo_std"),
                "ibm_iter_teorica_grover": val(ib, "iter_teorica_grover"),
                "ibm_ratio_vs_teorico": val(ib, "ratio_vs_teorico"),
                "ibm_tasa_exito_pct": val(ib, "tasa_exito_pct"),
                "ibm_profundidad_promedio": val(ib, "profundidad_promedio"),
                "ibm_compuertas_promedio": val(ib, "compuertas_promedio"),
                "ibm_compuertas_por_qubit": val(ib, "compuertas_por_qubit"),
                "ibm_evaluaciones_promedio": val(ib, "evaluaciones_promedio"),
                "ibm_entropia_promedio": val(ib, "entropia_promedio"),
                "ibm_speedup_iter_vs_clasico": speedup_iter(cl, ib),
                "ibm_speedup_tiempo_vs_clasico": speedup_tiempo(cl, ib),
            })
    print(f"  CSV exportado: {path}")


# ─── Resumen en consola ───────────────────────────────────────────────────────

def imprimir_resumen_consola(datos):
    linea = "─" * 106
    print(f"\n{'═' * 106}")
    print("  ANÁLISIS COMPARATIVO: Clásico | Aer | QRydDemo | IBM (hardware real)")
    print(f"{'═' * 106}")
    for bits in QUBITS:
        n = 2 ** bits
        cl = datos[bits]["clasico"]
        ae = datos[bits]["aer"]
        qr = datos[bits]["qryd"]
        ib = datos[bits]["ibm"]
        teorico = math.floor((math.pi / 4) * math.sqrt(n))

        print(f"\n{linea}")
        print(f"  {bits} qubits  |  N = {n:,}  |  Grover teórico: {teorico} iteraciones")
        print(linea)

        def fmt_corridas(m): return str(m["n_corridas"]) if m else "—"
        def fmti(m): return f"{m['iter_promedio']:.1f} ± {m['iter_std']:.1f}" if m else "—"
        def fmtt(m):
            v = m.get("tiempo_promedio") if m else None
            return f"{v:.4f} s" if v is not None else "—"
        def fmt_speedup(m):
            if cl and m and m.get("iter_promedio"):
                return f"{cl['iter_promedio'] / m['iter_promedio']:.2f}x"
            return "—"
        def fmt_ratio(m):
            v = m.get("ratio_vs_teorico") if m else None
            return f"{v:.2f}" if v is not None else "—"
        def fmt_tasa(m):
            v = m.get("tasa_exito_pct") if m else None
            return f"{v:.1f}%" if v is not None else "—"
        def fmt_prof(m):
            v = m.get("profundidad_promedio") if m else None
            return f"{v:.1f}" if v is not None else "—"
        def fmt_comp(m):
            v = m.get("compuertas_promedio") if m else None
            return f"{v:.1f}" if v is not None else "—"
        def fmt_cpq(m):
            v = m.get("compuertas_por_qubit") if m else None
            return f"{v:.2f}" if v is not None else "—"
        def fmt_entropia(m):
            v = m.get("entropia_promedio") if m else None
            return f"{v:.3f}" if v is not None else "—"

        def fila(nombre, fc, fa, fq, fi):
            print(f"  {nombre:<30}{fc:>18}{fa:>18}{fq:>18}{fi:>18}")

        fila("Métrica", "Clásico", "Aer", "QRydDemo", "IBM")
        print(f"  {'-'*30}{'-'*18}{'-'*18}{'-'*18}{'-'*18}")
        fila("Corridas", fmt_corridas(cl), fmt_corridas(ae), fmt_corridas(qr), fmt_corridas(ib))
        fila("Iter. promedio ± std", fmti(cl), fmti(ae), fmti(qr), fmti(ib))
        fila("Speedup iter. vs Clásico", "—", fmt_speedup(ae), fmt_speedup(qr), fmt_speedup(ib))
        fila("Ratio iter / teórico", "—", fmt_ratio(ae), fmt_ratio(qr), fmt_ratio(ib))
        fila("Tiempo promedio", fmtt(cl), fmtt(ae), fmtt(qr), fmtt(ib))
        fila("Tasa de éxito (%)", "N/A", fmt_tasa(ae), fmt_tasa(qr), fmt_tasa(ib))
        fila("Profundidad circuito", "N/A", fmt_prof(ae), fmt_prof(qr), fmt_prof(ib))
        fila("Número de compuertas", "N/A", fmt_comp(ae), fmt_comp(qr), fmt_comp(ib))
        fila("Compuertas / qubit", "N/A", fmt_cpq(ae), fmt_cpq(qr), fmt_cpq(ib))
        fila("Entropía Shannon medic.", "N/A", fmt_entropia(ae), fmt_entropia(qr), fmt_entropia(ib))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Recolectando datos...")
    datos = recolectar_datos()

    disponibles = {
        sis: any(datos[b][sis] is not None for b in QUBITS)
        for sis in ("clasico", "aer", "qryd", "ibm")
    }
    print(f"  Datos disponibles → Clásico: {disponibles['clasico']}  |  "
          f"Aer: {disponibles['aer']}  |  QRydDemo: {disponibles['qryd']}  |  "
          f"IBM: {disponibles['ibm']}")

    imprimir_resumen_consola(datos)

    print("\nGenerando gráficos...")
    fig_iteraciones(datos)
    fig_tiempo(datos)
    fig_speedup(datos)
    fig_profundidad(datos)
    fig_compuertas(datos)
    fig_tasa_exito(datos)
    fig_eficiencia(datos)
    fig_tabla(datos)

    print("\nExportando tabla CSV...")
    exportar_tabla_csv(datos)

    print(f"\nTodos los archivos guardados en: {GRAFICOS_DIR}")


if __name__ == "__main__":
    main()
