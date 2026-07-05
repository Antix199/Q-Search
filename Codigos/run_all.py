"""Ejecuta 25 corridas de los 3 modelos de búsqueda por nivel de qubits.

  - Clásico:  2, 4, 6, 8, 10, 12 qubits
  - Aer:      2, 4, 6, 8, 10, 12 qubits
  - QRydDemo: 2, 4, 6 qubits  (8+ excede el límite del emulador)
"""

import time
from pathlib import Path

from classic_search import busqueda_clasica_masiva
from quantum_search_aer import main as ejecutar_aer

RESULTADOS_DIR = Path(__file__).resolve().parent / "results"
QUBITS_CLASICO_AER = [2, 4, 6, 8, 10, 12]
QUBITS_QRYD = [2, 4, 6]
CORRIDAS = 25


def csv_clasico(bits): return RESULTADOS_DIR / f"results_classic_{bits}q.csv"
def csv_aer(bits):     return RESULTADOS_DIR / f"results_quantum_{bits}q.csv"
def csv_qryd(bits):    return RESULTADOS_DIR / f"results_qryd_{bits}q.csv"


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ejecutar_qryd(bits, csv_path):
    from quantum_search_qryd import main as _qryd
    _qryd(bits=bits, csv_path=csv_path)


def main():
    RESULTADOS_DIR.mkdir(exist_ok=True)
    total_start = time.perf_counter()
    _log(f"=== Inicio: {CORRIDAS} corridas × 6 niveles de qubits ===")

    # ── 1. Clásico ────────────────────────────────────────────────────────────
    _log("--- CLÁSICO ---")
    for bits in QUBITS_CLASICO_AER:
        _log(f"  {bits}q (N={2**bits:,}) – {CORRIDAS} corridas")
        t0 = time.perf_counter()
        for c in range(1, CORRIDAS + 1):
            busqueda_clasica_masiva(bits=bits, csv_path=csv_clasico(bits))
            if c % 5 == 0:
                _log(f"    {bits}q Clásico {c}/{CORRIDAS}")
        _log(f"  {bits}q Clásico listo en {time.perf_counter()-t0:.1f}s")

    # ── 2. Aer ────────────────────────────────────────────────────────────────
    _log("--- AER (simulador local) ---")
    for bits in QUBITS_CLASICO_AER:
        _log(f"  {bits}q (N={2**bits:,}) – {CORRIDAS} corridas")
        t0 = time.perf_counter()
        for c in range(1, CORRIDAS + 1):
            ejecutar_aer(bits=bits, csv_path=csv_aer(bits))
            if c % 5 == 0:
                elapsed = time.perf_counter() - t0
                eta = (elapsed / c) * (CORRIDAS - c)
                _log(f"    {bits}q Aer {c}/{CORRIDAS}  {elapsed:.1f}s  ETA {eta:.0f}s")
        _log(f"  {bits}q Aer listo en {time.perf_counter()-t0:.1f}s")

    # ── 3. QRydDemo ───────────────────────────────────────────────────────────
    _log("--- QRYDDEMO (2, 4, 6 qubits) ---")
    for bits in QUBITS_QRYD:
        _log(f"  {bits}q (N={2**bits:,}) – {CORRIDAS} corridas")
        t0 = time.perf_counter()
        errores = 0
        for c in range(1, CORRIDAS + 1):
            try:
                ejecutar_qryd(bits=bits, csv_path=csv_qryd(bits))
                if c % 5 == 0:
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed / c) * (CORRIDAS - c)
                    _log(f"    {bits}q QRyd {c}/{CORRIDAS}  {elapsed:.1f}s  ETA {eta:.0f}s")
            except Exception as exc:
                errores += 1
                _log(f"    [ERROR] {bits}q QRyd corrida {c}: {exc}")
                if errores >= 5:
                    _log(f"    Saltando {bits}q QRyd por errores repetidos.")
                    break
        _log(f"  {bits}q QRydDemo listo en {time.perf_counter()-t0:.1f}s  (errores: {errores})")

    total = time.perf_counter() - total_start
    _log(f"=== EXPERIMENTOS COMPLETOS en {total/60:.1f} min ===")
    _log("Generando análisis comparativo...")

    import compare_searches
    compare_searches.main()
    _log("=== ANÁLISIS COMPLETO ===")


if __name__ == "__main__":
    main()
