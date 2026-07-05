"""Baseline clasico para comparar contra Grover en simulacion.

Este script ejecuta una busqueda secuencial (fuerza bruta) sobre un espacio
de 2^BITS elementos, mide cuantas iteraciones fueron necesarias para hallar
la llave y guarda metricas de tiempo/iteraciones en CSV para analisis.
"""

import random
import time
import csv
import uuid
import argparse
from datetime import datetime, timezone
from pathlib import Path

RESULTADOS_DIR = Path(__file__).resolve().parent / "results"
RESULTADOS_CSV = RESULTADOS_DIR / "results_classic.csv"
BITS = 10


def _preparar_csv(encabezado, csv_path):
    """Asegura compatibilidad de esquema al guardar resultados historicos."""
    if not csv_path.exists():
        return True

    with csv_path.open("r", encoding="utf-8", newline="") as archivo:
        primera_linea = archivo.readline().strip()

    esperado = ",".join(encabezado)
    if primera_linea == esperado:
        return False

    respaldo = csv_path.with_name(
        f"{csv_path.stem}_legacy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    csv_path.rename(respaldo)
    print(f"[INFO] CSV previo movido a: {respaldo}")
    return True


def guardar_resultado_csv(fila, csv_path=RESULTADOS_CSV):
    """Persiste una corrida clasica con columnas estables para graficacion."""
    csv_path.parent.mkdir(exist_ok=True)
    encabezado = [
        "timestamp",
        "run_id",
        "bits",
        "llave_objetivo_bin",
        "llave_objetivo_int",
        "iteraciones_requeridas",
        "tiempo_busqueda_s",
    ]
    escribir_encabezado = _preparar_csv(encabezado, csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=encabezado)
        if escribir_encabezado:
            writer.writeheader()
        writer.writerow(fila)


def busqueda_clasica_masiva(bits, csv_path=RESULTADOS_CSV):
    """Busca la llave por recorrido lineal y registra costo experimental."""
    run_id = str(uuid.uuid4())
    espacio_busqueda = 2 ** bits
    llave_objetivo = random.randint(0, espacio_busqueda - 1)
    llave_objetivo_bin = format(llave_objetivo, f"0{bits}b")

    print(f"--- Búsqueda Clásica ---")
    print(f"Run ID: {run_id}")
    print(f"Qubits (Bits): {bits}")
    print(f"Espacio total (N): {espacio_busqueda:,}")
    print(f"Llave objetivo: {llave_objetivo_bin} ({llave_objetivo})")
    print(f"Buscando la llave... (Esto puede tomar tiempo en la CPU)")

    inicio_tiempo = time.perf_counter()

    iteraciones_requeridas = 0
    # Modelo clasico de referencia: una consulta por candidato en orden.
    for adivinanza in range(espacio_busqueda):
        iteraciones_requeridas += 1
        if adivinanza == llave_objetivo:
            break

    fin_tiempo = time.perf_counter()
    tiempo_total = fin_tiempo - inicio_tiempo

    print("\n¡Llave encontrada!")
    print(f"Iteraciones requeridas: {iteraciones_requeridas:,}")
    print(f"Tiempo de CPU: {tiempo_total:.2f} segundos")

    # Se guarda una fila por corrida para series temporales y boxplots.
    guardar_resultado_csv(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "run_id": run_id,
            "bits": bits,
            "llave_objetivo_bin": llave_objetivo_bin,
            "llave_objetivo_int": llave_objetivo,
            "iteraciones_requeridas": iteraciones_requeridas,
            "tiempo_busqueda_s": round(tiempo_total, 6),
        },
        csv_path=csv_path,
    )
    print(f"Resultado guardado en: {csv_path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta la busqueda clasica y guarda resultados en CSV.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=BITS,
        help=f"Cantidad de bits a usar (default: {BITS}).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=RESULTADOS_CSV,
        help=(
            "Ruta del CSV de salida "
            f"(default: {RESULTADOS_CSV})."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    busqueda_clasica_masiva(bits=args.bits, csv_path=args.csv_out)
