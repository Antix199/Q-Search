"""Experimento de Grover en el simulador QRydDemo (Rydberg atoms).

Adapta quantum_computer.py para ejecutar en el emulador de QRydDemo
usando qiskit-qryd-provider. Registra las métricas extendidas exigidas:
shots, backend, seeds, tasa de éxito, distribución de mediciones,
profundidad del circuito, conteo de compuertas y qubits auxiliares.

Configuración del token de API
───────────────────────────────
Opción A – archivo .env en la raíz del proyecto:
    QRYD_API_TOKEN=<tu_token_aqui>

Opción B – variable de entorno exportada en la sesión:
    export QRYD_API_TOKEN=<tu_token_aqui>

Requisitos:
    pip install qiskit-qryd-provider python-dotenv
"""

import csv
import json
import math
import os
import random
import time
import uuid
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit_qryd_provider import QRydProvider

# ─── Parámetros experimentales ───────────────────────────────────────────────
# QRydDemo emulator: recomendado n ≤ 20 qubits para tiempos razonables.
NUM_QUBITS = 8
SHOTS = 1024
SEED_SIMULADOR = 42
SEED_TRANSPILER = 42
FACTOR_MAX_BUSQUEDA = 2
UMBRAL_FRECUENCIA_OBJETIVO = 0.9
MAX_EVALUACIONES_K = 12
MAX_TIEMPO_BUSQUEDA_S = 180
VERBOSE_ITERACIONES = False

# Nombre del backend en QRydDemo (ajustar según catálogo disponible en la API).
# Opciones típicas: "qryd_emulator_square", "qryd_emulator_triangular"
QRYD_BACKEND_NAME = "qryd_emulator$square"

RESULTADOS_DIR = Path(__file__).resolve().parent / "results"
RESULTADOS_CSV = RESULTADOS_DIR / "results_qryd.csv"

# ─── Carga de token ───────────────────────────────────────────────────────────
load_dotenv()


def obtener_api_token() -> str:
    """Lee QRYD_API_TOKEN desde entorno o .env; lanza EnvironmentError si falta."""
    token = os.environ.get("QRYD_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "QRYD_API_TOKEN no encontrado. "
            "Defínelo en un archivo .env o exporta la variable de entorno:\n"
            "  export QRYD_API_TOKEN=<tu_token>"
        )
    return token


# ─── CSV ─────────────────────────────────────────────────────────────────────
ENCABEZADO_CSV = [
    "timestamp",
    "run_id",
    # backend e identificadores de reproducibilidad
    "backend",
    "backend_version",
    "bits",
    "llave_objetivo_bin",
    "llave_objetivo_int",
    "seed_simulador",
    "seed_transpiler",
    # plan de búsqueda
    "iteraciones_teoricas",
    "iteraciones_maximas_evaluadas",
    "iteraciones_planificadas",
    "evaluaciones_realizadas",
    "corte_temprano",
    "razon_corte",
    "iteraciones_requeridas",
    "iteracion_primer_exito",
    # métricas de medición
    "shots",
    "umbral_frecuencia_objetivo",
    "estado_top1",
    "frecuencia_top1",
    "frecuencia_llave",
    "tasa_exito",               # frecuencia_llave / shots  ∈ [0, 1]
    "encontrada",
    "distribucion_mediciones",  # JSON completo de counts
    # métricas del circuito transpilado
    "profundidad_circuito",
    "num_compuertas",
    "num_qubits_auxiliares",
    "ops_por_compuerta",        # JSON: {gate_name: count}
    # tiempos
    "tiempo_total_s",
    "tiempo_busqueda_s",
]


def _preparar_csv(csv_path: Path) -> bool:
    """Verifica compatibilidad de esquema; respalda CSV obsoleto si el header cambió."""
    if not csv_path.exists():
        return True
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        primera = fh.readline().strip()
    if primera == ",".join(ENCABEZADO_CSV):
        return False
    respaldo = csv_path.with_name(
        f"{csv_path.stem}_legacy_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    csv_path.rename(respaldo)
    print(f"[INFO] CSV previo respaldado en: {respaldo}")
    return True


def guardar_resultado_csv(fila: dict, csv_path: Path = RESULTADOS_CSV) -> None:
    csv_path.parent.mkdir(exist_ok=True)
    escribir_encabezado = _preparar_csv(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ENCABEZADO_CSV)
        if escribir_encabezado:
            writer.writeheader()
        writer.writerow(fila)


# ─── Construcción del circuito ────────────────────────────────────────────────
def construir_oraculo(bits: int, llave: str) -> QuantumCircuit:
    """Oráculo de fase que marca un único estado objetivo sin qubits ancilla.

    Implementación:
      1. X en cada qubit cuyo bit correspondiente en `llave` es '0'
         (convierte el estado objetivo en |11…1⟩).
      2. Puerta MCPhase(π) de n-1 controles → |11…1⟩ adquiere fase -1.
      3. X inverso para restaurar la base computacional.

    Profundidad: O(n) tras descomposición de MCPhase en compuertas nativas.
    """
    oracle = QuantumCircuit(bits)
    for qubit in range(bits):
        if llave[bits - 1 - qubit] == "0":
            oracle.x(qubit)
    oracle.mcp(math.pi, list(range(bits - 1)), bits - 1)
    for qubit in range(bits):
        if llave[bits - 1 - qubit] == "0":
            oracle.x(qubit)
    return oracle


def extraer_metricas_circuito(qc: QuantumCircuit) -> dict:
    """Extrae profundidad, conteo total de compuertas y qubits auxiliares."""
    ops: dict = dict(qc.count_ops())
    return {
        "profundidad": qc.depth(),
        "num_compuertas": sum(ops.values()),
        "num_qubits_auxiliares": qc.num_ancillas,
        "ops_por_compuerta": ops,
    }


def generar_iteraciones_candidatas(iteraciones_maximas: int) -> list[int]:
    """Submuestrea k uniformemente cuando el barrido completo es inviable."""
    if iteraciones_maximas <= 0:
        return [1]
    if iteraciones_maximas <= MAX_EVALUACIONES_K:
        return list(range(1, iteraciones_maximas + 1))
    candidatos = {
        1 + (idx * (iteraciones_maximas - 1)) // (MAX_EVALUACIONES_K - 1)
        for idx in range(MAX_EVALUACIONES_K)
    }
    candidatos.add(iteraciones_maximas)
    return sorted(candidatos)


# ─── Ejecución en QRydDemo ────────────────────────────────────────────────────
def ejecutar_iteracion_grover(backend, oracle: QuantumCircuit, bits: int, k: int) -> dict:
    """Construye el circuito de Grover para k iteraciones, transpila y ejecuta."""
    qc = QuantumCircuit(bits, bits)
    qc.h(range(bits))
    qc.compose(grover_operator(oracle).power(k), inplace=True)
    qc.measure(range(bits), range(bits))

    qc_t = transpile(
        qc,
        backend=backend,
        optimization_level=3,
        seed_transpiler=SEED_TRANSPILER,
    )
    metricas = extraer_metricas_circuito(qc_t)

    job = backend.run(qc_t, shots=SHOTS, seed_simulator=SEED_SIMULADOR)
    conteos: dict = job.result().get_counts(qc_t)
    estado_top1, freq_top1 = max(conteos.items(), key=lambda x: x[1])

    return {
        "estado_top1": estado_top1,
        "frecuencia_top1": freq_top1,
        "conteos": conteos,
        **metricas,
    }


def buscar_iteraciones_necesarias(
    backend, oracle: QuantumCircuit, bits: int, llave_objetivo: str
) -> dict:
    """Explora k candidatas hasta éxito, timeout o fin del plan."""
    iteraciones_teoricas = math.floor((math.pi / 4) * math.sqrt(2**bits))
    iteraciones_maximas = max(1, iteraciones_teoricas * FACTOR_MAX_BUSQUEDA)
    plan = generar_iteraciones_candidatas(iteraciones_maximas)
    umbral_cuentas = math.ceil(SHOTS * UMBRAL_FRECUENCIA_OBJETIVO)

    print(f"Iteraciones teóricas óptimas : {iteraciones_teoricas:,}")
    print(f"k planificadas               : {len(plan)} (hasta k={iteraciones_maximas})")
    print(f"Umbral de éxito              : {umbral_cuentas}/{SHOTS} mediciones")

    mejor = {
        "iteracion": None, "frecuencia_llave": -1,
        "estado_top1": None, "frecuencia_top1": 0,
        "profundidad": 0, "num_compuertas": 0,
        "num_qubits_auxiliares": 0, "ops_por_compuerta": {},
        "conteos": {},
    }
    primer_exito = None
    inicio = time.perf_counter()
    evaluaciones = 0
    razon_corte = "max_k"

    for k in plan:
        if time.perf_counter() - inicio >= MAX_TIEMPO_BUSQUEDA_S:
            razon_corte = "timeout"
            break

        corrida = ejecutar_iteracion_grover(backend, oracle, bits, k)
        evaluaciones += 1
        freq_llave = corrida["conteos"].get(llave_objetivo, 0)

        if freq_llave > mejor["frecuencia_llave"]:
            mejor = {
                "iteracion": k,
                "frecuencia_llave": freq_llave,
                "estado_top1": corrida["estado_top1"],
                "frecuencia_top1": corrida["frecuencia_top1"],
                "profundidad": corrida["profundidad"],
                "num_compuertas": corrida["num_compuertas"],
                "num_qubits_auxiliares": corrida["num_qubits_auxiliares"],
                "ops_por_compuerta": corrida["ops_por_compuerta"],
                "conteos": corrida["conteos"],
            }

        if VERBOSE_ITERACIONES:
            print(
                f"k={k:>3} | top1={corrida['estado_top1']} "
                f"({corrida['frecuencia_top1']}/{SHOTS}) | "
                f"llave={freq_llave}/{SHOTS} | "
                f"depth={corrida['profundidad']} | gates={corrida['num_compuertas']}"
            )

        if (
            corrida["estado_top1"] == llave_objetivo
            and freq_llave >= umbral_cuentas
            and primer_exito is None
        ):
            primer_exito = k

    encontrada = (
        mejor["estado_top1"] == llave_objetivo
        and mejor["frecuencia_llave"] >= umbral_cuentas
    )
    if encontrada and razon_corte != "timeout":
        razon_corte = "exito"

    return {
        "encontrada": encontrada,
        "corte_temprano": razon_corte == "timeout",
        "razon_corte": razon_corte,
        "iteraciones_teoricas": iteraciones_teoricas,
        "iteraciones_maximas": iteraciones_maximas,
        "iteraciones_planificadas": len(plan),
        "evaluaciones_realizadas": evaluaciones,
        "iteraciones_requeridas": mejor["iteracion"],
        "iteracion_primer_exito": primer_exito,
        "umbral_frecuencia_objetivo": UMBRAL_FRECUENCIA_OBJETIVO,
        "estado_top1": mejor["estado_top1"],
        "frecuencia_top1": mejor["frecuencia_top1"],
        "frecuencia_llave": mejor["frecuencia_llave"],
        "tasa_exito": round(mejor["frecuencia_llave"] / SHOTS, 6),
        "distribucion_mediciones": mejor["conteos"],
        "profundidad": mejor["profundidad"],
        "num_compuertas": mejor["num_compuertas"],
        "num_qubits_auxiliares": mejor["num_qubits_auxiliares"],
        "ops_por_compuerta": mejor["ops_por_compuerta"],
        "tiempo_busqueda_s": time.perf_counter() - inicio,
    }


# ─── Punto de entrada ─────────────────────────────────────────────────────────
def main(bits: int = NUM_QUBITS, csv_path: Path = RESULTADOS_CSV) -> None:
    run_id = str(uuid.uuid4())
    inicio_total = time.perf_counter()

    api_token = obtener_api_token()
    provider = QRydProvider(token=api_token)
    backend = provider.get_backend(QRYD_BACKEND_NAME)
    backend_version = getattr(backend, "version", "N/A")

    espacio = 2**bits
    llave_int = random.randint(0, espacio - 1)
    llave_str = format(llave_int, f"0{bits}b")

    print("=" * 60)
    print("  Simulador QRydDemo – Experimento de Grover")
    print("=" * 60)
    print(f"Run ID          : {run_id}")
    print(f"Backend         : {QRYD_BACKEND_NAME} (v{backend_version})")
    print(f"Qubits          : {bits}   →   N = {espacio:,}")
    print(f"Llave objetivo  : {llave_str}  ({llave_int})")
    print(f"Shots           : {SHOTS}")
    print(f"Seed simulador  : {SEED_SIMULADOR}")
    print(f"Seed transpiler : {SEED_TRANSPILER}")
    print("-" * 60)

    oracle = construir_oraculo(bits, llave_str)
    resultado = buscar_iteraciones_necesarias(backend, oracle, bits, llave_str)

    print("\n" + "=" * 60)
    print("  Resultado de búsqueda")
    print("=" * 60)
    print(f"Estado Top-1       : {resultado['estado_top1']}  ({resultado['frecuencia_top1']}/{SHOTS})")
    print(f"Frecuencia llave   : {resultado['frecuencia_llave']}/{SHOTS}")
    print(f"Tasa de éxito      : {resultado['tasa_exito']:.4f}  ({resultado['tasa_exito']*100:.2f}%)")
    print(f"k óptimo           : {resultado['iteraciones_requeridas']}")
    print(f"Primer k ≥ umbral  : {resultado['iteracion_primer_exito']}")
    print(f"Profundidad circ.  : {resultado['profundidad']}")
    print(f"Núm. compuertas    : {resultado['num_compuertas']}")
    print(f"Qubits auxiliares  : {resultado['num_qubits_auxiliares']}")
    print(f"Ops por compuerta  : {resultado['ops_por_compuerta']}")
    print(f"Distribución       : {json.dumps(resultado['distribucion_mediciones'])}")
    print(f"Encontrada         : {resultado['encontrada']}")
    print(f"Razón de corte     : {resultado['razon_corte']}")

    guardar_resultado_csv(
        {
            "timestamp": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            "run_id": run_id,
            "backend": QRYD_BACKEND_NAME,
            "backend_version": backend_version,
            "bits": bits,
            "llave_objetivo_bin": llave_str,
            "llave_objetivo_int": llave_int,
            "seed_simulador": SEED_SIMULADOR,
            "seed_transpiler": SEED_TRANSPILER,
            "iteraciones_teoricas": resultado["iteraciones_teoricas"],
            "iteraciones_maximas_evaluadas": resultado["iteraciones_maximas"],
            "iteraciones_planificadas": resultado["iteraciones_planificadas"],
            "evaluaciones_realizadas": resultado["evaluaciones_realizadas"],
            "corte_temprano": resultado["corte_temprano"],
            "razon_corte": resultado["razon_corte"],
            "iteraciones_requeridas": resultado["iteraciones_requeridas"],
            "iteracion_primer_exito": resultado["iteracion_primer_exito"],
            "shots": SHOTS,
            "umbral_frecuencia_objetivo": resultado["umbral_frecuencia_objetivo"],
            "estado_top1": resultado["estado_top1"],
            "frecuencia_top1": resultado["frecuencia_top1"],
            "frecuencia_llave": resultado["frecuencia_llave"],
            "tasa_exito": resultado["tasa_exito"],
            "encontrada": resultado["encontrada"],
            "distribucion_mediciones": json.dumps(resultado["distribucion_mediciones"]),
            "profundidad_circuito": resultado["profundidad"],
            "num_compuertas": resultado["num_compuertas"],
            "num_qubits_auxiliares": resultado["num_qubits_auxiliares"],
            "ops_por_compuerta": json.dumps(resultado["ops_por_compuerta"]),
            "tiempo_total_s": round(time.perf_counter() - inicio_total, 6),
            "tiempo_busqueda_s": round(float(resultado["tiempo_busqueda_s"]), 6),
        },
        csv_path=csv_path,
    )
    print(f"\nResultados guardados en: {csv_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el algoritmo de Grover en el simulador QRydDemo "
            "y guarda métricas extendidas en CSV."
        )
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=NUM_QUBITS,
        help=f"Número de qubits (default: {NUM_QUBITS}).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=RESULTADOS_CSV,
        help=f"Ruta del CSV de salida (default: {RESULTADOS_CSV}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(bits=args.bits, csv_path=args.csv_out)
