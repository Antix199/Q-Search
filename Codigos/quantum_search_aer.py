"""Experimento de Grover simulado con Qiskit Aer.

Objetivo metodologico:
- generar una llave objetivo aleatoria,
- evaluar iteraciones candidatas de Grover (k),
- declarar hallazgo cuando Top-1 coincide y supera un umbral de frecuencia,
- registrar tiempos/iteraciones para comparacion con el baseline clasico.

Nota: este script mide rendimiento de simulacion clasica de un circuito
cuantico, no tiempo de ejecucion en hardware cuantico real.
"""

import csv
import math
import random
import time
import uuid
import argparse
from datetime import datetime, timezone
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit_aer import AerSimulator

# Tamano del problema: espacio de busqueda = 2**NUM_QUBITS.
NUM_QUBITS = 10
# Numero de mediciones por circuito (mas shots = mejor estadistica, mas tiempo).
SHOTS = 1024
# Semilla del simulador para reproducibilidad de resultados.
SEED_SIMULADOR = 42
# Semilla del transpiler para reproducibilidad en la optimizacion del circuito.
SEED_TRANSPILER = 42
# Multiplicador del k teorico para definir el rango maximo de exploracion.
FACTOR_MAX_BUSQUEDA = 2
# Umbral minimo de frecuencia de la llave para declarar hallazgo robusto.
UMBRAL_FRECUENCIA_OBJETIVO = 0.9
# Limite de valores k realmente evaluados (submuestreo del rango total).
MAX_EVALUACIONES_K = 24
# Tiempo maximo permitido para la busqueda de k antes de cortar por timeout.
MAX_TIEMPO_BUSQUEDA_S = 90
# Traza detallada por iteracion k para diagnostico experimental.
VERBOSE_ITERACIONES = False

RESULTADOS_DIR = Path(__file__).resolve().parent / "results"
RESULTADOS_CSV = RESULTADOS_DIR / "results_quantum.csv"


def _preparar_csv(encabezado, csv_path):
    """Asegura compatibilidad de esquema al mantener historico de resultados."""
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


def seleccionar_simulador_aer():
    """Selecciona Aer en GPU si esta disponible; si no, usa CPU."""
    # Intento GPU primero; si no hay build CUDA o falla, cae en CPU.
    try:
        sim_gpu = AerSimulator()
        sim_gpu.set_options(method="statevector", device="GPU")
        prueba = QuantumCircuit(1)
        prueba.h(0)
        prueba.measure_all()
        sim_gpu.run(prueba, shots=8).result()
        return sim_gpu, "GPU"
    except Exception:
        sim_cpu = AerSimulator()
        sim_cpu.set_options(method="automatic", device="CPU")
        return sim_cpu, "CPU"


def guardar_resultado_csv(fila, csv_path=RESULTADOS_CSV):
    """Guarda una corrida cuantica con metadatos de corte y convergencia."""
    csv_path.parent.mkdir(exist_ok=True)
    encabezado = [
        "timestamp",
        "run_id",
        "simulador",
        "device",
        "bits",
        "llave_objetivo_bin",
        "llave_objetivo_int",
        "iteraciones_teoricas",
        "iteraciones_maximas_evaluadas",
        "iteraciones_planificadas",
        "evaluaciones_realizadas",
        "corte_temprano",
        "razon_corte",
        "iteraciones_requeridas",
        "iteracion_primer_exito",
        "umbral_frecuencia_objetivo",
        "shots",
        "estado_top1",
        "frecuencia_top1",
        "frecuencia_llave",
        "encontrada",
        "tiempo_total_s",
        "tiempo_busqueda_s",
        "profundidad",
    ]
    escribir_encabezado = _preparar_csv(encabezado, csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=encabezado)
        if escribir_encabezado:
            writer.writeheader()
        writer.writerow(fila)


def generar_iteraciones_candidatas(iteraciones_maximas):
    """Reduce costo: submuestrea k cuando el barrido completo es inviable."""
    if iteraciones_maximas <= 0:
        return [1]

    if iteraciones_maximas <= MAX_EVALUACIONES_K:
        return list(range(1, iteraciones_maximas + 1))

    candidatos = set()
    for idx in range(MAX_EVALUACIONES_K):
        # Muestreo uniforme para evitar evaluar miles de k en 20 qubits.
        k = 1 + (idx * (iteraciones_maximas - 1)) // (MAX_EVALUACIONES_K - 1)
        candidatos.add(k)

    candidatos.add(iteraciones_maximas)
    return sorted(candidatos)


def construir_oraculo(bits, llave):
    """Construye un oraculo de fase que marca un unico estado objetivo."""
    oracle = QuantumCircuit(bits)
    for qubit in range(bits):
        if llave[bits - 1 - qubit] == "0":
            oracle.x(qubit)
    oracle.mcp(math.pi, list(range(bits - 1)), bits - 1)
    for qubit in range(bits):
        if llave[bits - 1 - qubit] == "0":
            oracle.x(qubit)
    return oracle


def ejecutar_iteracion_grover(simulador, oracle, bits, iteracion):
    """Ejecuta una k fija de Grover y devuelve conteos y profundidad."""
    qc = QuantumCircuit(bits, bits)
    qc.h(range(bits))
    qc.compose(grover_operator(oracle).power(iteracion), inplace=True)
    qc.measure(range(bits), range(bits))

    circuito_optimizado = transpile(
        qc,
        backend=simulador,
        optimization_level=3,
        seed_transpiler=SEED_TRANSPILER,
    )
    resultado = simulador.run(
        circuito_optimizado,
        shots=SHOTS,
        seed_simulator=SEED_SIMULADOR,
    ).result()
    conteos = resultado.get_counts(circuito_optimizado)
    estado_top1, frecuencia_top1 = max(conteos.items(), key=lambda item: item[1])

    return {
        "estado_top1": estado_top1,
        "frecuencia_top1": frecuencia_top1,
        "conteos": conteos,
        "profundidad": circuito_optimizado.depth(),
    }


def buscar_iteraciones_necesarias(simulador, oracle, bits, llave_objetivo):
    """Explora k candidatas hasta exito, timeout o fin del plan de busqueda."""
    iteraciones_teoricas = math.floor((math.pi / 4) * math.sqrt(2**bits))
    iteraciones_maximas = max(1, iteraciones_teoricas * FACTOR_MAX_BUSQUEDA)
    iteraciones_planificadas = generar_iteraciones_candidatas(iteraciones_maximas)

    print(f"Iteraciones teoricas requeridas: {iteraciones_teoricas:,}")
    print(f"Iteraciones maximas a evaluar: {iteraciones_maximas:,}")
    print(f"k planificadas: {len(iteraciones_planificadas)}")
    print(f"Tiempo limite de busqueda: {MAX_TIEMPO_BUSQUEDA_S}s")
    umbral_cuentas = math.ceil(SHOTS * UMBRAL_FRECUENCIA_OBJETIVO)
    print(f"Umbral de hallazgo: {umbral_cuentas}/{SHOTS} mediciones de la llave")

    mejor = {
        "iteracion": None,
        "frecuencia_llave": -1,
        "estado_top1": None,
        "frecuencia_top1": 0,
        "profundidad": 0,
    }
    primer_exito = None

    inicio_busqueda = time.perf_counter()
    evaluaciones_realizadas = 0
    razon_corte = "max_k"

    for iteracion in iteraciones_planificadas:
        # Limite duro para impedir corridas excesivamente largas.
        if time.perf_counter() - inicio_busqueda >= MAX_TIEMPO_BUSQUEDA_S:
            razon_corte = "timeout"
            break

        corrida = ejecutar_iteracion_grover(simulador, oracle, bits, iteracion)
        evaluaciones_realizadas += 1
        frecuencia_llave = corrida["conteos"].get(llave_objetivo, 0)

        if frecuencia_llave > mejor["frecuencia_llave"]:
            mejor = {
                "iteracion": iteracion,
                "frecuencia_llave": frecuencia_llave,
                "estado_top1": corrida["estado_top1"],
                "frecuencia_top1": corrida["frecuencia_top1"],
                "profundidad": corrida["profundidad"],
            }

        if VERBOSE_ITERACIONES:
            print(
                f"k={iteracion:>3} | top1={corrida['estado_top1']} "
                f"({corrida['frecuencia_top1']}/{SHOTS}) | llave={frecuencia_llave}/{SHOTS}"
            )

        # Se marca el primer exito, pero no se corta para poder hallar el k optimo.
        if (
            corrida["estado_top1"] == llave_objetivo
            and frecuencia_llave >= umbral_cuentas
            and primer_exito is None
        ):
            primer_exito = iteracion

    encontrada = (
        mejor["estado_top1"] == llave_objetivo
        and mejor["frecuencia_llave"] >= umbral_cuentas
    )
    razon_final = razon_corte
    if encontrada and razon_corte != "timeout":
        razon_final = "exito"

    return {
        "encontrada": encontrada,
        "corte_temprano": razon_corte == "timeout",
        "razon_corte": razon_final,
        "iteraciones_teoricas": iteraciones_teoricas,
        "iteraciones_maximas": iteraciones_maximas,
        "iteraciones_planificadas": len(iteraciones_planificadas),
        "evaluaciones_realizadas": evaluaciones_realizadas,
        "iteraciones_requeridas": mejor["iteracion"],
        "iteracion_primer_exito": primer_exito,
        "umbral_frecuencia_objetivo": UMBRAL_FRECUENCIA_OBJETIVO,
        "estado_top1": mejor["estado_top1"],
        "frecuencia_top1": mejor["frecuencia_top1"],
        "frecuencia_llave": mejor["frecuencia_llave"],
        "profundidad": mejor["profundidad"],
        "tiempo_busqueda_s": time.perf_counter() - inicio_busqueda,
    }


def main(bits=NUM_QUBITS, csv_path=RESULTADOS_CSV):
    """Punto de entrada experimental: ejecuta corrida y persiste resultados."""
    run_id = str(uuid.uuid4())
    inicio = time.perf_counter()
    espacio_busqueda = 2**bits

    llave_int = random.randint(0, espacio_busqueda - 1)
    llave_str = format(llave_int, f"0{bits}b")

    print("--- Simulador Aer Local ---")
    print(f"Run ID: {run_id}")
    print(f"Llave objetivo generada: {llave_str} ({llave_int})")

    oracle = construir_oraculo(bits, llave_str)

    simulador, device = seleccionar_simulador_aer()
    print(f"Simulador seleccionado: {simulador.name} ({device})")
    busqueda = buscar_iteraciones_necesarias(
        simulador=simulador,
        oracle=oracle,
        bits=bits,
        llave_objetivo=llave_str,
    )

    print("\n--- Resultado de busqueda cuantica ---")
    print(
        f"Mejor estado Top-1: {busqueda['estado_top1']} "
        f"({busqueda['frecuencia_top1']}/{SHOTS})"
    )
    print(f"Frecuencia de la llave objetivo: {busqueda['frecuencia_llave']}/{SHOTS}")
    print(f"Iteraciones (k) de maxima frecuencia: {busqueda['iteraciones_requeridas']}")
    print(f"Primer k que supera umbral: {busqueda['iteracion_primer_exito']}")
    print(
        f"Evaluaciones realizadas: {busqueda['evaluaciones_realizadas']}/"
        f"{busqueda['iteraciones_planificadas']}"
    )
    if busqueda["encontrada"]:
        print("[VERIFICADO] La llave objetivo fue encontrada como Top-1.")
    elif busqueda["razon_corte"] == "timeout":
        print("[INFO] Busqueda detenida por limite de tiempo.")
    else:
        print("[INFO] No se encontro como Top-1 dentro del rango evaluado.")

    guardar_resultado_csv(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "run_id": run_id,
            "simulador": simulador.name,
            "device": device,
            "bits": bits,
            "llave_objetivo_bin": llave_str,
            "llave_objetivo_int": llave_int,
            "iteraciones_teoricas": busqueda["iteraciones_teoricas"],
            "iteraciones_maximas_evaluadas": busqueda["iteraciones_maximas"],
            "iteraciones_planificadas": busqueda["iteraciones_planificadas"],
            "evaluaciones_realizadas": busqueda["evaluaciones_realizadas"],
            "corte_temprano": busqueda["corte_temprano"],
            "razon_corte": busqueda["razon_corte"],
            "iteraciones_requeridas": busqueda["iteraciones_requeridas"],
            "iteracion_primer_exito": busqueda["iteracion_primer_exito"],
            "umbral_frecuencia_objetivo": busqueda["umbral_frecuencia_objetivo"],
            "shots": SHOTS,
            "estado_top1": busqueda["estado_top1"],
            "frecuencia_top1": busqueda["frecuencia_top1"],
            "frecuencia_llave": busqueda["frecuencia_llave"],
            "encontrada": busqueda["encontrada"],
            "tiempo_total_s": round(time.perf_counter() - inicio, 6),
            "tiempo_busqueda_s": round(float(busqueda["tiempo_busqueda_s"]), 6),
            "profundidad": busqueda["profundidad"],
        },
        csv_path=csv_path,
    )
    print(f"Resultados guardados en: {csv_path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta el experimento de Grover en Aer y guarda resultados en CSV.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=NUM_QUBITS,
        help=f"Cantidad de qubits a usar (default: {NUM_QUBITS}).",
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
    main(bits=args.bits, csv_path=args.csv_out)
