"""Experimento de Grover en HARDWARE CUANTICO REAL de IQM Resonance.

Este modulo es el equivalente en HARDWARE del modulo QRydDemo
(`quantum_search_qryd.py`) y el gemelo directo de `quantum_search_ibm.py`:
implementa EXACTAMENTE el mismo algoritmo de Grover, el mismo barrido de
iteraciones k, las mismas metricas extendidas y las MISMAS columnas de CSV, para
que la comparacion entre modelos (clasico / Aer / QRyd / IBM / IQM) sea logica y
directa.

Por que IQM Resonance
─────────────────────
Es una QPU superconductora (misma familia fisica que IBM), con provider NATIVO de
Qiskit (`iqm-client[qiskit]`) y plan gratuito ("freemium"). Por eso el circuito de
Grover y la transpilacion ISA quedan IDENTICOS a los de IBM: solo cambian la
autenticacion, la seleccion de backend y la primitiva de ejecucion (aqui es el
`backend.run(...)` clasico de Qiskit, igual que QRydDemo).

Correspondencia con el Avance 2 (Q-Search)
──────────────────────────────────────────
- Nivel 2 (Aer + QRydDemo): simulacion con analisis de recursos, n = 2..6.
- Nivel 3 (hardware real): ejecucion REAL, n = 2..4. El objetivo no es demostrar
  escalamiento sino cuantificar la caida de la tasa de exito por decoherencia y
  errores. Por eso se registran las mismas metricas de recursos que QRydDemo
  (profundidad, compuertas, tasa de exito, distribucion de mediciones, etc.).

Diferencias tecnicas frente al simulador (obligatorias en hardware real)
────────────────────────────────────────────────────────────────────────
1. Transpilacion a ISA circuit del backend con generate_preset_pass_manager
   (los circuitos del simulador NO corren tal cual en hardware).
2. Ejecucion con el runner clasico de Qiskit `backend.run(circuitos, shots=...)`.
3. Para minimizar overhead/cola, el barrido de k de una corrida se agrupa en UNA
   SOLA llamada (lista de circuitos). El modo por lotes agrupa ademas todas las
   corridas de un nivel de qubits en una unica llamada. El RESULTADO (mejor k,
   metricas, seleccion) es identico al del barrido secuencial de QRyd/Aer.

Autenticacion (via .env, NO se hardcodea nada)
──────────────────────────────────────────────
    IQM_TOKEN=<API token del dashboard de IQM Resonance>
    IQM_SERVER_URL=https://resonance.iqm.tech   (raiz de Resonance, comun a todos)
    IQM_DEVICE=<alias del dispositivo, p.ej. emerald / garnet / deneb>
El token, la Server URL y el alias del dispositivo (campo "Quantum computer
alias") se obtienen en la ficha del dispositivo en https://resonance.iqm.tech/.
En iqm-client 34 el dispositivo se elige por alias (quantum_computer=), NO por
una URL especifica de dispositivo.

Uso
───
    python quantum_search_iqm.py --dry-run    # estima, NO envia job
    python quantum_search_iqm.py              # lote 25 corridas x [2,4] (confirma)
    python quantum_search_iqm.py --bits 2 --una-corrida   # 1 corrida (interfaz Aer)

Requisitos:
    pip install "iqm-client[qiskit]" qiskit python-dotenv
"""

import argparse
import csv
import json
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from iqm.qiskit_iqm import IQMProvider

# ─── Parametros del algoritmo (IDENTICOS a Aer/QRyd/IBM para comparacion justa) ─
NUM_QUBITS = 2                      # nivel por defecto en hardware real
SHOTS = 512                        # shots por circuito. 1024 igualaria a Aer/QRyd
                                   # exacto, pero consume el doble de QPU.
SEED_TRANSPILER = 42
FACTOR_MAX_BUSQUEDA = 2            # rango de k explorado = 2 * k_teorico
UMBRAL_FRECUENCIA_OBJETIVO = 0.9  # umbral de frecuencia de la llave para "exito"
MAX_EVALUACIONES_K = 12           # submuestreo de k (igual que QRyd)
VERBOSE_ITERACIONES = False

# ─── Presupuesto de QPU: objetivo del Nivel 3 (hardware real) ─────────────────
QUBITS_OBJETIVO_HW = [2, 4]        # Avance 2: ejecucion real solo para n = 2 y 4
CORRIDAS_POR_QUBIT_HW = 25        # 25 corridas por nivel, igual que los demas

# ─── Runtime / transpilacion ──────────────────────────────────────────────────
OPTIMIZATION_LEVEL = 1            # nivel del pass manager (ISA obligatorio)
ESTIM_SEG_OVERHEAD_POR_JOB = 3.0  # estimacion GRUESA de consumo (solo --dry-run)
ESTIM_SEG_POR_CIRCUITO = 1.0

# Server URL raiz de IQM Resonance y alias de dispositivo por defecto
# (ambos se sobreescriben con IQM_SERVER_URL / IQM_DEVICE del .env).
IQM_SERVER_URL_DEFAULT = "https://resonance.iqm.tech"
IQM_DEVICE_DEFAULT = "emerald"

# Limite de circuitos por job del dispositivo (Emerald: "Max. circuits" = 100).
# El modo lote trocea el envio en bloques de este tamano.
MAX_CIRCUITOS_POR_JOB = 100

RESULTADOS_DIR = Path(__file__).resolve().parent / "results"


def _csv_iqm(bits: int) -> Path:
    """Ruta del CSV por nivel de qubits (una familia de archivos por bits)."""
    return RESULTADOS_DIR / f"results_iqm_{bits}q.csv"


RESULTADOS_CSV = _csv_iqm(NUM_QUBITS)

load_dotenv()


# ─── CSV: MISMAS columnas que quantum_search_qryd.py / quantum_search_ibm.py ───
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
    # plan de busqueda
    "iteraciones_teoricas",
    "iteraciones_maximas_evaluadas",
    "iteraciones_planificadas",
    "evaluaciones_realizadas",
    "corte_temprano",
    "razon_corte",
    "iteraciones_requeridas",
    "iteracion_primer_exito",
    # metricas de medicion
    "shots",
    "umbral_frecuencia_objetivo",
    "estado_top1",
    "frecuencia_top1",
    "frecuencia_llave",
    "tasa_exito",
    "encontrada",
    "distribucion_mediciones",
    # metricas del circuito transpilado (ISA)
    "profundidad_circuito",
    "num_compuertas",
    "num_qubits_auxiliares",
    "ops_por_compuerta",
    # tiempos
    "tiempo_total_s",
    "tiempo_busqueda_s",
]


def _preparar_csv(csv_path: Path) -> bool:
    """Respalda el CSV previo si el encabezado cambio; devuelve si escribir header."""
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
    """Guarda una corrida (una fila) con las mismas columnas que QRyd/IBM."""
    csv_path.parent.mkdir(exist_ok=True)
    escribir_encabezado = _preparar_csv(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ENCABEZADO_CSV)
        if escribir_encabezado:
            writer.writeheader()
        writer.writerow(fila)


# ─── Construccion del circuito (IDENTICA a Aer/QRyd/IBM) ──────────────────────
def construir_oraculo(bits: int, llave: str) -> QuantumCircuit:
    """Oraculo de fase que marca un unico estado objetivo sin qubits ancilla.

    1. X en cada qubit cuyo bit en `llave` es '0'  → objetivo pasa a |11..1>.
    2. MCPhase(pi) con n-1 controles            → |11..1> adquiere fase -1.
    3. X inverso para restaurar la base computacional.
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


def generar_iteraciones_candidatas(iteraciones_maximas: int) -> list:
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


def construir_circuito_grover(bits: int, llave: str, k: int) -> QuantumCircuit:
    """Circuito de Grover con k iteraciones y medicion (pre-transpile)."""
    oracle = construir_oraculo(bits, llave)
    qc = QuantumCircuit(bits, bits)
    qc.h(range(bits))
    qc.compose(grover_operator(oracle).power(k), inplace=True)
    qc.measure(range(bits), range(bits))
    return qc


def _plan_de_busqueda(bits: int):
    """Devuelve (iteraciones_teoricas, iteraciones_maximas, plan de k)."""
    iteraciones_teoricas = math.floor((math.pi / 4) * math.sqrt(2 ** bits))
    iteraciones_maximas = max(1, iteraciones_teoricas * FACTOR_MAX_BUSQUEDA)
    plan = generar_iteraciones_candidatas(iteraciones_maximas)
    return iteraciones_teoricas, iteraciones_maximas, plan


# ─── Autenticacion y backend ──────────────────────────────────────────────────
def crear_provider() -> IQMProvider:
    """Crea el provider de IQM Resonance (raiz + alias de dispositivo).

    En iqm-client 34 la Server URL es la raiz comun (https://resonance.iqm.tech)
    y el dispositivo se selecciona por alias con quantum_computer=. El token NO se
    pasa como argumento: iqm-client lo lee de la variable de entorno IQM_TOKEN
    (que load_dotenv() ya cargo desde .env). Mezclar ambas fuentes (arg + env) es
    un error explicito en iqm-client, por eso solo usamos la variable de entorno.
    """
    url = os.environ.get("IQM_SERVER_URL", "").strip() or IQM_SERVER_URL_DEFAULT
    device = os.environ.get("IQM_DEVICE", "").strip() or IQM_DEVICE_DEFAULT

    try:
        return IQMProvider(url, quantum_computer=device)
    except Exception as exc:  # noqa: BLE001
        raise EnvironmentError(
            "No se pudo crear el provider de IQM Resonance.\n"
            "Define en un archivo .env (en la raiz del proyecto):\n"
            "    IQM_TOKEN=<tu API token del dashboard de IQM Resonance>\n"
            "    IQM_SERVER_URL=https://resonance.iqm.tech\n"
            "    IQM_DEVICE=<alias del dispositivo, p.ej. emerald>\n"
            f"Detalle: {exc}"
        ) from exc


def mostrar_uso_qpu(provider: IQMProvider) -> None:
    """Imprime informacion del dispositivo seleccionado (IQM no expone usage())."""
    try:
        backend = provider.get_backend()
        print(f"[INFO] Dispositivo IQM: {backend.name} "
              f"({backend.num_qubits} qubits)")
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] No se pudo consultar el dispositivo IQM: {exc}")


def seleccionar_backend(provider: IQMProvider, min_qubits: int):
    """Selecciona la QPU de IQM Resonance apuntada por IQM_SERVER_URL.

    En Resonance la URL ya identifica el dispositivo, asi que basta con
    get_backend() y verificar que tenga suficientes qubits. Devuelve None si el
    dispositivo no tiene >= min_qubits qubits o si no se puede obtener.
    """
    try:
        backend = provider.get_backend()
    except Exception as exc:  # noqa: BLE001
        print(f"[AVISO] No se pudo obtener el backend de IQM: {exc}")
        return None
    if backend.num_qubits < min_qubits:
        print(f"[AVISO] El dispositivo {backend.name} tiene {backend.num_qubits} "
              f"qubits (< {min_qubits} requeridos). Se salta este nivel.")
        return None
    return backend


# ─── Preparacion y seleccion (misma logica de seleccion que QRyd/IBM) ─────────
def _preparar_circuitos_corrida(pass_manager, bits: int, llave: str):
    """Transpila a ISA el barrido de k de UNA corrida.

    Devuelve (plan, lista_isa, lista_metricas, creg_name) en el orden de `plan`.
    """
    _, _, plan = _plan_de_busqueda(bits)
    logicos = [construir_circuito_grover(bits, llave, k) for k in plan]
    creg_name = logicos[0].cregs[0].name  # normalmente 'c'
    isa = pass_manager.run(logicos)
    metricas = [extraer_metricas_circuito(c) for c in isa]
    return plan, isa, metricas, creg_name


def _seleccionar_mejor(bits: int, llave: str, plan: list,
                       resultados_por_k: list, tiempo_busqueda_s: float) -> dict:
    """Elige el mejor k por frecuencia de la llave (MISMA logica que QRyd/IBM).

    `resultados_por_k` es una lista alineada con `plan`, cada elemento:
      {conteos, profundidad, num_compuertas, num_qubits_auxiliares,
       ops_por_compuerta}
    """
    iteraciones_teoricas, iteraciones_maximas, _ = _plan_de_busqueda(bits)
    # Umbral en cuentas basado en los shots realmente medidos.
    total_shots = sum(resultados_por_k[0]["conteos"].values()) if resultados_por_k else SHOTS
    umbral_cuentas = math.ceil((total_shots or SHOTS) * UMBRAL_FRECUENCIA_OBJETIVO)

    mejor = {
        "iteracion": None, "frecuencia_llave": -1,
        "estado_top1": None, "frecuencia_top1": 0,
        "profundidad": 0, "num_compuertas": 0,
        "num_qubits_auxiliares": 0, "ops_por_compuerta": {},
        "conteos": {},
    }
    primer_exito = None
    evaluaciones = 0

    for k, corrida in zip(plan, resultados_por_k):
        evaluaciones += 1
        conteos = corrida["conteos"]
        estado_top1, freq_top1 = max(conteos.items(), key=lambda x: x[1])
        freq_llave = conteos.get(llave, 0)

        if freq_llave > mejor["frecuencia_llave"]:
            mejor = {
                "iteracion": k,
                "frecuencia_llave": freq_llave,
                "estado_top1": estado_top1,
                "frecuencia_top1": freq_top1,
                "profundidad": corrida["profundidad"],
                "num_compuertas": corrida["num_compuertas"],
                "num_qubits_auxiliares": corrida["num_qubits_auxiliares"],
                "ops_por_compuerta": corrida["ops_por_compuerta"],
                "conteos": conteos,
            }

        if VERBOSE_ITERACIONES:
            print(f"    k={k:>3} | top1={estado_top1} ({freq_top1}) | "
                  f"llave={freq_llave} | depth={corrida['profundidad']}")

        if estado_top1 == llave and freq_llave >= umbral_cuentas and primer_exito is None:
            primer_exito = k

    encontrada = (
        mejor["estado_top1"] == llave and mejor["frecuencia_llave"] >= umbral_cuentas
    )
    razon_corte = "exito" if encontrada else "max_k"

    return {
        "encontrada": encontrada,
        "corte_temprano": False,   # en hardware no hay timeout de barrido
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
        "tasa_exito": round(mejor["frecuencia_llave"] / (total_shots or SHOTS), 6),
        "distribucion_mediciones": mejor["conteos"],
        "profundidad": mejor["profundidad"],
        "num_compuertas": mejor["num_compuertas"],
        "num_qubits_auxiliares": mejor["num_qubits_auxiliares"],
        "ops_por_compuerta": mejor["ops_por_compuerta"],
        "tiempo_busqueda_s": tiempo_busqueda_s,
    }


def _counts_desde_resultado(result, idx: int) -> dict:
    """Conteos (dict bitstring->cuentas) del circuito idx de backend.run()."""
    return result.get_counts(idx)


def fila_csv(resultado: dict, *, run_id: str, backend_name: str, backend_version,
             bits: int, llave_str: str, llave_int: int, shots: int,
             tiempo_total_s: float) -> dict:
    """Ensambla la fila del CSV con las columnas exactas de QRyd/IBM."""
    return {
        "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_id": run_id,
        "backend": backend_name,
        "backend_version": backend_version,
        "bits": bits,
        "llave_objetivo_bin": llave_str,
        "llave_objetivo_int": llave_int,
        "seed_simulador": "",   # el hardware real no usa semilla de simulacion
        "seed_transpiler": SEED_TRANSPILER,
        "iteraciones_teoricas": resultado["iteraciones_teoricas"],
        "iteraciones_maximas_evaluadas": resultado["iteraciones_maximas"],
        "iteraciones_planificadas": resultado["iteraciones_planificadas"],
        "evaluaciones_realizadas": resultado["evaluaciones_realizadas"],
        "corte_temprano": resultado["corte_temprano"],
        "razon_corte": resultado["razon_corte"],
        "iteraciones_requeridas": resultado["iteraciones_requeridas"],
        "iteracion_primer_exito": resultado["iteracion_primer_exito"],
        "shots": shots,
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
        "tiempo_total_s": round(tiempo_total_s, 6),
        "tiempo_busqueda_s": round(float(resultado["tiempo_busqueda_s"]), 6),
    }


def estimar_consumo(n_jobs: int, n_circuitos: int) -> float:
    """Estimacion GRUESA (orden de magnitud) del tiempo de QPU en segundos."""
    return n_jobs * ESTIM_SEG_OVERHEAD_POR_JOB + n_circuitos * ESTIM_SEG_POR_CIRCUITO


# ─── Interfaz compatible con Aer/QRyd/IBM: main(bits, csv_path) ───────────────
def main(bits: int = NUM_QUBITS, csv_path: Path = None, shots: int = SHOTS,
         auto_si: bool = True) -> None:
    """Ejecuta UNA corrida (barrido de k completo) de un nivel en HW real.

    Interfaz identica a quantum_search_qryd.main(bits, csv_path). El barrido de k
    de esta corrida se agrupa en una sola llamada a backend.run para minimizar cola.
    """
    if csv_path is None:
        csv_path = _csv_iqm(bits)

    provider = crear_provider()
    backend = seleccionar_backend(provider, min_qubits=bits)
    if backend is None:
        print(f"[AVISO] Sin QPU de IQM con >= {bits} qubits. Se salta este nivel.")
        return

    run_id = str(uuid.uuid4())
    inicio_total = time.perf_counter()
    llave_int = random.randint(0, 2 ** bits - 1)
    llave_str = format(llave_int, f"0{bits}b")
    backend_version = getattr(backend, "version", "N/A")

    print("=" * 60)
    print("  IQM Resonance – Grover en HARDWARE REAL (una corrida)")
    print("=" * 60)
    print(f"Run ID          : {run_id}")
    print(f"Backend         : {backend.name} (v{backend_version})")
    print(f"Qubits          : {bits}   →   N = {2 ** bits:,}")
    print(f"Llave objetivo  : {llave_str}  ({llave_int})")
    print(f"Shots           : {shots}")

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
    )
    plan, isa, metricas, _ = _preparar_circuitos_corrida(pm, bits, llave_str)
    print(f"k planificadas  : {len(plan)}  (hasta k={max(plan)})")

    if not auto_si:
        print("\nESTO CONSUMIRA TIEMPO REAL DE QPU.")
        if input("Escribe 'si' para continuar: ").strip().lower() not in (
            "si", "sí", "s", "yes", "y"
        ):
            print("Cancelado. Nada consumido.")
            return

    inicio_job = time.perf_counter()
    job = backend.run(isa, shots=shots)
    print(f"Job ID          : {job.job_id()}  |  estado: {job.status()}")
    print("Esperando resultados (puede tardar por la cola)...")
    result = job.result()
    tiempo_busqueda = time.perf_counter() - inicio_job

    resultados_por_k = []
    for idx, met in enumerate(metricas):
        resultados_por_k.append({"conteos": _counts_desde_resultado(result, idx), **met})

    resultado = _seleccionar_mejor(bits, llave_str, plan, resultados_por_k, tiempo_busqueda)

    print("\n--- Resultado ---")
    print(f"Top-1           : {resultado['estado_top1']} ({resultado['frecuencia_top1']}/{shots})")
    print(f"Frecuencia llave: {resultado['frecuencia_llave']}/{shots}")
    print(f"Tasa de exito   : {resultado['tasa_exito']*100:.2f}%")
    print(f"k optimo (emp.) : {resultado['iteraciones_requeridas']}")
    print(f"Encontrada      : {resultado['encontrada']}")

    guardar_resultado_csv(
        fila_csv(
            resultado, run_id=run_id, backend_name=backend.name,
            backend_version=backend_version, bits=bits, llave_str=llave_str,
            llave_int=llave_int, shots=shots,
            tiempo_total_s=time.perf_counter() - inicio_total,
        ),
        csv_path=csv_path,
    )
    print(f"Resultado guardado en: {csv_path}")


# ─── Modo por LOTES: una llamada por nivel de qubits (25 corridas juntas) ──────
def ejecutar_lote(dry_run: bool = False, auto_si: bool = False,
                  qubits=None, corridas: int = None, shots: int = None) -> None:
    """Ejecuta CORRIDAS_POR_QUBIT_HW corridas para cada nivel de QUBITS_OBJETIVO_HW.

    Para cada nivel de qubits agrupa TODAS las corridas (cada una con su barrido
    de k) en UNA SOLA llamada a backend.run. Esto minimiza cola y produce
    exactamente las mismas 25 filas que produciria el barrido secuencial.

    Con --dry-run NO se envia nada: transpila, cuenta circuitos y estima consumo.
    """
    qubits = qubits or QUBITS_OBJETIVO_HW
    corridas = corridas or CORRIDAS_POR_QUBIT_HW
    shots = shots or SHOTS

    print("=" * 70)
    print("  IQM Resonance – Grover en HARDWARE REAL (lote por niveles)")
    print("=" * 70)
    print(f"Niveles de qubits : {qubits}")
    print(f"Corridas/qubit    : {corridas}")
    print(f"Shots/circuito    : {shots}")

    provider = crear_provider()
    mostrar_uso_qpu(provider)

    for bits in qubits:
        backend = seleccionar_backend(provider, min_qubits=bits)
        if backend is None:
            print(f"[AVISO] Sin QPU de IQM con >= {bits} qubits. Se salta {bits}q.")
            continue
        backend_version = getattr(backend, "version", "N/A")

        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=OPTIMIZATION_LEVEL,
            seed_transpiler=SEED_TRANSPILER,
        )

        # Preparar TODAS las corridas de este nivel; aplanar sus circuitos de k.
        print("\n" + "-" * 70)
        print(f"Nivel {bits} qubits  (N = {2 ** bits:,})  |  backend: {backend.name}")
        print("Transpilando circuitos a ISA...")

        todos_isa = []
        corridas_meta = []  # una entrada por corrida
        for _ in range(corridas):
            run_id = str(uuid.uuid4())
            llave_int = random.randint(0, 2 ** bits - 1)
            llave_str = format(llave_int, f"0{bits}b")
            plan, isa, metricas, creg = _preparar_circuitos_corrida(pm, bits, llave_str)
            base = len(todos_isa)
            todos_isa.extend(isa)
            corridas_meta.append({
                "run_id": run_id, "llave_int": llave_int, "llave_str": llave_str,
                "plan": plan, "metricas": metricas, "creg": creg,
                "indices": list(range(base, base + len(isa))),
            })

        n_circuitos = len(todos_isa)
        n_jobs = math.ceil(n_circuitos / MAX_CIRCUITOS_POR_JOB)
        print(f"Corridas          : {corridas}")
        print(f"Circuitos totales : {n_circuitos}  (en {n_jobs} job(s) de "
              f"<= {MAX_CIRCUITOS_POR_JOB} circuitos)")
        print(f"Shots totales     : {n_circuitos * shots:,}")
        print(f"Estimacion QPU    : ~{estimar_consumo(n_jobs, n_circuitos):.0f} s "
              f"(MUY aproximada; el valor real depende del dispositivo IQM)")

        if dry_run:
            print(f"[DRY-RUN] {bits}q: no se envio job. Nada consumido.")
            continue

        if not auto_si:
            print(f"\nESTO CONSUMIRA TIEMPO REAL DE QPU (nivel {bits}q).")
            if input("Escribe 'si' para enviar el job: ").strip().lower() not in (
                "si", "sí", "s", "yes", "y"
            ):
                print(f"Cancelado el nivel {bits}q. Nada consumido.")
                continue

        # Envio troceado en bloques de <= MAX_CIRCUITOS_POR_JOB (limite del device).
        # Se reconstruyen los conteos por indice global para no romper la
        # correspondencia con corridas_meta (que indexa sobre todos_isa).
        conteos_por_indice = [None] * n_circuitos
        inicio_job = time.perf_counter()
        for chunk_i, base in enumerate(range(0, n_circuitos, MAX_CIRCUITOS_POR_JOB), start=1):
            chunk = todos_isa[base:base + MAX_CIRCUITOS_POR_JOB]
            job = backend.run(chunk, shots=shots)
            print(f"Job {chunk_i}/{n_jobs}       : {job.job_id()}  |  "
                  f"{len(chunk)} circuitos  |  estado: {job.status()}")
            print("Esperando resultados (puede tardar por la cola)...")
            result = job.result()
            for local in range(len(chunk)):
                conteos_por_indice[base + local] = _counts_desde_resultado(result, local)
        tiempo_job = time.perf_counter() - inicio_job
        # Atribucion aproximada del tiempo a cada corrida.
        tiempo_por_corrida = tiempo_job / max(1, corridas)
        print(f"Resultados recibidos en {tiempo_job:.1f}s (incluye cola).")

        encontradas = 0
        for c in corridas_meta:
            resultados_por_k = []
            for idx_local, met in zip(c["indices"], c["metricas"]):
                resultados_por_k.append(
                    {"conteos": conteos_por_indice[idx_local], **met}
                )
            resultado = _seleccionar_mejor(
                bits, c["llave_str"], c["plan"], resultados_por_k, tiempo_por_corrida
            )
            guardar_resultado_csv(
                fila_csv(
                    resultado, run_id=c["run_id"], backend_name=backend.name,
                    backend_version=backend_version, bits=bits,
                    llave_str=c["llave_str"], llave_int=c["llave_int"],
                    shots=shots, tiempo_total_s=tiempo_por_corrida,
                ),
                csv_path=_csv_iqm(bits),
            )
            encontradas += int(resultado["encontrada"])

        print(f"Nivel {bits}q listo: {encontradas}/{corridas} encontradas "
              f"→ {_csv_iqm(bits).name}")

    print("\nDispositivo IQM tras el lote:")
    mostrar_uso_qpu(provider)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grover en hardware real de IQM Resonance (backend.run). Mismo "
            "algoritmo y metricas que QRydDemo/IBM. Por defecto corre el lote de "
            "25 corridas."
        )
    )
    parser.add_argument(
        "--dry-run", "--estimar", dest="dry_run", action="store_true",
        help="Transpila y estima circuitos/shots/consumo SIN enviar job.",
    )
    parser.add_argument(
        "--si", dest="auto_si", action="store_true",
        help="No pedir confirmacion interactiva antes de gastar QPU.",
    )
    parser.add_argument(
        "--una-corrida", dest="una_corrida", action="store_true",
        help="Ejecuta una sola corrida de --bits (interfaz compatible con QRyd).",
    )
    parser.add_argument(
        "--bits", type=int, default=NUM_QUBITS,
        help=f"Qubits para --una-corrida (default: {NUM_QUBITS}).",
    )
    parser.add_argument(
        "--qubits", type=int, nargs="+", default=None,
        help=(
            "Niveles de qubits del lote (default: "
            f"{QUBITS_OBJETIVO_HW}). Ej: --qubits 4 para correr solo n=4."
        ),
    )
    parser.add_argument(
        "--shots", type=int, default=SHOTS,
        help=f"Shots por circuito (default: {SHOTS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.una_corrida:
        main(bits=args.bits, shots=args.shots, auto_si=args.auto_si)
    else:
        ejecutar_lote(dry_run=args.dry_run, auto_si=args.auto_si,
                      qubits=args.qubits, shots=args.shots)
