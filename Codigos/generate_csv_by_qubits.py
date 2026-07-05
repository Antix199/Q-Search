"""Genera CSV separados por cantidad de qubits para los tres modelos de búsqueda.

Ejecuta corridas de búsqueda clásica, cuántica (Aer) y opcionalmente QRydDemo
para cada nivel de qubits definido en QUBITS_OBJETIVO, guardando los resultados
en archivos CSV separados dentro del directorio results/.

Uso:
    python generate_csv_by_qubits.py              # solo clásico + Aer
    python generate_csv_by_qubits.py --qryd       # añade QRydDemo
    python generate_csv_by_qubits.py --solo-qryd  # solo QRydDemo (si ya tienes clásico+Aer)
    python generate_csv_by_qubits.py --limpiar    # borra CSV previos antes de empezar
    python generate_csv_by_qubits.py --ibm --dry-run  # estima HW real IBM (no gasta QPU)
    python generate_csv_by_qubits.py --ibm            # ejecuta en HW real IBM (pide confirmación)

Nota IBM: --ibm NO usa QUBITS_OBJETIVO/CORRIDAS_POR_QUBIT (demasiado caros para
hardware real). Delega en quantum_search_ibm.ejecutar_lote(), que usa constantes
conservadoras (QUBITS_OBJETIVO_HW, CORRIDAS_POR_QUBIT_HW, SHOTS_HW) y agrupa todo
en un solo job de QPU. Es independiente de los modos clásico/Aer/QRyd.
"""

import argparse
from pathlib import Path

from classic_search import busqueda_clasica_masiva
from quantum_search_aer import main as ejecutar_cuantico

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "results"

QUBITS_OBJETIVO = [2, 4, 6, 8, 10, 12]
CORRIDAS_POR_QUBIT = 25


def _csv_clasico(bits):
    return RESULTADOS_DIR / f"results_classic_{bits}q.csv"


def _csv_cuantico(bits):
    return RESULTADOS_DIR / f"results_quantum_{bits}q.csv"


def _csv_qryd(bits):
    return RESULTADOS_DIR / f"results_qryd_{bits}q.csv"


def _limpiar_si_aplica(path_csv, limpiar):
    if limpiar and path_csv.exists():
        path_csv.unlink()
        print(f"  [limpio] {path_csv.name}")


def _ejecutar_qryd(bits, csv_path):
    """Importa y ejecuta QRydDemo solo si se solicita (evita importar token innecesariamente)."""
    from quantum_search_qryd import main as ejecutar_qryd
    ejecutar_qryd(bits=bits, csv_path=csv_path)


def _ejecutar_ibm(dry_run=False):
    """Ejecuta el lote en HARDWARE REAL de IBM (import perezoso del runtime)."""
    from quantum_search_ibm import ejecutar_lote
    ejecutar_lote(dry_run=dry_run)


def main(con_qryd=False, solo_qryd=False, limpiar=False,
         con_ibm=False, ibm_dry_run=False):
    RESULTADOS_DIR.mkdir(exist_ok=True)

    # El modo IBM (hardware real) es independiente y con presupuesto propio.
    # No entra en el bucle QUBITS_OBJETIVO porque seria demasiado caro en QPU.
    if con_ibm:
        print("=== Modo IBM (hardware cuántico real) ===")
        _ejecutar_ibm(dry_run=ibm_dry_run)
        return

    print("=== Generación de CSV por qubits ===")
    print(f"Qubits objetivo : {QUBITS_OBJETIVO}")
    print(f"Corridas/qubit  : {CORRIDAS_POR_QUBIT}")
    print(f"Clásico + Aer   : {not solo_qryd}")
    print(f"QRydDemo        : {con_qryd or solo_qryd}")

    for bits in QUBITS_OBJETIVO:
        csv_clasico = _csv_clasico(bits)
        csv_cuantico = _csv_cuantico(bits)
        csv_qryd = _csv_qryd(bits)

        if not solo_qryd:
            _limpiar_si_aplica(csv_clasico, limpiar)
            _limpiar_si_aplica(csv_cuantico, limpiar)
        if con_qryd or solo_qryd:
            _limpiar_si_aplica(csv_qryd, limpiar)

        print("\n" + "=" * 70)
        print(f"Lote: {bits} qubits  (N = {2**bits:,})")

        for corrida in range(1, CORRIDAS_POR_QUBIT + 1):
            print("\n" + "-" * 70)
            print(f"[{bits}q] Corrida {corrida}/{CORRIDAS_POR_QUBIT}")

            if not solo_qryd:
                print(f"  → Clásico")
                busqueda_clasica_masiva(bits=bits, csv_path=csv_clasico)

                print(f"  → Aer (simulador local)")
                ejecutar_cuantico(bits=bits, csv_path=csv_cuantico)

            if con_qryd or solo_qryd:
                print(f"  → QRydDemo")
                _ejecutar_qryd(bits=bits, csv_path=csv_qryd)

    print("\nProceso finalizado. CSV generados en results/.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Genera CSV de experimentos por cantidad de qubits."
    )
    parser.add_argument(
        "--qryd", action="store_true",
        help="Incluir QRydDemo además de clásico y Aer.",
    )
    parser.add_argument(
        "--solo-qryd", action="store_true",
        help="Ejecutar solo QRydDemo (asume que clásico y Aer ya tienen datos).",
    )
    parser.add_argument(
        "--limpiar", action="store_true",
        help="Borrar CSV previos antes de ejecutar.",
    )
    parser.add_argument(
        "--ibm", action="store_true",
        help="Ejecutar en hardware cuántico real de IBM (lote, presupuesto propio).",
    )
    parser.add_argument(
        "--dry-run", "--estimar", dest="dry_run", action="store_true",
        help="Con --ibm: solo estima consumo de QPU, sin enviar job.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(con_qryd=args.qryd, solo_qryd=args.solo_qryd, limpiar=args.limpiar,
         con_ibm=args.ibm, ibm_dry_run=args.dry_run)
