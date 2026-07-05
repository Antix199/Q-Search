"""Analiza CSV de resultados y genera graficos comparativos por qubits."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "results"
QUBITS_OBJETIVO = [2, 6, 8, 10, 12]


def leer_metricas(path_csv):
    """Lee iteraciones y tiempo de un CSV de resultados."""
    if not path_csv.exists():
        raise FileNotFoundError(f"No existe el archivo: {path_csv}")

    filas = []
    with path_csv.open("r", encoding="utf-8", newline="") as archivo:
        reader = csv.DictReader(archivo)
        for fila in reader:
            if not fila:
                continue
            try:
                iteraciones = int(fila["iteraciones_requeridas"])
                tiempo = float(fila["tiempo_busqueda_s"])
            except (KeyError, TypeError, ValueError):
                continue

            filas.append(
                {
                    "iteraciones_requeridas": iteraciones,
                    "tiempo_busqueda_s": tiempo,
                }
            )

    if not filas:
        raise ValueError(f"No se encontraron filas validas en: {path_csv}")

    return filas


def promedio(valores):
    return sum(valores) / len(valores)


def imprimir_resumen(nombre_modelo, filas):
    iteraciones = [fila["iteraciones_requeridas"] for fila in filas]
    tiempos = [fila["tiempo_busqueda_s"] for fila in filas]

    print(f"\nModelo: {nombre_modelo}")
    print(f"- Corridas: {len(filas)}")
    print(f"- Promedio iteraciones: {promedio(iteraciones):.2f}")
    print(f"- Promedio tiempo (s): {promedio(tiempos):.6f}")


def graficar_iteraciones(filas_clasico, filas_cuantico, salida, qubits):
    """Genera un grafico independiente por cantidad de qubits."""
    x_clasico = list(range(1, len(filas_clasico) + 1))
    y_clasico = [fila["iteraciones_requeridas"] for fila in filas_clasico]

    x_cuantico = list(range(1, len(filas_cuantico) + 1))
    y_cuantico = [fila["iteraciones_requeridas"] for fila in filas_cuantico]

    plt.figure(figsize=(12, 6))
    plt.plot(x_clasico, y_clasico, marker="o", linewidth=1.8, label="Clasico")
    plt.plot(x_cuantico, y_cuantico, marker="s", linewidth=1.8, label="Cuantico")
    plt.title(
        "Iteraciones Requeridas por Número de Ejecuciones del Código "
        f"en Espacio de Búsqueda de 2^{qubits}"
    )
    plt.xlabel("Numero de Ejecución")
    plt.ylabel("Iteraciones requeridas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida, dpi=180)
    plt.close()


def _csv_clasico(qubits):
    return RESULTADOS_DIR / f"results_classic_{qubits}q.csv"


def _csv_cuantico(qubits):
    return RESULTADOS_DIR / f"results_quantum_{qubits}q.csv"


def _grafico_salida(qubits):
    return RESULTADOS_DIR / f"comparacion_iteraciones_{qubits}q.png"


def analizar_por_qubits(qubits):
    csv_clasico = _csv_clasico(qubits)
    csv_cuantico = _csv_cuantico(qubits)
    salida_grafico = _grafico_salida(qubits)

    filas_clasico = leer_metricas(csv_clasico)
    filas_cuantico = leer_metricas(csv_cuantico)

    print("\n" + "=" * 72)
    print(f"Comparacion para {qubits} qubits")
    imprimir_resumen("Clasico", filas_clasico)
    imprimir_resumen("Cuantico", filas_cuantico)

    graficar_iteraciones(filas_clasico, filas_cuantico, salida_grafico, qubits)
    print(f"\nGrafico guardado en: {salida_grafico}")


def main():
    for qubits in QUBITS_OBJETIVO:
        try:
            analizar_por_qubits(qubits)
        except (FileNotFoundError, ValueError) as error:
            print("\n" + "=" * 72)
            print(f"Comparacion para {qubits} qubits")
            print(f"[ADVERTENCIA] {error}")


if __name__ == "__main__":
    main()

