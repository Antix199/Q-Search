# ProyectoCuantica — Q-Search

Comparación experimental entre búsqueda clásica por fuerza bruta y el algoritmo
de Grover, bajo condiciones idénticas (mismo oráculo, misma llave objetivo,
mismas iteraciones óptimas), en cuatro plataformas:

1. **Clásico** — búsqueda secuencial (baseline).
2. **Aer** — simulador local de Qiskit (CPU/GPU).
3. **QRydDemo** — emulador de átomos de Rydberg.
4. **IBM Quantum** — hardware cuántico real (Open Plan).

## Archivos

### Implementaciones de búsqueda
- `classic_search.py`: búsqueda clásica secuencial y exportación a CSV.
- `quantum_search_aer.py`: Grover en el simulador Aer local (GPU con fallback a CPU).
- `quantum_search_qryd.py`: Grover en el simulador QRydDemo (requiere `QRYD_API_TOKEN`).
- `quantum_search_ibm.py`: Grover en **hardware cuántico real** de IBM Quantum
  Platform vía `qiskit-ibm-runtime` + SamplerV2 (requiere `IBM_QUANTUM_TOKEN` y
  `IBM_QUANTUM_CRN`).

### Orquestación y análisis
- `generate_csv_by_qubits.py`: ejecuta lotes y crea CSV separados por qubits.
- `run_all.py`: ejecuta clásico + Aer (+ QRydDemo) en una sola corrida y analiza.
- `compare_searches.py`: análisis comparativo exhaustivo y generación de gráficos.
- `analyze_results.py`: promedios y gráfico de líneas desde los CSV.

### Datos
- `results/results_classic_{bits}q.csv`, `results_quantum_{bits}q.csv`,
  `results_qryd_{bits}q.csv`, `results_ibm_{bits}q.csv` — historial por nivel.
- `results/graficos/` — figuras y tabla comparativa (`full_comparison_table.csv`).

> Nota: los archivos de código están en inglés; las funciones/variables internas
> se mantienen en español.

## Requisitos

```bash
pip install -r requirements.txt
```

## Configuración de credenciales (`.env`)

En la raíz del proyecto, un archivo `.env` con:

```
QRYD_API_TOKEN=<token de QRydDemo>          # solo para quantum_search_qryd.py
IBM_QUANTUM_TOKEN=<API key de ~44 chars>    # solo para quantum_search_ibm.py
IBM_QUANTUM_CRN=<CRN o nombre de la instancia del Open Plan>
```

## Ejecución rápida

```bash
# Cada modelo por separado
python classic_search.py
python quantum_search_aer.py
python quantum_search_qryd.py

# Lotes por qubits
python generate_csv_by_qubits.py            # clásico + Aer
python generate_csv_by_qubits.py --qryd     # añade QRydDemo

# Análisis y gráficos
python compare_searches.py
python analyze_results.py
```

### Hardware real de IBM (consume ~10 min de QPU/mes en el Open Plan)

```bash
python quantum_search_ibm.py --dry-run      # estima consumo, NO envía job
python quantum_search_ibm.py                # ejecuta (pide confirmación 'si')
python generate_csv_by_qubits.py --ibm      # mismo lote vía orquestador
```

En hardware real **no se hace barrido de k**: se ejecuta solo la iteración óptima
teórica de Grover, con constantes conservadoras (`QUBITS_OBJETIVO_HW`,
`CORRIDAS_POR_QUBIT_HW`, `SHOTS_HW`) definidas arriba de `quantum_search_ibm.py`.
Los CSV `results_ibm_{bits}q.csv` usan las mismas columnas que Aer, por lo que
`compare_searches.py` los analiza sin cambios.

## Configuración

Las constantes ajustables están al inicio de cada script:

- `classic_search.py`: `BITS`.
- `quantum_search_aer.py`: `NUM_QUBITS`, `SHOTS`, `SEED_SIMULADOR`,
  `SEED_TRANSPILER`, `FACTOR_MAX_BUSQUEDA`, `UMBRAL_FRECUENCIA_OBJETIVO`,
  `VERBOSE_ITERACIONES`.
- `generate_csv_by_qubits.py`: `QUBITS_OBJETIVO`, `CORRIDAS_POR_QUBIT`.
- `quantum_search_ibm.py`: `QUBITS_OBJETIVO_HW`, `CORRIDAS_POR_QUBIT_HW`,
  `SHOTS_HW`.

## Nota sobre GPU (NVIDIA)

`quantum_search_aer.py` intenta usar GPU con Aer. Si el entorno no tiene build
CUDA de Aer o no detecta la GPU, cae automáticamente a CPU.
