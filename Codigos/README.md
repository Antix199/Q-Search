# ProyectoCuantica — Q-Search

Comparación experimental entre búsqueda clásica por fuerza bruta y el algoritmo
de Grover, bajo condiciones idénticas (mismo oráculo, misma llave objetivo,
mismas iteraciones óptimas), en cuatro plataformas:

1. **Clásico** — búsqueda secuencial (baseline).
2. **Aer** — simulador local de Qiskit (CPU/GPU).
3. **QRydDemo** — emulador de átomos de Rydberg.
4. **IQM Resonance** — hardware cuántico real superconductor (plan freemium).

## Archivos

### Implementaciones de búsqueda
- `classic_search.py`: búsqueda clásica secuencial y exportación a CSV.
- `quantum_search_aer.py`: Grover en el simulador Aer local (GPU con fallback a CPU).
- `quantum_search_qryd.py`: Grover en el simulador QRydDemo (requiere `QRYD_API_TOKEN`).
- `quantum_search_iqm.py`: Grover en **hardware cuántico real** de IQM Resonance
  vía `iqm-client[qiskit]` + `backend.run` (requiere `IQM_TOKEN`, `IQM_SERVER_URL`
  e `IQM_DEVICE`). Mismo Grover, mismo barrido de k, mismas métricas y mismas
  columnas CSV que los demás modelos.

### Orquestación y análisis
- `run_all.py`: ejecuta clásico + Aer + QRydDemo en una sola corrida y, al final,
  llama a `compare_searches`.
- `compare_searches.py`: análisis comparativo exhaustivo y generación de las
  9 figuras + tabla comparativa.

### Datos
- `results/results_classic_{bits}q.csv`, `results_quantum_{bits}q.csv`,
  `results_qryd_{bits}q.csv`, `results_iqm_{bits}q.csv` — historial por nivel.
- `results/graficos/` — figuras (`fig1`–`fig9`) y tabla comparativa
  (`full_comparison_table.csv`).

> Nota: los archivos de código están en inglés; las funciones/variables internas
> se mantienen en español.

## Requisitos

```bash
pip install -r requirements.txt
```

> **Convivencia de versiones:** `iqm-client` fija `qiskit<2.2`. Todos los módulos
> importan y corren con qiskit 2.1.x, pero `pip` puede mostrar avisos de versión.
> Si necesitas un qiskit más nuevo para otra cosa, usa un venv aparte para IQM.

## Configuración de credenciales (`.env`)

En la raíz del proyecto, un archivo `.env` (nunca se sube al repo) con:

```
QRYD_API_TOKEN=<token de QRydDemo>          # solo para quantum_search_qryd.py
IQM_TOKEN=<token del dashboard de IQM Resonance>   # solo para quantum_search_iqm.py
IQM_SERVER_URL=https://resonance.iqm.tech          # raíz común de Resonance
IQM_DEVICE=<alias del dispositivo, p.ej. emerald>  # campo "Quantum computer alias"
```

## Flujo de ejecución completo

```bash
# 1. Clásico + Aer + QRydDemo (25 corridas por nivel)
python run_all.py

# 2. Hardware real IQM (niveles n = 2 y n = 4)
python quantum_search_iqm.py --dry-run     # estima consumo, NO gasta QPU
python quantum_search_iqm.py               # ejecuta (pide confirmación 'si')

# 3. Análisis comparativo final (ya con datos de IQM)
python compare_searches.py
```

Cada modelo también se puede correr por separado:

```bash
python classic_search.py
python quantum_search_aer.py
python quantum_search_qryd.py
python quantum_search_iqm.py --qubits 4    # solo un nivel concreto
```

### Hardware real de IQM Resonance

`quantum_search_iqm.py` transpila a ISA (`generate_preset_pass_manager`) y ejecuta
con `backend.run`. El barrido de k de cada corrida se agrupa en un job; el modo
lote agrupa las 25 corridas de un nivel y las trocea en bloques de ≤ 100 circuitos
(límite del dispositivo). Los CSV `results_iqm_{bits}q.csv` usan las mismas
columnas que los demás modelos, por lo que `compare_searches.py` los analiza sin
cambios. Usa siempre `--dry-run` primero para ver el consumo.

## Configuración

Las constantes ajustables están al inicio de cada script:

- `classic_search.py`: `BITS`.
- `quantum_search_aer.py`: `NUM_QUBITS`, `SHOTS`, `SEED_SIMULADOR`,
  `SEED_TRANSPILER`, `FACTOR_MAX_BUSQUEDA`, `UMBRAL_FRECUENCIA_OBJETIVO`.
- `quantum_search_qryd.py`: `NUM_QUBITS`, `QRYD_BACKEND_NAME`, `SHOTS`.
- `quantum_search_iqm.py`: `QUBITS_OBJETIVO_HW`, `CORRIDAS_POR_QUBIT_HW`, `SHOTS`,
  `MAX_CIRCUITOS_POR_JOB`, `IQM_SERVER_URL_DEFAULT`, `IQM_DEVICE_DEFAULT`.
- `run_all.py`: `QUBITS_CLASICO_AER`, `QUBITS_QRYD`, `CORRIDAS`.

## Nota sobre GPU (NVIDIA)

`quantum_search_aer.py` intenta usar GPU con Aer. Si el entorno no tiene build
CUDA de Aer o no detecta la GPU, cae automáticamente a CPU.
