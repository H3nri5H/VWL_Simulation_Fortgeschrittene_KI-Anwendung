# VWL-Simulation-RL

**Multi-Agent Reinforcement Learning Simulation for Economics**

**Autoren: Nathan Blanck & Henri Herdel**

**Modul: Fortgeschrittene KI-Anwendung**

**Jahrgang: WINF123**

## Installation auf MacOS

### Voraussetzungen

- **Python 3.11.9**
- **Git**

### Schritt 1: Repository klonen

```bash
git clone https://github.com/H3nri5H/VWL_Simulation_Fortgeschrittene_KI-Anwendung.git
```

### Schritt 2: Virtual Environment erstellen

```bash
python -m venv venv
source venv/bin/activate
```

### Schritt 3: Requirements installieren

```bash
pip install -r requirements.txt
```

## 1. Training starten

```bash
python train.py --resume
```

"--resume" sorgt dafür, dass das Training beim letzten Checkpoint fortgeführt wird. Ohne das würde das Training bei Null beginnen.

**Was passiert:**

- Training läuft basierend auf der Anzahl an Iterationen die in der config.yaml gespeichert sind:
  training:
  iterations: 100
- Checkpoints werden alle **20 Iterationen** gespeichert (auch festlegbar in der config)
- Progress wird in der Console angezeigt

**Ausgabe:**

```
======================================================================
  VWL SIMULATION - TRAINING
======================================================================

Fresh training: 200 iterations
Environment: 10 firms, 3000 households
Resources: 4 workers, 0 GPUs

Building algorithm...
Algorithm built

----------------------------------------------------------------------
Iter   Reward       Min        Max        EpLen
----------------------------------------------------------------------
1      -2.34        -15.23     8.45       100
2      -1.89        -12.34     9.12       100
...
```

## 2. Simulation ausführen

```bash
python run_simulation.py
```

**Interaktiver Prozess:**

1. **Checkpoint auswählen** (z.B. Iteration 200)
2. **Seed eingeben** (für Reproduzierbarkeit) oder ENTER für random
3. **Steps festlegen** (Standard wird aus der config extrahiert)
4. Simulation läuft und speichert CSV-Dateien in `simulation_results/`

**Ergebnis:**

- .csv Datei für alle Unternehmen
- .csv Datei für alle Haushalte
- .txt für eine zusammenfassung der groben Simulationsergebnisse
