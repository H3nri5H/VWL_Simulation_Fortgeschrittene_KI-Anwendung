# VWL-Simulation-RL

**Multi-Agent Reinforcement Learning Simulation for Economics**
**Autoren: Nathan Blanck & Henri Herdel**
**Modul: Fortgeschrittene KI-Anwendung**
**Jahrgang: WINF123**


## Installation

### Voraussetzungen

- **Python 3.11.9** 
- **Git** (um das Repository zu klonen)

### Schritt 1: Repository klonen

### Schritt 2: Python Virtual Environment erstellen

#### Windows (CMD/PowerShell)

```bash
python -m venv .venv
venv\Scripts\activate
```

**Hinweis**: Wenn `venv\Scripts\activate` nicht funktioniert (PowerShell), versuche:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

### Schritt 3: Dependencies installieren

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## ⚡ Schnellstart

### 1. Training starten

```bash
python train.py
```

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

**Alternativ:**

Um vorheriges Training nicht zu löschen empfehlen wir zum "Anschauen" folgenden Trainingsbefehl zu verwenden:

```bash
python train.py --resume
```

Auf diese Art wird ab dem letzten gespeichert Checkpoint das Training fortgesetzt und beginnt somit ncht von vorne.
Das Training kann über Strg+C abgerochen werden. 

### 2. Simulation ausführen

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
