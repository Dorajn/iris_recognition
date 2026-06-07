# Iris Recognition

Biometric system based on Daugman's algorithm — eye segmentation, iris strip normalization, Gabor filter encoding, and Hamming distance matching with rotation compensation.

## Requirements

- Python 3.10+
- OpenCV, NumPy, Matplotlib
- Tkinter (GUI, usually included with a standard Python installation)

```bash
pip install numpy opencv-python matplotlib
```

## Data

Place images in the `data/` directory using the following structure:

```
data/
└── 001/
    └── L/
        ├── S5001L01.jpg   # template (enrollment)
        └── S5001L02.jpg   # verification
```

The project is designed for the [CASIA Iris](http://www.cbsr.ia.ac.cn/english/IrisDatabase.asp) dataset (`S500XL0Y.jpg` naming format).

## Usage

Run all commands from the `src/` directory:

```bash
cd src
```

| Script | Description |
|--------|-------------|
| `python enrollment.py` | User enrollment — builds the database in `memory-db/` |
| `python verification.py` | 1:N verification against the stored database |
| `python evaluation.py` | Terminal evaluation (genuine users / impostors) |
| `python app.py` | Graphical interface with tests and metric plots |

## Pipeline

1. **Segmentation** — pupil and iris detection (`segmentation.py`)
2. **Normalization** — rubber sheet model, 64×512 px strip (`pipeline.py`)
3. **Encoding** — 4 orientations × 2 Gabor phases → 8-bit code (`pipeline.py`)
4. **Matching** — Hamming distance with horizontal shift (`matcher.py`)

Parameters (match threshold, Gabor filters, etc.) are defined in `config.py`.

## Project Structure

```
src/
├── pipeline.py          # main processing pipeline
├── segmentation.py      # pupil and iris detection
├── matcher.py           # code comparison
├── enrollment.py        # database enrollment
├── verification.py      # user verification
├── evaluation.py        # evaluation (CLI)
├── evaluation_with_visual.py
├── app.py               # GUI (Tkinter)
└── config.py            # system parameters

memory-db/               # generated codes and database map
data/                    # images (not tracked in the repository)
```
