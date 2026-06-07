# Rozpoznawanie tęczówki

System biometryczny oparty na algorytmie Daugmana — segmentacja oka, normalizacja paska tęczówki, kodowanie filtrami Gabora i porównywanie odległością Hamminga z kompensacją rotacji.

## Wymagania

- Python 3.10+
- OpenCV, NumPy, Matplotlib
- Tkinter (interfejs graficzny, zwykle w standardowej instalacji Pythona)

```bash
pip install numpy opencv-python matplotlib
```

## Dane

Umieść zdjęcia w katalogu `data/` w strukturze:

```
data/
└── 001/
    └── L/
        ├── S5001L01.jpg   # wzorzec (enrollment)
        └── S5001L02.jpg   # weryfikacja
```

Projekt jest przygotowany pod zbiór [CASIA Iris](http://www.cbsr.ia.ac.cn/english/IrisDatabase.asp) (format `S500XL0Y.jpg`).

## Uruchomienie

Wszystkie komendy uruchamiaj z katalogu `src/`:

```bash
cd src
```

| Skrypt | Opis |
|--------|------|
| `python enrollment.py` | Rejestracja użytkowników — buduje bazę w `memory-db/` |
| `python verification.py` | Weryfikacja 1:N względem zapisanej bazy |
| `python evaluation.py` | Ewaluacja w terminalu (legalni użytkownicy / impostorzy) |
| `python app.py` | Interfejs graficzny z testami i wykresami metryk |

## Pipeline

1. **Segmentacja** — wykrycie źrenicy i tęczówki (`segmentation.py`)
2. **Normalizacja** — model rubber sheet, pasek 64×512 px (`pipeline.py`)
3. **Kodowanie** — 4 orientacje × 2 fazy Gabora → 8-bitowy kod (`pipeline.py`)
4. **Dopasowanie** — odległość Hamminga z przesunięciem w poziomie (`matcher.py`)

Parametry (próg dopasowania, filtry Gabora itd.) w `config.py`.

## Struktura projektu

```
src/
├── pipeline.py          # główny pipeline przetwarzania
├── segmentation.py      # wykrywanie źrenicy i tęczówki
├── matcher.py           # porównywanie kodów
├── enrollment.py        # rejestracja do bazy
├── verification.py      # weryfikacja użytkownika
├── evaluation.py        # ewaluacja (CLI)
├── evaluation_with_visual.py
├── app.py               # GUI (Tkinter)
└── config.py            # parametry systemu

memory-db/               # wygenerowane kody i mapa bazy
data/                    # zdjęcia (nie w repozytorium)
```
