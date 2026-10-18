"""Konstanten: Regler-Grenzen, feste Annahmen, Hintergründe, Experimente, Farben und Presets (Kern: mtp_taper.py, mtp_spectrum.py, mtp_welch.py)."""

FS = 1000.0                                  # Abtastrate in Hz (feste Annahme)
N_CHOICES = (128, 256, 512, 1024, 2048, 4096)
DEFAULT_N = 1024
NW_CHOICES = (2, 3, 4, 5, 6, 8, 10, 12)
DEFAULT_NW = 4
DEFAULT_K = 7                                # 2 NW - 1
WEIGHTS = ("gleich", "adaptiv")
DEFAULT_WEIGHTS = "gleich"
PAD = 4                                      # Auffüllfaktor der Spektren für die Darstellung
PADF = 8                                     # Auffüllfaktor des F-Tests (Linie zwischen zwei Bins)
F_GUARD = 64                                 # Randpunkte des F-Rasters (8 Bins mal PADF), die beim Linientest nicht zählen
LINES = ("keine", "eine", "zwei")
DEFAULT_LINES = "eine"
LINE_MIN, LINE_MAX, LINE_STEP, DEFAULT_LINE = 0.05, 0.43, 0.0005, 0.35        # Linie 1 als Bruchteil der Abtastrate
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 1.0, 24.0, 0.5, 8.0                 # Abstand der Linie 2 in Bins von f_s / n
REL_MIN, REL_MAX, DEFAULT_REL = -40, 0, 0                                      # Stärke der Linie 2 gegen Linie 1 in dB
SNR_MIN, SNR_MAX, DEFAULT_SNR = -30, 30, -10                                   # Linie 1 gegen Rauschvarianz 1, in dB je Abtastwert
SEED_MAX, DEFAULT_SEED = 999999, 37
ALPHA = 0.01                                 # Falschalarm je Aufzeichnung beim Linientest

# Pole (Frequenz als Bruchteil der Abtastrate, Radius) je Hintergrund; Varianz auf 1 normiert
BACKGROUNDS = {"weiß": (), "Resonanz": ((0.2, 0.97),), "Steil": ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9)),
               "Zwei scharfe Resonanzen": ((0.1, 0.998), (0.3, 0.998))}
DEFAULT_BACKGROUND = "Resonanz"

# Experimente (feste Ströme)
SWEEP_SEED = 400000
EXP_N = 1024
EXP_NW = (2, 3, 4, 6)
EXP_CONC_BAND_NW = 4
EXP_BW_RUNS = 400
EXP_BEST_NS = (128, 256, 1024)
EXP_BEST_BACKGROUNDS = ("Resonanz", "Steil", "Zwei scharfe Resonanzen")
EXP_BEST_NW = (2, 3, 4, 6, 8, 12)
EXP_BEST_SEGS = (16, 32, 64, 128, 256, 512)
EXP_BEST_RUNS = 100
EXP_VALLEY_RUNS = 300
EXP_VALLEY_WELCH_SEG = 256
EXP_RES_DELTAS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
EXP_RES_NW = (2, 3, 4, 6)
EXP_RES_SEGS = (512, 256, 128)
EXP_RES_RUNS = 150
EXP_RES_SNR = 10
EXP_TEST_NW = 4
EXP_TEST_BACKGROUNDS = ("weiß", "Resonanz", "Steil")
EXP_TEST_RUNS = 400
EXP_TEST_SNR = (-18, -16, -14, -12, -10, -8)
EXP_TEST_SEG = 292

COLOR_MT = "rgba(76,120,168,0.95)"
COLOR_WELCH = "rgba(150,150,150,0.85)"
COLOR_EXPECTED = "#E8833A"
COLOR_TRUE = "#2F6B65"
COLOR_LINE = "#B03A2E"
COLOR_TAPER = "#7B5EA7"

PRESETS = {
    "Gleiche Bandbreite": {"n": 1024, "nw": 4, "k": 7, "weights": "gleich", "background": "Resonanz", "lines": "keine", "line": 0.2, "gap": 8.0, "rel": 0, "snr": -10,
                           "seed": DEFAULT_SEED},
    "Steiler Hintergrund": {"n": 1024, "nw": 4, "k": 7, "weights": "gleich", "background": "Steil", "lines": "keine", "line": 0.2, "gap": 8.0, "rel": 0, "snr": -10,
                            "seed": DEFAULT_SEED},
    "Zwei Linien an der Bandgrenze": {"n": 1024, "nw": 4, "k": 7, "weights": "gleich", "background": "weiß", "lines": "zwei", "line": 0.2, "gap": 8.0, "rel": 0,
                                      "snr": 10, "seed": DEFAULT_SEED},
    "Linie im gefärbten Rauschen": {"n": 1024, "nw": 4, "k": 7, "weights": "gleich", "background": "Resonanz", "lines": "eine", "line": 0.35, "gap": 8.0, "rel": 0,
                                    "snr": -8, "seed": DEFAULT_SEED},
    "Kurze Aufzeichnung": {"n": 128, "nw": 3, "k": 5, "weights": "gleich", "background": "Resonanz", "lines": "keine", "line": 0.2, "gap": 8.0, "rel": 0, "snr": -10,
                           "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Gleiche Bandbreite": "Resonanz im Hintergrund, 7 Taper mit NW = 4 gegen Welch bei gleicher Rauschbandbreite (8 Bins): Welch hat mehr unabhängige Mittel, das Multitaper ein kastenförmiges Spektralfenster.",
    "Steiler Hintergrund": "57 dB Spannweite, gleiche Gewichte: Das Tal liegt um 3,2 dB zu hoch. Auf „adaptiv“ stellen: Die Gewichte holen es auf die wahre Dichte.",
    "Zwei Linien an der Bandgrenze": "Zwei gleich starke Linien genau 2NW = 8 Bins auseinander: Das Multitaper trennt sie gerade, Welch bei gleicher Bandbreite nicht; mit NW = 5 oder 6 verschmelzen sie.",
    "Linie im gefärbten Rauschen": "Eine Linie bei −8 dB je Abtastwert auf einem Hintergrund mit Resonanz: Der F-Test findet sie bei gültigem Falschalarm, der Fisher-g-Test schlägt wegen des gefärbten Hintergrunds überall an.",
    "Kurze Aufzeichnung": "Nur 128 Messwerte, 5 Taper mit NW = 3: Welch muss die Segmente bei gleicher Bandbreite auf 32 Messwerte kürzen, das Multitaper behält die Länge 128; der Gesamtfehler ist trotzdem nur etwa 3 bis 9 % kleiner (je nach Welch-Raster; Beste-gegen-beste-Tabelle).",
}
