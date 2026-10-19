# Multitaper: orthogonale Fenster auf der ganzen Aufzeichnung – kein Zauber, aber ein gültiger Linientest – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-multitaper-demo.streamlit.app/)**

Vierte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos
zum **Periodogramm**, zu **Fenstern** und zu **Bartlett und Welch**. Ein simuliertes Signal (Hintergrund mit bekannter Dichte, bis zu zwei Linien) wird mit K Slepian-Tapern
analysiert und gegen Welch bei gleicher Rauschbandbreite gemessen; dazu der harmonische F-Test.

**Namensfalle:** Mit „Multi-…“-Demos anderer Reihen (`multi-fidelity-demo`, `multicriteria-demo`) hat diese Demo nichts zu tun.

## Warum dieses Problem

Die Vorgängerdemo endet bei „Welch verschenkt Auflösung; Multitaper nutzt die ganze Aufzeichnung“. Diese Demo prüft die Behauptung. Ergebnis: Bei gleicher Rauschbandbreite
liefert Welch mit Hann und 50 % Überlappung **mehr** unabhängige Mittel als das Multitaper; der Gesamtfehler ist bei glatten Spektren beim Multitaper 7 bis 9 % kleiner, wenn Welch nur auf
Zweierpotenzen der Segmentlänge abgesucht wird; mit feinerem Welch-Raster bleibt der Vorsprung nur bei sehr kurzen Aufzeichnungen (n = 128: 3 bis 5 %), bei steilen Spektren ist Welch nicht schlechter.
Was das Multitaper zusätzlich kann: kastenförmige Bandgrenze (trennt Linien ab der Bandbreite, für NW bis 4 belegt) und einen Linientest, dessen Falschalarm bei glatt gefärbtem Hintergrund eingehalten wird (nicht bei sehr schmalen Resonanzen).

## Modell

Hintergrund weiß, mit Resonanz (Pol 0,97 bei 200 Hz), „Steil“ (57 dB Spannweite) oder zwei scharfe Resonanzen (Pole 0,998 bei 100 und 300 Hz), Varianz auf 1 normiert, plus null bis
zwei Linien. Taper: Slepian-Folgen (Percival und Walden), selbst berechnet in numpy (Sturm-Bisektion der kommutierenden Tridiagonalmatrix, inverse Iteration); scipy dient nur als Orakel
in den Tests. Der Erwartungswert des gleichgewichteten Schätzers ist exakt berechenbar (Taper-Autokorrelationen). Die Abtastrate ist fest 1000 Hz.

## Methodik

n = 128 bis 4096, NW = 2 bis 12, K = 1 bis 2·NW Taper, gleiche oder adaptive Gewichte (Thomson). F-Test auf dem 8-fach aufgefüllten Raster, Schwelle für 1 % Falschalarm je Aufzeichnung
durch Bonferroni über alle Rasterpunkte (konservativ). Experimente über feste Zufallsströme (Seeds ab 400000), jede Zahl ein Mittel über 100 bis 400 Läufe.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py` (Anteile bei 100 bis 400 Läufen: Standardfehler 2 bis 5 Prozentpunkte).

| Frage | Befund |
|---|---|
| Wie gut sind die Taper konzentriert (n = 1024, NW = 4)? | Energie außerhalb des Bandes beim 1. / 3. / 7. / 8. / 9. Taper: 2,9·10⁻¹⁰ / 1,2·10⁻⁶ / 6,3·10⁻² / 0,30 / 0,70. Der Sprung liegt bei Taper 2·NW − 1 = 7. Zum Vergleich außerhalb von ±4 Bins: Rechteck 2,5·10⁻², Hann 1,4·10⁻⁵, Blackman 1,1·10⁻⁶. |
| Wie viele unabhängige Mittel bei gleicher Rauschbandbreite 4 / 6 / 8 / 12 Bins (n = 1024)? | Multitaper 3,0 / 5,0 / 7,1 / 11,1 (= K); Welch mit Hann und 50 % Überlappung 3,9 / 6,7 / 8,6 / 14,4 (Formel 3,8 / 6,7 / 8,6 / 14,3). Das Multitaper senkt die Streuung nicht stärker als Welch. |
| Gesamtfehler, bester gegen bester (n = 1024; Welch auf den Zweierpotenzen 16 bis 512 abgesucht, Multitaper mit fester Taperzahl K = 2NW − 1)? | Resonanz: 1,23 dB (Welch) gegen 1,14 dB (Multitaper, gleiche Gewichte); mit Welch in Schritten von 16 Messwerten 1,14 dB, also gleichauf. Steil: 1,47 dB (Welch) gegen 2,32 dB (gleich) und 1,55 dB (adaptiv); mit weniger Tapern (K = 13 bei NW = 8) 1,49 dB, die 2,32 dB folgen aus der festen Taperzahl. Zwei scharfe Resonanzen: 2,29 gegen 2,28 / 2,45. Bei n = 128 bleibt der Vorsprung des Multitapers bei feinerem Welch-Raster (Schritt 4) bei rund 5 % (Zweierpotenzen: 9 %), bei n = 256 sind beide mit feinem Raster gleichauf. |
| Leckage im Tal (Steil, n = 1024)? | Multitaper mit NW = 4 und gleichen Gewichten um 3,2 dB über der wahren Dichte (NW = 2: 2,8; NW = 8: 3,6); mit adaptiven Gewichten adaptiv 0,0 dB; Welch mit Hann und das Hann-Periodogramm 0,0 dB; Rechteck-Periodogramm 9,1 dB. |
| Wie dicht dürfen zwei gleich starke Linien liegen (10 dB je Linie, Einbruch mindestens 3 dB)? | Multitaper für NW bis 4 ab dem Abstand der Rauschbandbreite 2·NW Bins: NW = 2 bei 4 Bins, NW = 3 bei 6, NW = 4 bei 8 (97 %); bei NW = 6 bleibt es bei 12 Bins bei 78 %. Welch mit Hann bei gleicher Rauschbandbreite 6 Bins (L = 256): erst bei 8 Bins 45 %, bei 12 Bins 100 % (beim Kriterium „Einbruch mindestens 3 dB“; ein flacheres Kriterium zählt Welch früher als aufgelöst, das Multitaper bleibt vorn). |
| Linientest mit 1 % Falschalarm je Aufzeichnung (n = 1024, NW = 4, K = 7)? | Falschalarm bei Resonanz-Hintergrund: F-Test 0,5 %, Fisher-g 100 %; bei weißem Rauschen beide 0,8 %; im wählbaren Hintergrund „Zwei scharfe Resonanzen“ (Pole 0,998) steigt der Falschalarm des F-Tests auf rund 22 % statt 1 %. Entdeckung bei −18 / −14 / −12 / −10 / −8 dB je Abtastwert: F-Test 2 / 13 / 38 / 70 / 94 %; Fisher-g (nur weißes Rauschen) 9 / 66 / 97 / 100 / 100 %. |

## Ehrliche Grenzen

- **Der Vergleich mit Welch hängt von der Definition der Bandbreite ab** (Rauschbandbreite); mit der Trennschärfe verschiebt sich das Bild (siehe Auflösung).
- **Der F-Test gilt für eine Linie im Band**; bei zwei Linien im selben Band wird F klein. Bei gefärbtem Rauschen ist die F-Verteilung nur genähert: In „Resonanz“ und „Steil“ weicht die Verteilungsfunktion an den Gipfeln um 0,05 bis 0,25 ab, bei den schmalen Resonanzen von „Zwei scharfe Resonanzen“ um mehr als 0,5 (der gepoolte Falschalarm bleibt in „Resonanz“ und „Steil“ trotzdem unter dem Nennwert, dort nicht). Die 0,03 aus dem Orakel-Test gelten für einen milden Fall (n = 128, NW = 3, K = 5).
- **Bonferroni über das Raster ist konservativ:** Falschalarm 0,5 bis 0,8 % statt 1 %; der Preis ist Empfindlichkeit.
- **Adaptive Gewichte** sind eine Näherung mit eigener Verzerrung; es gibt keinen geschlossenen Erwartungswert, die App zeigt ihn dort nicht.
- **Die Vergleiche benutzen für Welch die beste Länge aus einer Reihe** (16 bis 512), für das Multitaper das beste NW aus (2, 3, 4, 6, 8, 12); ein anderes Raster kann den Vorsprung des Multitapers bei glatten Spektren (7 bis 9 %) weitgehend aufheben.
- **Stationär, bis zwei Linien, weißes oder AR-gefärbtes Rauschen, kein Trend.**

## Verifikation

- **Kern gegen Handrechnung:** n = 3, NW = 1: Tridiagonalmatrix (−0,5; 0; −0,5 / 1; 1), Eigenwert 1,1861, Taper (0,4544; 0,7662; 0,4544), (0,7071; 0; −0,7071), (0,5418; −0,6426; 0,5418),
  Konzentrationen 0,9936 / 0,8045 / 0,2019.
- **Orakel (`tests/test_oracle_multitaper.py`):** Taper und Konzentrationen gegen `scipy.signal.windows.dpss`; F-Schwellen gegen `scipy.stats.f`; F-Statistik unter H0 gegen die F(2, 2K − 2)-Verteilung
  (Kolmogorow-Smirnow); Erwartungswert gegen Simulation (Rauschen und Linien); K_eff = K; Welch gegen `scipy.signal.welch`.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder `st.slider` an den Grenzen; die Auswahlregler Länge n, NW und K nur in Einzeltests), Hintergründe, Gewichte, Permalink, K gegen NW, Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich) |
| `mtp_taper.py` | Slepian-Taper (numpy), Autokorrelation, Konzentration |
| `mtp_spectrum.py` | Eigenspektren, Schätzer (gleich, adaptiv), F-Test, Fisher-g, Erwartungswert, Auflösungskriterium |
| `mtp_welch.py` | Welch-Vergleich, Fenster, K_eff, Log-Momente (wortgleich aus Stück 3) |
| `mtp_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Presets |
| `mtp_evaluation.py` | Analyse, Experimente, Verdikte |
| `mtp_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `mtp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mtp_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Generator, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Eigenwert-Gewichte** (Gewichtung mit λ_k statt 1): liegt zwischen gleichen und adaptiven Gewichten und fügt keine neue Aussage hinzu.
- **Konfidenzintervalle und Falschalarmtheorie im Detail:** hier nur der F-Test; die Theorie folgt in einem späteren Stück.
- **Multitaper-Spektrogramm, mehrere Linien im Band, Trendabzug:** nicht enthalten.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf rund 25 Sekunden (gemessen auf dem Entwicklungsrechner, rechnerabhängig; danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html).
