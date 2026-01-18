Statistik-Projekt: Analyse des World Happiness Report 2015

Dieses Projekt wurde im Rahmen des Moduls **Statistik II** erstellt. Es untersucht mithilfe von Python und statistischen Modellen, welche Faktoren das nationale Glücksempfinden ("Happiness Score") am stärksten beeinflussen.

## Forschungsfrage & Hypothese

**Hypothese:**
> *"Soziale Bindungen ('Family') haben einen größeren, positiven Einfluss auf das nationale Glück als die reine wirtschaftliche Stärke ('Economy/GDP')."*

## Methodik

Verwendet wurde der Datensatz des **World Happiness Report 2015**.

Da die Explorative Datenanalyse (EDA) eine starke Korrelation zwischen Wirtschaftskraft (GDP) und sozialen Faktoren (Family) zeigte (**Multikollinearität**), wurden zwei konkurrierende **Multiple Lineare Regressionsmodelle (OLS)** erstellt und verglichen:

* **Modell A (Fokus Wirtschaft):** Prädiktor `GDP` (+ Kontrollvariablen)
* **Modell B (Fokus Soziales):** Prädiktor `Family` (+ Kontrollvariablen)

Die Modellgüte wurde anhand des **Adjusted R²** bewertet. Zudem wurden die Voraussetzungen (Normalverteilung der Residuen, Homoskedastizität) geprüft.

## Datei-Struktur

* `projekt_analyse.py` - Das Hauptskript (enthält Datenvorbereitung, Modellierung & Visualisierung).
* `world_happiness_report.csv` - Der verwendete Datensatz.
* `requirements.txt` - Liste der benötigten Python-Bibliotheken.
* `README.md` - Diese Projektdokumentation.
* `*.png` - Automatisch generierte Grafiken der Analyse (werden beim Ausführen erstellt).

## Installation & Ausführung

Um den Code auszuführen, folgen Sie diesen Schritten:

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/DEIN-GITHUB-USER/DEIN-REPO-NAME.git](https://github.com/DEIN-GITHUB-USER/DEIN-REPO-NAME.git)
    cd DEIN-REPO-NAME
    ```

2.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Analyse starten:**
    ```bash
    python projekt_analyse.py
    ```
    *Hinweis: Während der Ausführung öffnen sich Fenster mit Visualisierungen. Bitte schließen Sie diese jeweils, damit das Skript fortfährt.*

## Ergebnisse

Die Analyse lieferte folgende Resultate:

* **Modell A (Wirtschafts-Fokus):** Erklärt **70.6%** der Varianz ($R^2 = 0.706$).
* **Modell B (Sozial-Fokus):** Erklärt **64.1%** der Varianz ($R^2 = 0.641$).

**Fazit:**
Die ursprüngliche Hypothese muss statistisch **verworfen** werden. Das Bruttoinlandsprodukt ist als Einzelindikator erklärungskräftiger als der Faktor Familie.

**Differenzierte Betrachtung:**
Die Analyse der Koeffizienten zeigt jedoch, dass der Faktor `Family` (Koeffizient: 2.57) einen steileren positiven Einfluss auf das Glück hat als `GDP` (Koeffizient: 1.90), wenn er isoliert betrachtet wird. Zudem zeigt sich, dass in Modell B der Faktor **"Vertrauen in die Regierung"** signifikant wird ($p < 0.05$), während er im Wirtschaftsmodell keine Rolle spielt.

## Autor

**Leo Hischier**
Hochschule Luzern (HSLU) - Modul Statistik II