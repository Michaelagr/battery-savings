# ⚡ Energie-Kosten-Analyse Dashboard

Ein umfassendes Streamlit-Dashboard zur Analyse der Energiekosten für deutsche Industrieunternehmen mit verschiedenen Szenarien für PV-Anlagen und Batteriespeicher.

## 🚀 Features

### 📊 Vier Szenarien-Analyse
1. **Baseline**: Grundlast ohne PV oder Batterie
2. **PV-Anlage**: Integration einer Photovoltaikanlage
3. **PV + Einfache Batterie**: Eigenverbrauchsoptimierung
4. **Smart Battery**: Intelligente Batterieoptimierung mit Peak-Shaving und Lastverschiebung

### ⚙️ Konfigurierbare Parameter
- **PV-Leistung**: Skalierbare PV-Anlage (0-10.000 kWp)
- **Batterie**: Leistung (0-1.000 kW) und Kapazität (0-5.000 kWh)
- **Kostenparameter**: Strompreis, Netzentgelt, EEG-Umlage, Mehrwertsteuer
- **Leistungspreis**: Unterschiedliche Tarife für >2500h und <2500h Vollaststunden

### 📈 Detaillierte Analysen
- Kostenvergleich aller Szenarien
- Lastprofil-Visualisierung
- PV-Generierung vs. Verbrauch
- Batterie-Performance-Analyse
- Detaillierte Kostenaufschlüsselung
- Einsparungsanalyse

## 🛠️ Installation

### Voraussetzungen
- Python 3.8 oder höher
- pip (Python Package Manager)

### Setup
1. **Virtuelle Umgebung aktivieren**:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Abhängigkeiten installieren**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dashboard starten**:
   ```bash
   streamlit run dashboard.py
   ```

## 📁 Daten-Upload

### Erforderliches Format
Die Lastprofil-Daten müssen folgende Spalten enthalten:
- `timestamp`: Zeitstempel (YYYY-MM-DD HH:MM:SS)
- `load_kw`: Last in kW (15-Minuten-Intervalle)

### Unterstützte Formate
- CSV-Dateien (.csv)
- Excel-Dateien (.xlsx, .xls)

### Beispiel-Datenstruktur
```csv
timestamp,load_kw
2024-01-01 00:00:00,450.2
2024-01-01 00:15:00,432.1
2024-01-01 00:30:00,445.8
...
```

## 🔧 Konfiguration

### PV-Konfiguration
- **PV-Leistung (kWp)**: Skalierungsfaktor für das PV-Lastprofil
- Standardwert: 400 kWp

### Batterie-Konfiguration
- **Batterie-Leistung (kW)**: Maximale Lade-/Entladeleistung
- **Batterie-Kapazität (kWh)**: Speicherkapazität
- Standardwerte: 100 kW / 200 kWh

### Kosten-Parameter
- **Basis-Strompreis**: €/MWh (Standard: 80 €/MWh)
- **Netzentgelt**: €/MWh (Standard: 25 €/MWh)
- **Leistungspreis >2500h**: €/kW/Jahr (Standard: 120 €/kW/Jahr)
- **Leistungspreis <2500h**: €/kW/Jahr (Standard: 80 €/kW/Jahr)
- **EEG-Umlage**: €/MWh (Standard: 6.5 €/MWh)
- **Mehrwertsteuer**: % (Standard: 19%)

## 📊 Szenarien im Detail

### Szenario 1: Baseline
- Berechnung der Grundkosten ohne erneuerbare Energien
- Berücksichtigung aller Kostenkomponenten
- Referenzpunkt für alle weiteren Analysen

### Szenario 2: PV-Anlage
- Integration einer skalierbaren PV-Anlage
- Berechnung des Eigenverbrauchs
- Analyse des PV-Überschusses
- Reduzierung der Netzlast

### Szenario 3: PV + Einfache Batterie
- Eigenverbrauchsoptimierung mit Batteriespeicher
- Laden bei PV-Überschuss
- Entladen bei Netzverbrauch
- Verbesserung der Eigenverbrauchsquote

### Szenario 4: Smart Battery
- Intelligente Batterieoptimierung
- **Peak-Shaving**: Reduzierung der Spitzenlast
- **Lastverschiebung**: Optimierung basierend auf Strompreisen
- **Eigenverbrauchsoptimierung**: Maximierung der PV-Nutzung

## 📈 KPIs und Metriken

### Kosten-Metriken
- Gesamtkosten pro Jahr
- Energiekosten vs. Leistungspreis
- Einsparungen im Vergleich zur Baseline
- Prozentuale Kosteneinsparung

### Technische Metriken
- Maximale Last (kW)
- Jahresverbrauch (MWh)
- PV-Eigenverbrauchsquote (%)
- Batterie-Zyklen pro Jahr
- Peak-Reduktion (%)

### PV-Metriken
- PV-Generierung (MWh/Jahr)
- PV-Überschuss (MWh/Jahr)
- Eigenverbrauchsrate (%)
- Netto-Last nach PV

## 🔍 Detaillierte Analyse

### Lastprofil-Tab
- Jahreslastprofil-Visualisierung
- Tägliche Durchschnittslast
- Statistische Kennzahlen

### PV-Profil-Tab
- PV-Generierung vs. Verbrauch
- Saisonale Muster
- Eigenverbrauchsanalyse

### Batterie-Analyse-Tab
- Batterie-Ladezustand über Zeit
- Lade-/Entladeleistung
- Performance-Optimierung

### Kostendetails-Tab
- Detaillierte Kostenaufschlüsselung
- Einsparungsanalyse
- Vergleich aller Szenarien

## 🎯 Anwendungsfälle

### Industrieunternehmen
- Analyse der Energiekosten
- Bewertung von PV-Investitionen
- Batteriespeicher-Optimierung
- Peak-Shaving-Strategien

### Energieberater
- Kundenberatung
- Szenarienvergleich
- Wirtschaftlichkeitsanalysen
- Technische Optimierung

### Projektentwickler
- Machbarkeitsstudien
- ROI-Berechnungen
- Technische Planung
- Kostenoptimierung

## ⚠️ Wichtige Hinweise

### Annahmen und Limitationen
- Die Analyse basiert auf den eingegebenen Parametern
- Vereinfachte Batterie-Modelle
- Standardisierte Kostenkomponenten
- Für präzise Geschäftsentscheidungen sind detaillierte Studien erforderlich

### Datenqualität
- Hochwertige Lastprofil-Daten für genaue Ergebnisse
- Vollständige Jahresdaten (15-Minuten-Intervalle)
- Korrekte Zeitstempel-Formatierung

## 🔧 Technische Details

### Algorithmen
- **PV-Modell**: Vereinfachtes Solarstrahlungsmodell für Deutschland
- **Batterie-Optimierung**: Heuristische Algorithmen für Eigenverbrauch und Peak-Shaving
- **Kostenberechnung**: Detaillierte deutsche Energiekosten-Struktur

### Performance
- Optimiert für große Datensätze (Jahresdaten)
- Effiziente Pandas-Operationen
- Interaktive Plotly-Visualisierungen

## 📞 Support

Bei Fragen oder Problemen:
1. Überprüfen Sie die Datenqualität
2. Kontrollieren Sie die Parameter-Einstellungen
3. Konsultieren Sie die Dokumentation
4. Wenden Sie sich an das Entwicklungsteam

---

**Entwickelt für deutsche Industrieunternehmen** ⚡