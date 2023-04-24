# ML

## Selber ausführen

- Import der Libraries der requirements.txt
- Bei lokalem Hosting alternativ install.cmd oder install.sh ausführen

## Datenbeschaffung

- Dataset von [Mapillary](https://www.mapillary.com/dataset/places)
- Bestehend aus Dashcam-Bildern von 30 Städten auf 6 Kontinenten
- Zu jedem Bild folgende Informationen:
  - Sommer->Winter
  - Winter->Sommer
  - Nacht->Tag
  - Tag->Nacht
  - Tag oder Nacht
  - Blickrichtung
  - Stadt
  - Metadaten:
    - Aufnahmedatum
    - Längen- und Breitengrad
    - Kamerawinkel
    - Panorama

## Datenanalyse

### Aussortieren

- Vorerste benutztung von 6 Städten verschiedener Kontinente:
  - San Francisco
  - Tokio
  - Berlin
  - Kampala
  - Sao Paulo
  - Melbourne
- Bilder aussortiert mit folgenden Eigenschaften:
  - Panorama
  - Winter->Sommer
  - Nachtbilder

[Py-Script zur Datenbereinigung](./flatten_dataset.py)

[Script zum ausführen im Datenbaum](./run_flatten_script.sh)