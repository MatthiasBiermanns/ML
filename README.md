# ML

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
- Vorerste benutztung von 5 Städten verschiedener Kontinente:
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