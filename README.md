# Annosennustetavuusmalli

Tämä python paketti sisältää tekoälypohjaisen annossennustettavuusmalln kaarimoduloidun rintasyövän hoitoa varten. Malli on kehitetty Keski-Suomen sairaalan Novan tarpeisiin. Projekti on osa Sanni Sinisalon gradua. Gradun pohjana toimii Kuopion yliopistollisessa sairaalassa (KYS) kirjoitettu koodi, jonka on kirjoittanut Akseli Leino väitöskirjansa yhteydessä.

Mallin koodia muokataan tarvittaessa, jotta se saadaan toimimaan. Malli koulutetaan Novan potilasdatalla. Rakennemaskin muodostamisen koodi on kehitetty Gradun aikana, koska Novalla ei ollut (vielä) käytössä ohjelmistoa, jolla KYS:issä rakennemaski luotiin.

## Asennusohjeet

### Vaatimukset
- Python 3.10+
- Projekti käyttää `pyproject.toml` -tiedostoa riippuvuuksien hallintaan
- Suositellaan [uv](https://docs.astral.sh/uv/) paketinhallintaatyökalun käyttöä
- Virtuaaliympäristöjen käyttö on vahvasti suositeltava (esim. `uv venv .venv`)

### Asennus


0. Aktivoi virtuaaliympäristö:
    - unix

```bash
source .venv/bin/activate
```
    - windows (cmd.exe)
```bash
.venv\Scripts\activate.bat
```

1. Kloonaa repositorio tai lataa tiedostot
2. Asenna paketti riippuvuuksineen:
```bash
pip install -e .
```

Tai kehitysriippuvuuksilla:
```bash
pip install -e ".[dev]"
```

## Käyttöohjeet

### Moduulit

#### Rakennemaskin luominen (`src/maski2/`)

**Käyttöesimerkki:**

#### Maskin koon pienentäminen (`src/?/`)

**Käyttöesimerkki:**

#### Muiden tarvittavien tiedostojen koon pienentäminen (`src/?/`)

**Käyttöesimerkki:**

#### DICOM-kuvatiedostojen visualisoiminen  (`examples/dicom_file/`)

**Käyttöesimerkki:**






## Lisenssi
Tämä projekti on lisensoitu MIT-lisenssillä. Katso lisätietoja `LICENSE`-tiedostosta.

## Tekijä
Sanni Sinisalo