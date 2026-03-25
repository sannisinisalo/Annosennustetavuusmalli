# Annosennustetavuusmalli

Tämä python paketti sisältää tekoälypohjaisen annossennustettavuusmalln kaarimoduloidun rintasyövän hoitoa varten. Malli on kehitetty Keski-Suomen sairaalan Novan tarpeisiin. Projekti on osa Sanni Sinisalon gradua. Gradun pohjana toimii Kuopion yliopistollisessa sairaalassa (KYS) kirjoitettu koodi, jonka on kirjoittanut Akseli Leino.

Mallin koodia muokataan tarvittaessa, jotta se saadaan toimimaan. Malli koulutetaan Novan potilasdatalla. Rakennemaskin muodostamisen koodi on kehitetty Gradun aikana, koska Novalla ei ollut käytössä ohjelmistoa, jolla KYS:issä rakennemaski luotiin.



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

### Aineiston esikäsittely

- Rakennemaskien luonti ja RTDose tiedostojen koiden muuttaminen eivät ole riippuvaisia toisitaan eli niiden ajamisen järjestyksellä ei ole väliä. 
- Tiedostojen resoluutioiden pientämisen eli downsamplaamisen tulisi olla esikästittleyn viimeinen vaihe.

#### Rakennemaskin luominen (`src/annosennustettavuusmalli/preprocessing/maski2/`)
- Rakennemaski muodostetaan potilaan CT-kuvista ja RTStruct-tiedostosta.
- Paketit:
    - Tiedostot ovat DICOM muodossa, ja niiden lukemiseen käyteään pydicom-pakettia.
    - rt_utils kirjaston RTStructBuilder:illa muunnetaan RTStruct-tiedoston sisältämät kontuurit voxelipohjaisiksi maskeiksi.
    - Maskien matemaattiseen käsittelyyn käytetään numpy kirjastoa.

#### RTDose tiedoston koon muuttaminen (`src/annosennustettavuusmalli/preprocessing/rtdose/`)
- Muutetaan RTDose samaan resoluutioon kuin potilaan CT-kuvat.
- Paketit:
    - DICOM:it luetaan pydicom:illa.
    - SimpleITK kirjastolla tehdään kuvan käsittely ja resamplaus.
    - numpy kirjastolla tehdään matriisilaskenta ja pikseliarvojen muuntaminen.

#### Downsamplaaminen (`src/annosennustettavuusmalli/preprocessing/downsamplaus/`)
- Pienennetään tiedostojen resoluutio eli downsamplataan 512x512 --> 256x256
- Koodilla DOwnsamplataan potilaiden CT-kuvat, maskit ja muokatut RTDose tiedostot
- Paketit:
    - DICOM:it luetaan pydicom:illa ja numpy:lla tehdään matriisilaskenta
    - Downsamplaus tapahtuu scipy.ndimage paketin zoom-komennolla

### Mallin kouluttaminen
- Mallin kouluttaminen tapahtuu malli1.py koodilla (`src/annosennustettavuusmalli/training/malli1/`)
- malli1.py hakee config.yaml tiedostosta tiedostopolut ja hyperparametrit
- Paketit:
    - Malli rakentaminen kouluttaminen tapahtuu PyTorch kirjastolla
    - torchio:lla käsitellään data ja muodostetaan datajono
    - mlflow kirjastoa käytetään koulutsvaiheessa lokien kirjoittamiseen ja tallentamiseen 
- Kouluttamisen vaiheet:
    1. Ladataan potilasdata ja jaetaan se eri datajoukoihin (koulutus, testaus ja validointi) 
    2. Luodaan UNet3+ verkko, optimointifunktio ja sakkofunktiot
    3. Aloitetaan kouluttaminen koulutussilmukassa ja määritetään miten oppimisnopeus muuttuu koulutusen edetessä 
        4. Yhden epokin silmukassa ennustetaan batchille annosjakauma
        5. Annosjakaumaa verrataan oikeaan tulokseen ja painotetaan tulos virheen perusteella
        6. Sakkofunktioiden avulla lasketaan gradientit painojen muutamikseksi
        7. Optimointifunktiolla päivitetään painot
    8. Epokin jälkeen mallia validoidaan validointijoukolla
    9. Tallennetaan paras malli sekä muut metriikat
    10. Testataan malli testausjoukolla 
    11. Visualisoidaan mallia

### Kansiorakenne
- Esikäsittelyn koodit hakevat tiedostot luokat.py luokkarakenteen kautta
- Mallin käyttää confgi.yaml tiedostoa
- Potilaat on jaettu hoitokohteen perusteella omiin kansioihin:
    - 'VN0ds' eli potilaalta on hoidettu vain vasen rinta
    - 'VN+ds' eli potilaalta on hoidettu vasen rinta sekä kainalon tai kaulan alueen imusolmukkeita
    - 'ON0ds' eli potilaalta on hoidettu vain oikea rinta
    - 'ON+ds' eli potilaalta on hoidettu oikea rinta sekä kainalon tai kaulan alueen imusolmukkeita
- Pääte ds viittaa downsamplaukseen eli esikäsiteltyihin tiedostoihin.
- Kansiorakenne on ollut seuraava: 
    - Peruspolku on kirjattu luokat.py tiedostoon BASE_DIR kohtaan ja config.yaml tiedosotoon data_paths kohtaan
    - Peruspolku vie kansioon, jonka alla on potilaskansiot nimillä Patient1_VN0, Patient2_VN0, Patient3_VN0, jne. 
    - Potilaskansioiden alla on erillisiä kansioita, oma kansio jokaiselle eri tyypin tiedostolle:
        - Alkuperäiset CT-kuvat kansiossa  'vanha ct'
        - RTStruct tiedosto kansiossa 'struct'
        - maski2.py koodilla luotu rakennemaski kansiossa 'maski'
        - rtdose.py koodilla muokattu RTDose tiedosto kansiossa 'dose'
        - Downsamplatut CT-kuvat kansiossa 'ct'
        - Downsamplattu maski kansiossa 'maskids'
        - Downsamplattu RTDose kansiossa 'doseds'



## Lisenssi
Tämä projekti on sisältää koodia, joka on kirjoitettu alla listattujen henkilöiden toimesta MIT- lisenssillä
- Sanni Sinisalo
- Akseli Leino 

Akseli Leino on tehnyt osan repositorion koodeista ja hänen tekemä koodi on ollut osalle pohjana, joita on muokattu tutkielmaa varten. 
Erillisten koodien alussa on tieto kuka koodin on alunperin tehnyt ja mahdollisesti kenen toimesta sitä on muokattu.

Katso lisätietoja `LICENSE`-tiedostosta.

## Tekijä
Sanni Sinisalo