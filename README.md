# Annosennustettavuusmalli

Tämä Python-paketti sisältää tekoälypohjaisen annosennustettavuusmallin kaarimoduloidun rintasyövän hoitoa varten. Malli on kehitetty Keski-Suomen sairaalan Novan tarpeisiin. Projekti on osa Sanni Sinisalon gradua. Gradun pohjana toimii Kuopion yliopistollisessa sairaalassa (KYS) kirjoitettu koodi, jonka on kirjoittanut Akseli Leino.

Mallin koodia muokataan tarvittaessa, jotta se saadaan toimimaan. Malli koulutetaan Novan potilasdatalla. Rakennemaskin muodostamisen koodi on kehitetty gradun aikana, koska Novalla ei ollut käytössä ohjelmistoa, jolla KYS:issä rakennemaski luotiin.



## Asennusohjeet

### Vaatimukset
- Python 3.13+
- Projekti käyttää `pyproject.toml` -tiedostoa riippuvuuksien hallintaan
- Suositellaan [uv](https://docs.astral.sh/uv/) paketinhallintatyökalun käyttöä
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

### Kansiorakenne
- Esikäsittelyn koodit hakevat tiedostot luokat.py luokkarakenteen kautta
- Mallin käyttää confgi.yaml tiedostoa
- Potilaat on jaettu hoitokohteen perusteella omiin kansioihin:
    - 'VN0ds' eli potilaalta on hoidettu vain vasen rinta
    - 'VN+ds' eli potilaalta on hoidettu vasen rinta sekä kainalon tai kaulan alueen imusolmukkeita
    - 'ON0ds' eli potilaalta on hoidettu vain oikea rinta
    - 'ON+ds' eli potilaalta on hoidettu oikea rinta sekä kainalon tai kaulan alueen imusolmukkeita
- Pääte ds viittaa downsamplaukseen
- Kansion alla oli potilaiden kansiot, jotka oli nimetty esim. Patient1_VN0, Patient2_VN0, jne. 
- Potilaskansioiden alla oli potilaan tiedostot omissa kansioissa.
- Tiedostopolut on kovakoodattu luokat.py ja config.ymal tiedostoihin.

### Aineiston esikäsittely

Aineiston esikäsittelyssä luodaan rakennemaskit, skaalataan RTDose tiedosto samaan resoluutioon kuin CT ja puolitetaan tiedostojen resoluutiot. 
Rakennemaskien luonti ja RTDose tiedostojen koiden muuttaminen eivät ole riippuvaisia toisistaan eli niiden ajamisen järjestyksellä ei ole väliä. Tiedostojen resoluutioiden pienentämisen eli downsamplaamisen tulee olla esikäsittelyn viimeinen vaihe.
Kaikki esikäsittelyn koodit lukevat DICOM eli .dcm muotoista dataa.

#### Rakennemaskin luominen (`src/annosennustettavuusmalli/preprocessing/maski2/`)
- Muodostetaan rakennemaski potilaan CT-kuvista ja RTStruct-tiedostosta.
- Paketit:
    - Pydicom: DICOM -tiedostojen lukeminen
    - Rt_utils RTStructBuilder: muunnetaan RTStruct-tiedoston sisältämät kontuurit vokseli-pohjaisiksi maskeiksi.
- Input: CT-kuvien pakka ja RTStruct tiedosto
- Output: rakennemaskin leikkeiden pakka, joka tallentuu potilaskansion alle omana kansiona nimellä 'maski'

#### RTDose tiedoston koon muuttaminen (`src/annosennustettavuusmalli/preprocessing/rtdose/`)
- Skaalataan RTDose samaan resoluutioon kuin potilaan CT-kuvat.
- Paketit:
    - Pydicom: DICOM -tiedostojen lukeminen.
    - SimpleITK: RTDosen käsittely ja resamplaus.
- Input: alkuperäinen RTDose ja CT-kuvat
- Output: skaalattu RTDose resoluutiossa 512x512, joka tallentuu potilaskansion alle omana kansiona nimellä 'dose'

#### Downsamplaaminen (`src/annosennustettavuusmalli/preprocessing/downsamplaus/`)
- Pienennetään tiedostojen resoluutio eli downsamplataan 512x512 --> 256x256
- Paketit:
    - Pydicom: DICOM -tiedostojen lukeminen.
    - Scipy.ndimage zoom: pienennetään resoluutio
- Input: CT-kuvat, skaalattu RTDose ja rakennemaski
- Output: downsalmatut CT-kuvat, skaalattu RTDose ja rakennemaski, jotka tallentuvat potilaskansion alle omina kansioina nimillä 'ct', 'doseds' ja 'maskids'

### Mallin kouluttaminen
- Mallin kouluttaminen tapahtuu malli1.py koodilla (`src/annosennustettavuusmalli/training/malli1/`)
- malli1.py hakee config.yaml tiedostosta tiedostopolut ja hyperparametrit
- Paketit:
    - PyTorch: mallin rakentaminen ja kouluttaminen
    - Torchio: datan käsittely ja datajonon muodostus
    - MLflow: metriikoiden kirjoittaminen ja tallentaminen koulutuksen aikana
- Input: kaikkien potilaiden downsamplatut CT-kuvat, skaalattu RTDose ja rakennemaski
- Output: yksi malli per epokki .pth-muodossa, joka tallentuu samaan kansioon, jossa koodi on, 'trained_models' nimen alle.

## Lisätietoa

Tarkempi dokumentointi ja selostus mallin jatkokäytöstä löytyvät projektista kirjoitetusta [Pro Gradu tutkielmasta](https://jyx.jyu.fi/jyx/Record/jyx_123456789_94998?sid=273647110).
Lisätietoa saa myös Akseli Leinon [artikkelista](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.17410), joka toimi pohjana tälle projektille ja gradulle.



## Lisenssi
Tämä projekti on sisältää koodia, joka on kirjoitettu alla listattujen henkilöiden toimesta MIT- lisenssillä
- Sanni Sinisalo
- Akseli Leino 

Akseli Leino on tehnyt osan repositorion koodeista, ja hänen tekemä koodi on ollut osalle pohjana, joita on muokattu tutkielmaa varten. Osa koodeista on myös tehty täysin Sanni Sinisalon toimesta. 
Erillisten koodien alussa on tieto, kuka koodin on alun perin tehnyt ja mahdollisesti kenen toimesta sitä on muokattu.

Katso lisätietoja `LICENSE`-tiedostosta.

## Tekijä
Sanni Sinisalo
