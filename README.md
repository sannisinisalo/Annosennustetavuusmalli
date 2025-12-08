# Annosennustetavuusmalli

Annosennustettavuusmallin koodit

Tässä kaniossa on kaikki Pro Gradu -tutkielmassa käytetyt koodit. 

Gradussa tehdään tekoälypohjainen annossennustettavuusmalli kaarimoduloiduille rintasyöville Keski-Suomen sairaalan Novaan. Gradun pohjana toimii Kuopion yliopistollisessa sairaalassa (KYS) kirjoitettu koodi, jonka on kirjoittanut Akseli Leino väitöskirjansa yhteydessä. 

Mallin koodia muokataan tarvittaessa, jotta se saadaan toimimaan. Malli koulutetaan Novan potilasdatalla. Rakennemaskin muodostamisen koodi on pitänyt kehittää Gradun aikana, koska Novalla ei ollut (vielä) käytössä ohjelmistoa, jolla KYS:issä rakennemaski luotiin.

Tärkeimmät tiedostot:
- /share/pytorch_training.ipynb = varsinainen mallin koodi 
- /share/utils/generate_dataset.py = aineisto ladataan sanakirjamaiseen rakenteeseen 
- /share/utils/custom_transforms.py = luodaan transformeja, jotka muokkaavat dataa, kun se ladataan massamuistista keskusmuistiin
- MASKI1.py = rakennemaskin luominen 