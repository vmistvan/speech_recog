# speech_recog
English, Hungarian speech recognition

# English, Hungarian speech recognition
# Hang szöveggé alakítása angolul és magyarul

## Figyelem! Fontos telepítési tudnivalók az install.txt-ben!

### Változás: végigellenőrzöm a python fájlokat! Lokális használatra felkészítve. Saját tapasztalat szerint a tanítás nagyjából egy 12GB videómemóriával rendelkező cuda képes videokártyát igényel!
### Változás2: mivel nem akarok saját hangfájlokat felrakni, és te sem feltétlen akarsz stúdiót nyitni, itt van az angol és néhány másik nyelv adatkészlete: https://www.openslr.org/83/
###

Az angol és a magyar python szkript is megtalálható, ellenben sem a modellek, sem a hangfájlok nem kerültek be, méretgondok miatt.
A hangfájlok fejlécét mellékeltem mindkét nyelven, az angolt a nevek alapján meg lehet keresni a neten, honnan lehet letölteni, a magyar fájlok elkészítéséhez pedig mintát adnak.

A modellek alapjául szolgáló backbone-okat, és a modelleket szintén nem tartalmazza a könyvtár, mivel a könyvtárak a megadott név alapján letöltik és a szkript végén lévő sorral lementik helyi meghajtóra a betanított, testreszabott modellt.

A magyar példaszriptben kibővítettem a magyarázatokat. Mivel egy logikus munkamenet, hogy a Google Colabon megy a tanulás, és csak a predikció - azaz a munkavégzésre használt szövegfelismerés történik helyi gépen, igen praktikus az, ha dupla komment (##) kezdetű sorok kikerülnek a kódsorok közül, és kellően sok kódblokkra tagoljuk a jupyter noteszt, hogy például a pip installt csak a munkakörnyezet kialakításakor futtassuk le, de az egyes részeket külön-külön - dataset beállítása, modell töltése és mentése - meg tudjuk hívni, így ha nem sikerül például a tréningnél a memóriafoglalás, akkor nem kell a modellt újra letölteni.

Jó hangolást! :)
