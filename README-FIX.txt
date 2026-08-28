OPRAVA SHMÚ RADAR E-INK

1. Nahraď v repozitári pôvodný index.html týmto novým index.html.
2. Nový index už nečíta latest_arrow.png z GitHub Pages deployu,
   ale priamo z vetvy main cez raw.githubusercontent.com.
3. Preto sa po každom GitHub Action update zobrazí najnovší obrázok
   bez potreby čakať na nový GitHub Pages deploy.

Priama URL obrázka pre Živý Obraz:
https://raw.githubusercontent.com/adykub-3dee/shmuradarsk/main/latest_arrow.png

Ak Živý Obraz akceptuje priamo obrázkovú URL, môžeš použiť túto adresu
a GitHub Pages vôbec nepotrebuješ.
