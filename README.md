# SHMÚ radar pre e-ink (GitHub Pages)

Tento balík vytvorí verejnú GitHub Pages stránku s poslednou radarovou snímkou SHMÚ upravenou pre 3-farebný e-ink displej:

- **biela** = pozadie
- **čierna** = mapa, texty, obrysy
- **červená** = zrážky
- **šípka** = jednoduchý smer pohybu zrážok z posledných snímok

Repozitár generuje 2 obrázky:

- `latest.png` = čistý e-ink radar bez šípky
- `latest_arrow.png` = e-ink radar so šípkou

Predvolená web stránka `index.html` zobrazuje **`latest_arrow.png`**.

---

## Obsah balíka

- `index.html` – minimalistická stránka pre e-ink displej
- `latest.png` – posledný radar bez šípky
- `latest_arrow.png` – posledný radar so šípkou
- `latest_info.json` – čas a zdroj poslednej snímky
- `scripts/update_radar.py` – hlavný generátor radaru
- `.github/workflows/update-radar.yml` – GitHub Action spúšťaná automaticky
- `requirements.txt` – Python knižnice

---

## Ako to nasadiť

### 1. Vytvor nový GitHub repozitár
Odporúčaný názov napríklad:

`shmu-radar-eink`

### 2. Nahraj obsah tohto ZIP balíka do koreňa repozitára
Ak nahrávaš cez web GitHubu, ZIP najprv **rozbaľ** a nahraj súbory/foldre dovnútra repo.

### 3. Zapni GitHub Pages
V GitHub repozitári otvor:

**Settings → Pages**

Nastav:

- **Source**: `Deploy from a branch`
- **Branch**: `main` (alebo `master` podľa tvojho repo)
- **Folder**: `/ (root)`

Výsledná adresa bude vyzerať napríklad takto:

`https://TVOJ-UCET.github.io/shmu-radar-eink/`

### 4. Povoľ GitHub Actions
Po prvom pushi spusti workflow:

**Actions → Update SHMU radar for e-ink → Run workflow**

alebo počkaj na automatické spustenie.

### 5. Túto URL vlož do Živého Obrazu
Do Živého Obrazu daj URL tvojej GitHub Pages stránky.

---

## Ako to funguje

GitHub Action každých 5 minút:

1. nájde najnovší dostupný radar SHMÚ,
2. stiahne aj niekoľko starších snímok,
3. prefarbí výstup pre 3-farebný e-ink,
4. odhadne hlavný smer pohybu zrážok,
5. dokreslí jednoduchú šípku,
6. prepíše `latest.png`, `latest_arrow.png` a `index.html`,
7. commitne zmeny späť do repozitára.

---

## Poznámky

- GitHub Actions pri `cron` nemusia štartovať úplne na sekundu presne. Pre e-ink to však zvyčajne nevadí.
- Radar je **zrážkový radar**, nie klasická mapa oblačnosti.
- Smer šípky je orientačný – počíta sa z posledných radarových snímok.
- Ak chceš zobrazovať verziu **bez šípky**, otvor `index.html` a zmeň:

```html
<img src="latest_arrow.png?v=..." alt="SHMÚ radar">
```

na:

```html
<img src="latest.png?v=..." alt="SHMÚ radar">
```

---

## Odporúčanie pre e-ink

Ak Živý Obraz umožňuje interval obnovy, nastav ho približne na **5 až 10 minút**.

