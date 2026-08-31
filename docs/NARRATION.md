# Guide de narration longue (livres, méditations, podcasts)

Ce guide explique comment utiliser ce fork de VoxCPM2 pour narrer des **contenus
longs** en français : livres audio, scripts de méditation guidée, scripts de
podcast, etc.

## Réponse courte

**Oui, c'est fait pour ça** — mais deux réalités comptent :

1. **La vitesse dépend du matériel.** Sur **GPU CUDA**, c'est rapide et pratique. Sur
   **CPU seul**, c'est ~40× plus lent que le temps réel : utilisable pour des extraits
   courts, très long pour un livre entier.
2. **Le découpage est automatique et obligatoire.** Le moteur ne peut pas traiter plus
   de ~8 192 tokens d'un coup — au-delà il s'arrête sur une erreur « KV cache is full »
   (voir `src/voxcpm/model/voxcpm2.py`). L'app et le script découpent le texte en
   phrases pour rester bien en-dessous de cette limite, sans que tu aies à t'en soucier.

## La chaîne de production

Le texte brut ne devient pas un livre audio en une étape. Cinq étapes s'enchaînent,
chacune dans un module de `narration/` — testable et utilisable indépendamment :

| Étape | Module | Ce qu'elle fait |
|---|---|---|
| **0. Lecture** | `narration/epub.py` | Lit un `.epub` dans l'ordre du *spine* et en tire des chapitres titrés — un `.txt` se découpe lui sur les lignes `---` |
| **0 bis. Générique** | `narration/credits.py` | Ajoute au livre le générique de début et de fin qu'exigent les distributeurs, comme deux chapitres à part entière |
| **1. Préparation** | `narration/text_fr.py` ou `text_en.py` | Réécrit le texte tel qu'un narrateur le dirait : `1789` → « mille sept cent quatre-vingt-neuf », `M. Dupont` → « Monsieur Dupont », `XIVe siècle` → « quatorzième siècle », `14h30`, `1 250 €`, `3,5 %`… |
| **2. Découpage** | `narration/chunking.py` | Coupe en segments sous la limite du moteur, **sans jamais couper une phrase**, et décide la durée du silence après chaque segment selon la ponctuation |
| **3. Synthèse** | moteur VoxCPM2 | Même seed partout → voix identique du début à la fin |
| **4. Mastering** | `narration/audio.py` | Rogne les silences parasites, supprime les clics aux jointures, insère les pauses, normalise la sonie **une fois par chapitre** |
| **4 bis. Chaîne studio** | `narration/polish.py` | Passe-haut à 80 Hz, dé-esseur, compresseur et limiteur avant le calage du niveau — et mesure de la sonie en **LUFS** (EBU R128), la norme des plateformes de streaming |
| **5. Assemblage** | `narration/assemble.py` | Réunit les chapitres en un seul M4B/MP3 avec marqueurs de chapitres |
| **6. Livraison** | `narration/delivery.py` | Découpe, échantillonne et encode les fichiers qu'un distributeur accepte (MP3 192 kbps CBR, 44,1 kHz) |

Entre les étapes 2 et 3, un **cache par segment** (`narration/cache.py`) rend la
narration reprenable : voir plus bas.

## Vitesse : à quoi s'attendre

| Matériel | Vitesse (RTF) | 10 min d'audio | Livre de 3 h |
|---|---|---|---|
| RTX 4090 (CUDA) | ~0.30 (≈3× plus rapide que le réel) | ~3 min | **< 1 h** |
| Apple M4 Pro (Metal) | ~1.76 | ~18 min | ~5 h |
| **CPU seul (float32, cette machine)** | **~41** (41× plus lent) | **~7 h** | **plusieurs jours** |

> RTF = *Real-Time Factor* : temps de calcul ÷ durée audio produite. Plus c'est bas, mieux c'est.

### Activer le GPU (fortement recommandé pour les livres)

1. Installe une version CUDA de PyTorch dans le venv (voir https://pytorch.org — CUDA ≥ 12.0).
2. Lance avec `--device cuda` :
   ```
   .\.venv\Scripts\python.exe app.py --host 127.0.0.1 --port 8808 --device cuda --no-denoiser
   ```
   Le modèle demande ~8 Go de VRAM.

### Vitesse CPU : float32 par défaut (~1,5× plus rapide)

Sur CPU, le `bfloat16` du checkpoint est **émulé** et lent. Ce fork force donc
`float32` sur CPU (voir `pick_runtime_dtype` dans `src/voxcpm/model/utils.py`), ce qui
est **~1,5× plus rapide** (mesuré : RTF 60,9 → 41,1 sur la même phrase, soit −33 % de
temps) au prix d'un peu plus de RAM.

Pour revenir à l'ancien comportement bfloat16 : `set VOXCPM_CPU_DTYPE=bfloat16` avant de
lancer l'app (ou export sous bash).

## Trois façons de narrer

### 1. Onglet « 📚 Livre audio » — pour un livre depuis l'interface

1. Dans l'onglet **📚 Livre audio**, choisis une voix dans la liste **🎭 Voix
   prédéfinies** et clique **« 🔊 Écouter un aperçu »** pour la comparer aux autres.
   L'écoute est instantanée : les aperçus sont pré-générés dans
   `assets/voice_previews/`. C'est cette liste qui détermine la voix du livre.
   Pour une voix sur mesure, laisse-la sur **« Personnalisé / manuel »** et décris
   la voix dans l'onglet **🎙️ Studio**.
2. Charge ton `.txt` **ou ton `.epub`**, ou colle le texte. Un EPUB remplit aussi
   le titre et l'auteur, qui serviront de métadonnées au fichier assemblé.
3. Clique **« 🔍 Analyser sans générer »** : tu vois le nombre de chapitres, de
   segments, la durée estimée, et **le premier segment tel qu'il sera réellement lu**
   (après préparation du texte). C'est le moment de repérer un nombre ou une
   abréviation mal interprétés — avant d'engager des heures de calcul.
4. Clique **« 📖 Narrer le livre »**. Chaque chapitre terminé est écrit sur disque
   et devient écoutable immédiatement ; l'avancement s'affiche au fur et à mesure.
5. Clique **« 📦 Assembler le livre audio »** pour obtenir un fichier unique,
   et **« ✅ Vérifier la conformité de dépôt »** pour savoir, chapitre par
   chapitre, ce qu'un distributeur accepterait ou renverrait.

### 2. Script `narrate_book.py` — pour un livre entier en ligne de commande

Le plus robuste pour les longs contenus.

Sépare les chapitres de ton `.txt` par une ligne contenant seulement `---` :

```
Chapitre premier. ...

---

Chapitre deuxième. ...
```

Puis :
```
# Aperçu : découpage, durée estimée, et texte préparé — sans rien générer :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme" --dry-run

# Génération (une .wav par chapitre dans output/book_<nom>/) :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme"

# Génération + assemblage direct en M4B avec chapitres :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme" ^
    --assemble m4b --title "Mon Livre" --author "Edwin"

# Sur GPU :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "..." --device cuda
```

### 3. Onglet « 🎙️ Studio » — pour des extraits

Idéal pour tester des voix, générer une méditation ou un segment de podcast.
Deux options utiles dans les **Réglages avancés** :

- **Préparation du texte français** — applique l'étape 1 de la chaîne.
- **Mastering livre audio** — applique l'étape 4 (activé par défaut).

## Narrer dans une voix clonée

Le moteur sait cloner une voix depuis un enregistrement, mais seul l'onglet
Studio y avait accès — donc pour un extrait, jamais pour un livre. C'est branché
dans la narration longue :

```
scripts\narrate_book.py livre.epub --reference-audio ma_voix.wav ^
    --reference-text "le texte exact prononcé dans l'enregistrement"
```

`--reference-audio` remplace `--voice` et `--description` : un enregistrement est
une réponse complète à la question « quelle voix ? ».

**Une voix clonée peut devenir une voix préréglée**, listée dans le menu comme les
autres. Ajoute une entrée à `conf/preset_voices.json` avec un chemin **relatif** au
dépôt :

```json
{
  "name": "Edwin Dérivé (V1)",
  "reference": "assets/voices/edwin_derive_v1.wav",
  "reference_text": "le texte exact prononcé dans l'enregistrement",
  "seed": 1234, "cfg": 2.0, "diffusion_steps": 10, "lang": "fr"
}
```

Pas besoin de `description` : l'enregistrement *est* la description. Elle
s'utilise ensuite partout — menu de l'onglet, et `--voice "Edwin Dérivé (V1)"` en
ligne de commande, qui va chercher la référence et sa transcription tout seul.

**Les enregistrements ne sont jamais versionnés** (`assets/voices/` est dans le
`.gitignore`). Ce dépôt est public, et un échantillon de voix est précisément ce
qui permet à n'importe qui d'usurper celle de son propriétaire. L'entrée qui
pointe vers le fichier est versionnée ; le fichier, non. Un préréglage dont
l'enregistrement est absent le signale au démarrage et retombe sur sa
description, plutôt que d'échouer en pleine génération.

**Ce qu'il faut enregistrer** : un extrait **court et très propre** vaut mieux
qu'un long avec du souffle. Le débruiteur n'est pas chargé pendant la narration
(il bloque au téléchargement depuis cette machine), donc **ce qui est dans le
fichier est ce qui sera copié** — respiration, écho de la pièce, ventilateur
compris. Et c'est cette voix qui portera le livre pendant des heures : lis un
passage au rythme et sur le ton que tu veux entendre, pas une phrase neutre.

`--reference-text` est facultatif et vaut le coup : le moteur met les mots en
face de l'audio et clone plus fidèlement.

**La transcription doit couvrir tout l'enregistrement, et rien de plus.** C'est le
piège le plus coûteux du clonage, parce qu'il est silencieux : l'enregistrement
sonne parfaitement bien tout seul. Si le fichier contient de la parole que la
transcription ne mentionne pas — typiquement un clip coupé après la phrase
transcrite, qui mord sur la suivante — le moteur en déduit que le texte s'épuise
avant l'audio, et **termine trop tôt chaque segment du livre**. Mesuré ici : une
même voix, coupée à 8,8 s avec une transcription d'une phrase, sort tronquée ;
recoupée à 5,8 s là où finit cette phrase, elle sort saine et équivalente à la
référence complète de 19 s. Une référence courte ne coûte rien ; une référence
mal alignée coûte tout.

Le pré-vol le vérifie tout seul, avant même de charger le modèle — donc aussi en
`--dry-run`, et dans les deux onglets :

```
Voix        : clonée de edwin_derive_v1_phrase.wav (avec transcription)
              référence saine (4.3s de parole, 16 car/s)
```

Il compare le temps de **parole réelle** (silences de début, de fin et pauses
exclues) au nombre de caractères de la transcription, et signale les deux
décalages : `undertranscribed` (plus de parole que de texte) et `overtranscribed`
(des mots qui ne sont pas dans l'enregistrement), plus une référence sans
transcription, trop courte, trop longue ou saturée. **Il avertit, il ne refuse
pas** : les bornes sont des heuristiques calées sur peu d'enregistrements, et une
prise volontairement lente ne doit pas devenir inutilisable pour autant.

**Le cache suit la voix.** L'empreinte qui adresse un segment inclut un **hachage
du contenu** de l'enregistrement, pas son chemin. Deux conséquences voulues :
réenregistrer dans le même fichier ne ressert pas l'ancienne voix, et déplacer le
fichier ne jette pas le cache. Un même passage cloné et décrit ne peuvent pas se
confondre en cache.

## Narrer en anglais

`--language en` (ou le menu **Langue du livre** dans l'onglet) bascule deux choses :
la préparation du texte et la formulation du générique.

```
.\.venv\Scripts\python.exe scripts\narrate_book.py book.epub --language en ^
    --voice "..." --title "Around the Moon" --author "Jules Verne"
```

L'anglais a ses propres irrégularités, et `narration/text_en.py` les traite :

- **Une année se dit, elle ne se compte pas.** `1789` devient *seventeen
  eighty-nine*, `1905` devient *nineteen oh five*, `2005` devient *two thousand
  five*. Ce qui distingue une année d'une quantité est le séparateur de milliers :
  `1,789 men` se compte, `in 1789` se dit. `--no-text-prep` ou `read_years=False`
  désactive.
- **Les suffixes ordinaux** dépendent des deux derniers chiffres : `21st` →
  *twenty-first*, mais `11th` → *eleventh* et non *eleven-first*.
- **Le point d'un titre n'est pas une fin de phrase.** `Mr. Dupont` devient
  *Mister Dupont* — laisser le point inventerait un point final au milieu de la
  phrase, et le découpage la couperait là.
- Monnaies avec leurs centimes (*and fifty cents*), pourcentages, heures, chiffres
  romains après un mot déclencheur (`chapter XIV`).

Le générique suit :

> « Around the Moon », by Jules Verne. Narrated by a synthetic voice.
>
> You have been listening to « Around the Moon », by Jules Verne… Recorded in
> twenty twenty-six. This text is in the public domain.

**Ce qui reste français** : les voix préréglées sont décrites en français et
sonneront avec un accent. Pour de l'anglais natif, décris une voix anglaise dans
l'onglet Studio, ou clone une voix anglophone.

## Partir d'un EPUB

Un `.epub` se charge directement, dans l'onglet **📚 Livre audio** comme en ligne de
commande :

```
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.epub --voice "Narrateur profond & calme" --dry-run
```

Ce qui en est tiré :

- **L'ordre de lecture vient du *spine***, jamais du nom des fichiers — sinon le
  chapitre 10 passerait avant le 2.
- **Les titres viennent de la table des matières du livre** (nav EPUB 3 ou NCX
  EPUB 2), à défaut du premier titre du document. Ce sont eux qui deviennent les
  marqueurs de chapitres du M4B.
- **Les fichiers contenant plusieurs chapitres sont recoupés sur leurs titres.**
  Beaucoup de livres — ceux du projet Gutenberg notamment — sont découpés en
  fichiers de taille fixe : sans ce recoupage, *Autour de la Lune* donnerait
  6 énormes chapitres au lieu de ses 25 vrais. `--no-epub-split` désactive.
- **Les pages de garde sont écartées** en dessous de `--epub-min-chars`
  caractères (140 par défaut) — une couverture n'est pas un chapitre.
- **L'appareil éditorial est retiré** : l'en-tête et la licence du projet
  Gutenberg (17 000 caractères d'anglais juridique, soit ~20 min de narration en
  fin de livre) sont coupés sur les marqueurs officiels `*** START OF … ***` et
  `*** END OF … ***`, et une table des matières présente dans le corps du livre
  est écartée quand la majorité de ses lignes sont des titres de chapitres.
  **Rien n'est retiré en silence** : chaque suppression est listée dans le plan
  et sous le bouton de chargement. `--keep-boilerplate` désactive.
- **Un EPUB protégé par DRM est refusé** avec un message clair, plutôt que narré
  en bruit binaire.

Deux limites à connaître :

- Un livre **entièrement contenu dans un seul fichier** reste un seul chapitre :
  avec un seul document, rien ne permet de distinguer un titre de livre au-dessus
  de ses chapitres d'un chapitre au-dessus de ses scènes. Insère des `---` pour
  découper toi-même.
- Un livre **scanné** (images seules, sans texte) est refusé : il n'y a rien à
  lire. Il faut passer par une reconnaissance de caractères d'abord.

Le texte importé reste **modifiable dans la zone de texte** avant génération : ce
qui est narré est ce que tu y vois, `---` compris.

## Reprise après interruption

C'est le point critique sur CPU, où un chapitre prend des heures.

- **Par chapitre** : un chapitre dont le `.wav` existe déjà est ignoré. `--force` le
  régénère.
- **Par segment** : chaque segment généré est mis en cache dans
  `output/book_<nom>/.cache/`, indexé par le **contenu** (texte + description + seed +
  CFG + étapes + modèle). Si tu relances après une interruption, seuls les segments
  manquants sont calculés — pas tout le chapitre.

Conséquences pratiques :

- Corriger une coquille dans un paragraphe n'invalide que les segments de ce
  paragraphe. Le reste du livre est réutilisé tel quel.
- Changer de voix (ou de seed) invalide tout, ce qui est correct : c'est un autre
  narrateur.
- Le cache occupe de la place. `--no-cache` le désactive ; supprimer le dossier
  `.cache/` est sans risque une fois le livre terminé.

## Qualité audio : la norme ACX

Les plateformes de livres audio vérifient trois choses. Le mastering vise ces valeurs,
et chaque chapitre est mesuré à l'écriture :

| Mesure | Cible | Pourquoi |
|---|---|---|
| RMS (sonie) | entre **-23 et -18 dBFS** (défaut : -20) | volume homogène d'un chapitre à l'autre |
| Crête | **≤ -3 dBFS** | marge avant saturation |
| Bruit de fond | **≤ -60 dBFS** | silences réellement silencieux |

Le RMS est mesuré **sur la parole seule** : les silences entre phrases sont exclus du
calcul. Sans cela, un chapitre aux pauses généreuses mesurerait plusieurs dB trop bas,
et le corriger pousserait la parole au-dessus du plafond de crête.

Vérifier un livre assemblé :
```
.\.venv\Scripts\python.exe scripts\assemble_audiobook.py output\book_mon_livre --check
```

## Contrôle qualité automatique

Le moteur échoue rarement, mais il échoue **localement** : un segment sur quelques
dizaines revient coupé au milieu d'un mot, muet, ou parti en boucle bien après la fin
de son texte. Sur GPU on régénère le chapitre. Sur CPU un chapitre représente des
heures, donc la seule réparation abordable porte sur le segment fautif — encore
faut-il le trouver. Écouter quatre heures de narration pour repérer onze secondes
n'est pas une méthode.

Chaque segment généré est donc confronté **au texte qui l'a produit**. C'est ce
couplage qui rend la détection possible : l'audio seul ne peut pas dire si deux
secondes constituent une phrase complète, mais deux secondes pour deux cents
caractères sont une troncature, sans ambiguïté.

| Code | Gravité | Ce qui est détecté |
|---|---|---|
| `silent` | fatal | rien n'est revenu |
| `truncated` | fatal | beaucoup moins d'audio que le texte ne l'implique |
| `runaway` | fatal | beaucoup plus — le moteur a bouclé ou divagué |
| `clipped` | fatal | échantillons saturés, irrécupérables au mastering |
| `gap` | suspect | long silence interne, signature d'une proposition sautée |
| `abrupt_end` | suspect | s'arrête au niveau de parole, sans décroissance |
| `looped` | suspect | l'enveloppe de niveau se répète |

Seuls les défauts **fatals** déclenchent une régénération, avec une seed dérivée de
la seed d'origine — donc reproductible : le même livre relancé de zéro répare le même
segment de la même façon. Le meilleur essai est conservé, jamais le dernier : un
second tirage peut être pire que le premier, et garder silencieusement le pire
rendrait la réparation nuisible.

```bash
# Comportement par défaut : un nouvel essai par segment fatalement défectueux
python scripts/narrate_book.py livre.txt --voice "..."

# Plus insistant sur un livre qu'on ne veut pas réécouter segment par segment
python scripts/narrate_book.py livre.txt --voice "..." --qc-retries 3

# Signaler sans régénérer (utile pour auditer un livre déjà produit)
python scripts/narrate_book.py livre.txt --voice "..." --qc-retries 0

# Sortie en code d'erreur s'il reste un défaut — pour un enchaînement automatisé
python scripts/narrate_book.py livre.txt --voice "..." --qc-strict
```

Le bilan est écrit dans `output/book_<nom>/qc_report.json` : un segment par entrée,
avec sa durée, son débit et ses défauts. `--no-qc` désactive tout.

**Sur les seuils de débit.** Ce sont eux qui portent la détection de troncature, et
ils viennent de la mesure, pas d'une estimation : sur la même phrase de 81
caractères, les quatorze voix préréglées produisent entre **15,8 et 24,1 caractères
par seconde** (médiane 20,2), soit plus de 50 % d'écart entre la plus lente et la plus
rapide. Les bornes (35 et 6 car/s) sont donc placées largement en dehors de cette
plage — choisir une voix rapide ne doit jamais ressembler à un défaut — tout en
restant franchies par une troncature qui perdrait la moitié d'une phrase.

Cette plage n'a **pas bougé** quand le jeu de voix est passé de sept à quatorze :
mêmes 15,8 et 24,1 aux deux extrémités. C'est ce qui lui donne du crédit — doubler
l'échantillon n'a déplacé aucune borne. Un test verrouille chacune des valeurs
mesurées, pour qu'un réglage ultérieur ne puisse pas les faire dériver sans alerte.

## Générique de début et de fin

Un livre audio n'est pas seulement le livre lu. **Tous les distributeurs** — ACX
et Audible, et derrière eux Amazon, Apple Books, Kobo, Google Play — exigent que
l'enregistrement s'annonce : le premier fichier ouvre sur le titre, l'auteur et
le narrateur, le dernier les nomme à nouveau. Un dépôt sans générique est refusé
au contrôle qualité avant même qu'on écoute une ligne du texte.

Le générique est donc **ajouté par défaut**, comme deux chapitres à part entière :

```
chapitre_001.wav   Générique de début
chapitre_002.wav   … le livre …
chapitre_027.wav   Générique de fin
```

En faire des chapitres est délibéré : ils passent par la même préparation du
texte, la **même voix et la même graine**, le même mastering et le même cache que
le livre — ils sonnent donc comme le narrateur, pas comme une annonce rapportée.

```
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.epub --voice "..." ^
    --title "Autour de la Lune" --author "Jules Verne" --year 2026 --public-domain
```

| Option | Effet |
|---|---|
| `--narrator "Nom"` | Narrateur humain cité au générique |
| `--publisher "Studio"` | Production créditée à la fin |
| `--year 2026` | Année créditée à la fin |
| `--public-domain` | Ajoute « Texte du domaine public » |
| `--no-credits` | N'ajoute aucun générique |

Ce que ça donne :

> « Autour de la Lune », de Jules Verne.
> Lu par une voix de synthèse.

> Vous venez d'écouter « Autour de la Lune », de Jules Verne, lu par une voix de
> synthèse. Enregistrement réalisé en deux mille vingt-six. Texte du domaine public.

**La voix de synthèse est déclarée** quand aucun narrateur humain n'est nommé.
Ce n'est pas une précaution ajoutée par prudence : Audible distribue ces titres
via un programme séparé et les étiquette comme tels. Faire passer une lecture
machine pour une performance humaine est ce qui fait fermer un compte. Nommer un
narrateur avec `--narrator` remplace la mention.

Le plan avant génération dit ce qu'il manque pour une distribution :

```
Générique   : début et fin ajoutés — manque encore l'auteur
```

## La chaîne studio : ce que le distributeur ne contrôle pas

Les niveaux ACX disent qu'un chapitre est *acceptable*. Ils ne disent rien de ce
qu'on entend. Quatre traitements tournent donc sur chaque chapitre assemblé,
**avant** le calage du niveau (`--no-polish`, ou la case « Chaîne studio ») :

| Étage | Pourquoi |
|---|---|
| **Passe-haut à 80 Hz** | Sous 80 Hz il n'y a rien d'une voix, mais du grondement qui mange de la marge et fatigue au casque |
| **Dé-esseur** | Les sifflantes sont le premier défaut qui trahit une voix de synthèse en français |
| **Compresseur** | Un livre audio s'écoute en marchant, en voiture : l'écart entre une phrase murmurée et une phrase appuyée doit se resserrer |
| **Limiteur** | Sans lui, le compresseur **dégrade** le résultat — voir plus bas |

**Le limiteur n'était pas prévu, la mesure l'a imposé.** Le compresseur seul baisse
les tenues sans toucher les crêtes courtes : le facteur de crête *monte* (17,7 →
19,0 dB mesuré sur un vrai chapitre), et le plafond de −3 dBFS oblige alors la
normalisation à reculer, faisant perdre 1,3 LU à tout le chapitre pour quelques
échantillons. Tenir les crêtes est ce qui permet au reste de sonner à son niveau.

Mesuré sur les trois chapitres d'un livre réellement narré :

| | crête/RMS | bande sifflante | sonie |
|---|---|---|---|
| Générique de début | 17,7 → **14,4 dB** | −13,9 → **−17,8 dB** | −20,98 → −20,52 LUFS |
| Le texte | 14,4 → 14,4 dB | −14,0 → **−15,3 dB** | −20,27 → −19,64 LUFS |
| Générique de fin | 14,7 → **14,0 dB** | −23,4 → **−25,9 dB** | −20,96 → −20,94 LUFS |

### La sonie en LUFS

ACX raisonne en RMS ; **Spotify, Apple Books et YouTube normalisent en LUFS**
(ITU-R BS.1770 / EBU R128), qui pondère le spectre comme l'oreille. Un chapitre
parfaitement calé à −20 dBFS RMS peut arriver trop fort ou trop faible chez eux, et
rien dans le rapport ACX ne l'aurait dit. La mesure est donc ajoutée au rapport —
**reportée, jamais éliminatoire** : aucune norme ne fixe une cible unique, et
inventer un seuil que personne n'exige serait pire que donner le chiffre.

L'implémentation a été **confrontée à ffmpeg** (`-af ebur128`) sur de vrais
chapitres : −20,98 contre −20,9 ; −20,27 contre −20,2 ; −20,96 contre −20,9.

## Forme des fichiers : ce qu'ACX vérifie en plus du niveau

Un chapitre parfaitement calibré en sonie est quand même refusé s'il **commence
sur la première syllabe**. La norme porte aussi sur la forme du fichier :

| Contrôle | Norme ACX | Où c'est appliqué |
|---|---|---|
| Sonie (RMS) | −23 à −18 dBFS | mastering, une passe par chapitre |
| Crête | ≤ −3 dBFS | mastering |
| Bruit de fond | ≤ −60 dBFS | mesuré, reporté |
| **Silence en tête** | **0,5 à 1 s** | 0,75 s posé par le mastering |
| **Silence en queue** | **1 à 5 s** | 2 s posées par le mastering |
| **Durée d'un fichier** | **≤ 120 min** | mesurée, reportée |

`narration/audio.py` mesure les six et `acx_report()` dit lesquels passent. Les
valeurs par défaut visent le **milieu** de chaque fenêtre, pas son bord : un
chapitre reste conforme même si le rognage laisse un peu de silence à lui.

## Réparer un segment sans renarrer le livre

Un livre, c'est des heures de calcul. Quand **une** phrase sort tronquée ou
bafouillée, tout régénérer — même seulement son chapitre — est absurde : le reste
était bon, et le cache le contient encore.

```
.\.venv\Scripts\python.exe scripts\repair_segment.py output\book_mon_livre --list
.\.venv\Scripts\python.exe scripts\repair_segment.py output\book_mon_livre --segment ch003/seg012
```

`--list` lit le cache et **ne charge pas le modèle** : savoir ce qui cloche ne doit
pas coûter une minute d'attente. `--all-fatal` répare d'un coup tout ce qui est
fatalement défectueux. Une nouvelle prise moins bonne que l'ancienne est **refusée**
et signalée — relancer la commande en tire une autre.

Cela repose sur le `plan.json` écrit à côté des chapitres, qui mémorise quelle
entrée du cache contient quelle phrase. Un livre narré avant que ce fichier existe
se rattrape en relançant `narrate_book.py` avec les mêmes arguments : tout vient du
cache, donc **c'est affaire de secondes** (mesuré : 23 s sur un livre déjà narré).

## Assemblage en un fichier unique

```
# M4B avec marqueurs de chapitres (nécessite ffmpeg) :
.\.venv\Scripts\python.exe scripts\assemble_audiobook.py output\book_mon_livre ^
    --title "Mon Livre" --author "Edwin" --format m4b
```

**Sans ffmpeg, rien n'est perdu** : le script produit quand même le WAV complet et le
fichier de marqueurs, puis affiche la commande exacte à lancer une fois ffmpeg installé.
Les heures de synthèse ne dépendent jamais d'un encodeur manquant.

Pour l'installer sous Windows, sans droits administrateur :

```
winget install --id Gyan.FFmpeg -e --scope user
```

Il faut ensuite **rouvrir le terminal** pour que le `PATH` soit pris en compte.

**La couverture du livre est reprise automatiquement** quand la source est un
`.epub` : elle est extraite à côté des chapitres (`couverture.jpg`) et intégrée
au M4B. `--cover mon_image.jpg` impose la tienne, `--no-cover` n'en met aucune.

**Le débit** vaut par défaut **64 kbps AAC** pour un M4B et **128 kbps** pour un MP3.
Ce n'est pas un compromis : 64k AAC mono est à peu près ce qu'Audible diffuse
lui-même pour un livre audio fini, et la parole ne gagne quasiment rien au-dessus.
`--bitrate 192k` (ou le menu **Débit** dans l'onglet) le monte, pour une copie
d'archive ou un fichier qui sera ré-encodé ensuite.

Les titres de chapitres viennent, dans l'ordre : de `--titles`, puis d'un fichier
`titles.txt` à côté des WAV (écrit automatiquement par `narrate_book.py` à partir de la
première ligne de chaque chapitre), puis des noms de fichiers.

## Déposer chez un distributeur (ACX, Audible, Amazon…)

Le M4B est ce qu'on écoute. **Ce n'est pas ce qu'on dépose.** ACX — et les
plateformes qui s'alignent dessus — prend **un fichier par chapitre**, encodé à
une spécification fixe, plus un extrait commercial, et refuse l'ensemble pour des
détails qui n'ont rien à voir avec la qualité de la narration.

```
.\.venv\Scripts\python.exe scripts\export_acx.py output\book_mon_livre
```

Ou d'un seul trait depuis le texte : ajoute `--export-acx` à `narrate_book.py`.
Le contrôle seul, sans rien produire, est aussi dans l'onglet **📚 Livre audio**,
bouton **« ✅ Vérifier la conformité de dépôt »**.

```
output/book_mon_livre/          ->   output/book_mon_livre/acx/
  chapitre_001.wav                     001 - Generique de debut.mp3
  chapitre_002.wav                     002 - Chapitre premier.mp3
  ...                                  ...
  titles.txt                           extrait_commercial.mp3
                                       rapport_acx.json
```

Ce que le script fait :

1. **Contrôle chaque chapitre** sur toute la spécification — sonie, crête, bruit
   de fond, silence aux deux bouts, durée, taille — et dit lesquels reviendraient,
   avec la raison en clair.
2. **Découpe ce qui est trop long**, dans une pause et non au milieu d'un mot.
   La limite est calculée, pas supposée : à 192 kbps constant, les 120 minutes et
   les 170 Mo se croisent, et c'est le plus contraignant des deux qui décide
   (~118 min).
3. **Extrait un extrait commercial** de 1 à 5 min du premier vrai chapitre —
   jamais du générique : un acheteur ne se décide pas en entendant le titre.
4. **Encode en MP3 192 kbps CBR, 44,1 kHz, mono**, ce qui demande ffmpeg.

**Sans ffmpeg, rien n'est perdu** : les WAV sont écrits, les commandes
d'encodage sont listées dans `acx/encoder.txt`, et l'encodage peut se faire plus
tard ou sur une autre machine. Des heures de synthèse ne doivent pas dépendre
d'un binaire manquant.

| Option | Effet |
|---|---|
| `--check` | Contrôle et n'écrit rien |
| `--sample-seconds 240` | Longueur de l'extrait (60 à 300 s) |
| `--sample-start 60` | Démarre l'extrait plus loin dans le chapitre |
| `--sample-chapter 4` | Choisit le chapitre à échantillonner |
| `--no-sample` | Pas d'extrait |
| `--keep-wav` | Garde les WAV intermédiaires |

**Vérifié pour de vrai** sur un livre narré de bout en bout : les quatre fichiers
produits sortent en `mp3`, `44100 Hz`, `1 canal`, **débit constant de 192 000 bps**
exactement, sans en-tête Xing — c'est-à-dire du CBR, et non du VBR déguisé.

Le script **sort en code d'erreur** s'il reste un fichier hors norme, ce qui le
rend utilisable dans un enchaînement automatisé.

À savoir : le **44,1 kHz est une exigence de format**, pas un gain de qualité —
la synthèse ne produit pas cette fréquence, le rééchantillonnage se fait à
l'encodage. Et la limite de taille est lue dans son sens le plus strict
(170 × 10⁶ octets) : être sous une limite qui s'avère plus large coûte un
fichier de plus, être au-dessus coûte un dépôt refusé.

## Prononciation : lexique personnalisé

`conf/pronunciation_fr.json` associe ce qui est écrit à ce qui doit être prononcé.
C'est l'outil pour les noms propres d'un roman, les sigles et les mots étrangers :

```json
{
  "SNCF": "S N C F",
  "Nietzsche": "Nitche",
  "Aurélien Krähenbühl": "Aurélien Krènebul"
}
```

Le remplacement est insensible à la casse et ne s'applique qu'à des mots entiers.
Les clés commençant par `_` sont des commentaires.

### Les homographes : quand le même mot se dit de deux façons

`« il est »` et `« à l'est »` s'écrivent pareil et ne se prononcent pas pareil.
Une entrée de lexique qui vise le mot seul casse forcément l'un des deux, donc
une valeur peut être un **objet à contexte** :

```json
"est": { "prononcer": "èsste", "après": "à l'|dans l'|vers l'|l'" },
"plus": { "prononcer": "pluss", "avant": "de|que|d'" }
```

`après` et `avant` sont des expressions régulières ; seul ce qui suit le contexte
est remplacé, le contexte lui-même est conservé. Résultat :

```
La SNCF est à l'est. Il est tard.
→ La S N C F est à l'èsste. Il est tard.
```

`conf/pronunciation_fr.json` contient une **série de modèles désactivés** pour les
pièges classiques du français — *est, fils, couvent, portions, violent, content,
négligent, plus*. Ils sont désactivés à dessein : **écoutez d'abord**. Si la voix
lit déjà correctement « le couvent », corriger ne peut que dégrader. Quand vous en
repérez un faux, retirez le préfixe `_` de la ligne et ajustez l'orthographe
phonétique à l'oreille.

## Ce que la préparation du texte corrige (et ses limites)

Sont gérés : nombres cardinaux et ordinaux (`1er`, `2e`, `1re`), décimales, sommes en
euros/dollars/livres, pourcentages, heures (`14h30`), abréviations (`M.`, `Mme`, `Dr`,
`Me`, `St`, `etc.`, `av. J.-C.`, `n°`, `p. 42`), chiffres romains, tirets de dialogue,
guillemets, et le balisage Markdown.

Les chiffres romains ne sont développés que dans des contextes **non ambigus** :
après un mot déclencheur (`chapitre XIV`, `tome III`), en forme ordinale (`XIXe`), ou
seuls sur une ligne de titre. C'est délibéré : « Le » est L + e, « Ce » est C + e — les
développer partout ferait lire « Le manuscrit » comme « cinquantième manuscrit ».

Désactiver globalement : `--no-text-prep` (script) ou décocher la case (interface).

## Réglages recommandés par usage

| Usage | Voix suggérée | Réglages |
|---|---|---|
| **Livre audio (fiction)** | *Narrateur profond & calme* / *Narratrice douce & naturelle* | défauts (CFG 2.0, 10 étapes) |
| **Documentaire / non-fiction** | *Narrateur documentaire velouté* / *Narrateur moderne & professionnel* | défauts |
| **Méditation guidée** | *Méditation guidée (grave & lente)* | `--pause-sentence 0.8 --pause-paragraph 1.6` |
| **Podcast** | *Conteur jeune & dynamique* / *Narratrice chaleureuse & conversationnelle* | `--pause-paragraph 0.6` (rythme plus soutenu) |

## Cohérence de la voix sur un long texte

La voix reste identique d'un segment à l'autre parce que **le même seed est réutilisé
pour tous les segments** (une paire description + seed régénère exactement la même voix).
C'est ce qui garantit un narrateur constant sur tout un livre.

L'option expérimentale `--continuity` va plus loin : chaque segment est enchaîné à
partir du précédent (continuation par *prompt-cache*) pour des jointures encore plus
fluides. Le mécanisme fonctionne mais il est **beaucoup plus lent** — à régler sur GPU.

## Limites à connaître

- **Longueur par appel** : ~8 192 tokens max (découpage automatique, donc transparent).
- **Durée par segment** : le moteur vise ~6× la longueur du texte et s'arrête tout seul ;
  garde des segments de taille raisonnable (défaut 300 caractères).
- **Sortie** : les chapitres sont écrits en WAV 16 bits, un fichier par chapitre. Un
  livre entier n'est jamais chargé en mémoire — ni à la génération, ni à l'assemblage
  (qui écrit en flux).
