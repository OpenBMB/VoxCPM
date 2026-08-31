---
name: audiobook-text-prep
description: "Préparer un manuscrit pour la narration audio, phonétique comprise : conversion du texte, nettoyage de ce qui ne se lit pas à voix haute, abréviations, lexique de prononciation, contrôle après narration. Déclencheurs : 'préparer un livre pour narration', 'texte pour livre audio', 'prononciation', 'phonétique', 'lexique', 'mot mal prononcé', 'préparer le manuscrit', 'narration TTS'. À utiliser AVANT de narrer, et APRÈS pour auditer. NE PAS utiliser pour : le mastering audio, la conformité ACX, l'assemblage M4B — voir scripts/export_acx.py."
license: Proprietary
---

# Préparer un manuscrit pour la narration

Un manuscrit est écrit pour être **vu**. La narration le donne à **entendre**.
Tout ce qui vit dans cet écart — un ISBN, un astérisque, une table des
matières, une abréviation jamais développée — se lit très bien des yeux et se
dit très mal à voix haute.

Ce skill couvre la préparation du texte et la prononciation. Il ne couvre ni le
mastering ni la conformité au dépôt.

## Pourquoi c'est le maillon décisif

Un défaut de préparation ne se voit pas au moment où on le crée. Il se découvre
trois heures de GPU plus tard, dans un livre terminé, et il se répète à
l'identique dans tous les livres suivants. Sur un catalogue, une erreur de
préparation ne coûte pas un livre : elle coûte le catalogue.

Chaque piège listé ici a été payé une fois. Aucun n'est théorique.

## La chaîne

```
manuscrit.md
   │  scripts/prepare_manuscript.py        ← retire ce qui ne se dit pas
   ▼
livre.txt  ────────────────────────────────┐
   │  scripts/scan_risky_words.py          │  ← liste les mots à risque
   │  scripts/narrate_book.py --dry-run    │  ← pré-vol, sans GPU
   ▼                                       │
narration                                  │
   │  scripts/audit_pronunciation.py       │  ← ce qui a été mal dit (ASR)
   ▼                                       │
candidats ─── scripts/try_pronunciation.py │  ← faire entendre les variantes
   │                                       │
   ▼                                       │
conf/pronunciation_fr.json  ───────────────┘  ← le lexique s'enrichit
```

Le lexique est le capital du catalogue. Chaque livre corrigé rend le suivant
meilleur, et sur trois cents livres cet effet cumulé compte plus que n'importe
quel réglage.

## 1. Convertir — `prepare_manuscript.py`

```bash
python scripts/prepare_manuscript.py manuscrit_complet.md -o livre.txt --report
```

Il retire ce qui ne se lit pas et **dit toujours ce qu'il a retiré**. Un texte
supprimé sans le signaler est un texte perdu sans le savoir.

Quatre pièges qu'il traite, tous rencontrés :

**`---` veut dire deux choses.** Filet horizontal dans le manuscrit, séparateur
de chapitres pour le narrateur. Une conversion naïve découpe le livre à la page
de copyright.

**Le liminaire n'est pas de la narration.** ISBN, copyright, table des matières,
adresse web : lus à voix haute, ils ouvrent le livre sur son propre code-barres.
La dédicace et l'avertissement médical, eux, se gardent.

**Les crochets ne sont pas une seule chose.** Sur un corpus de vingt livres :
993 marqueurs, dont `[PAUSE]` 969 fois. `[PAUSE]` devient un vrai silence ;
`[rire]`, `[silence prolongé]` sont des didascalies et disparaissent ; mais
`[nom du département]` ou `[ton mari / ta femme]` **sont la phrase** — un blanc
que le lecteur remplit — et seuls leurs crochets partent.

**Un tableau ne se lit pas.** À voix haute, c'est une suite de barres verticales.

## 2. Repérer les mots à risque — `scan_risky_words.py`

```bash
python scripts/scan_risky_words.py queue/*.txt --top 30 --min 10
```

Classe sigles, noms propres et mots étrangers **par fréquence**. Un sigle lu
999 fois coûte 999 fautes ; un nom propre lu deux fois en coûte deux. À temps
d'écoute égal, on corrige le premier.

Il ne corrige rien. Il dit où regarder.

## 3. Les abréviations maison — le cas qu'aucun outil ne devine

`HFD` revenait 160 fois dans un livre, **jamais développé** : le texte l'emploie
dès le chapitre 1. Un lecteur remonte quelques pages ou devine ; un auditeur
entend « ache-èf-dé » cent soixante fois sans jamais savoir de quoi on parle.

Ce n'est pas un problème de prononciation, c'est un problème de sens, et il
n'existe qu'à l'oral. **Le chercher fait partie de la préparation** : toute
abréviation fréquente doit être développée, ou introduite une première fois.

Les lexiques s'empilent, donc un livre déclare les siens sans perdre les
communs :

```bash
--lexicon conf/pronunciation_fr.json --lexicon conf/lexique_mon_livre.json
```

Attention aux accords. `HFD` → « dépression de haut fonctionnement » demandait
trois règles : `les HFD` au pluriel (les adjectifs qui suivent étaient déjà
accordés), `mécanisme HFD` avec l'article, et le reste au singulier. Le lexique
applique la clé la plus longue d'abord, ce qui les fait cohabiter.

## 4. Le pré-vol — gratuit, et il attrape ce qui coûterait trois heures

```bash
python scripts/narrate_book.py livre.txt --voice "..." --dry-run
```

Ne charge pas le modèle. Vérifier : le **nombre de chapitres** (un seul chapitre
pour 60 000 caractères signifie que le découpage a raté), le **premier segment**
(c'est ce que l'auditeur entendra en premier), et la **santé de la référence**
pour une voix clonée.

**Le titre.** Un `.txt` ne porte pas de métadonnées, donc sans `--title` le
générique annonce le nom du fichier. Cinq livres se sont ouverts sur
« livre-un-esprits-reprogrammes » avant qu'on l'entende.

## 5. Auditer après narration — `audit_pronunciation.py`

```bash
python scripts/audit_pronunciation.py output/book_mon_livre --sample 150
```

**C'est la pièce qui rend trois cents livres possibles.** Corriger la
prononciation demande de savoir quels mots sonnent faux ; le savoir demande
d'écouter ; personne n'écoutera trois cents livres. L'arbitre ne peut donc pas
être une oreille.

Il fait relire par une machine ce qu'une autre vient de dire, et compare au
texte source. Le cache de segments garde déjà les deux côte à côte : rien n'est
à re-narrer. **Lancer l'audit avant le balayage** (`--keep deliverables` efface
le cache).

**Ce que l'audit voit** : noms propres, sigles, mots étrangers — ils n'ont pas
de filet grammatical, donc la divergence est franche.

**Ce qu'il ne voit pas** : les mots grammaticaux. Whisper corrige ce qu'il
entend d'après le sens, donc « ce livres » revient « ces livres ». C'est
l'oreille qui a attrapé « ces » prononcé « ce », et c'est l'oreille qui
attrapera les suivants. Ne pas prétendre le contraire.

## 6. Choisir l'orthographe — `try_pronunciation.py`

```bash
python scripts/try_pronunciation.py --phrase "..." --mot ces \
    --candidats "cés" "sés" --voice "Aurore — livre audio"
```

Produit un extrait par candidat, dans la voix du livre. **Le premier fichier est
toujours le texte non modifié** : sans lui on compare des corrections entre
elles sans savoir si l'une bat le défaut de départ.

La règle que porte le lexique lui-même : **écoutez d'abord**. Si le moteur dit
déjà bien « TDAH », n'y touchez pas — une correction inutile ne peut que
dégrader.

## Ce qui n'est pas de la préparation, mais qu'on confond avec

Trois défauts ressemblent à de la prononciation et n'en sont pas. Les corriger
dans le texte ne sert à rien :

- **`truncated`** — le moteur a coupé le segment. Une phrase de plus de 300
  caractères n'est pas lue lentement, elle est tronquée : le découpage la coupe
  désormais sur une virgule.
- **`runaway`** — un fragment de deux caractères fait babiller le moteur.
  Régénérer n'y change rien (mesuré : 0,6 s de babil devenu 1,6 s) ; les
  fragments sont fusionnés avec leur voisin.
- **plancher de bruit** — une voix clonée hérite du fond de sa référence, et
  l'ACX refuse au-dessus de −60 dBFS. Cela se règle au mastering, pas dans le
  texte.

## L'ordre qui fait gagner du temps

1. Convertir et **lire le rapport** de ce qui a été retiré.
2. Scanner les mots à risque, traiter les abréviations maison.
3. Pré-vol, vérifier chapitres, premier segment, titre.
4. Narrer **un** livre.
5. Auditer, faire écouter les candidats, enrichir le lexique.
6. Alors seulement, lancer le lot.

Narrer vingt livres avant d'auditer le premier, c'est découvrir vingt fois la
même faute.
