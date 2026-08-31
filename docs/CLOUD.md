# Narrer ailleurs que sur son poste

Ce guide sert à une chose : pouvoir produire un livre audio **depuis n'importe où**,
sans dépendre de la machine qu'on a sous la main. Trois routes, selon ce qu'on
cherche — la permanence, la vitesse, ou la gratuité.

Une seule commande les prépare toutes les trois, parce que la seule chose qui les
distingue est la présence d'un GPU, et le script la détecte au lieu de la demander :

```bash
curl -fsSL https://raw.githubusercontent.com/Eddyosas008/VoxCPM/claude/repo-analysis-improvement-dg0ies/scripts/cloud_setup.sh | bash
```

Il installe les paquets système, clone le dépôt, choisit la roue PyTorch adaptée
(CUDA ou CPU — les roues CPU sont bien plus légères), installe le projet et
télécharge le modèle. Relancé, il ne refait rien : chaque étape vérifie avant
d'agir.

## Ce que pèse l'installation

| | |
|---|---|
| Modèle VoxCPM2 (cache Hugging Face) | ~4,6 Go |
| Environnement Python avec PyTorch | ~1,6 Go |
| **Total** | **~6,2 Go** |

Autrement dit : l'espace disque n'est jamais le facteur limitant. **La RAM l'est.**
Le modèle demande environ **8,7 Go résidents en float32**, et c'est au chargement
des poids qu'il meurt quand ils manquent. En dessous de 12 Go de RAM, il faut
`VOXCPM_CPU_DTYPE=bfloat16` : empreinte divisée par deux (~4,4 Go), un peu plus
lent parce que le bfloat16 est émulé sur processeur.

## Route 1 — un VPS, pour la permanence

**Ce que ça apporte** : la machine tourne en continu. On lance une narration, on
ferme son portable, on récupère les chapitres deux jours plus tard. La chaîne
étant reprenable au segment près, une coupure ne coûte que le segment en cours.

**Ce que ça n'apporte pas** : de la vitesse. Un VPS d'entrée de gamme a 2 cœurs,
c'est-à-dire moins qu'un portable courant.

Exemple mesuré sur un Hostinger **KVM 2** — 2 cœurs, 8 Go de RAM, 100 Go de
disque, Ubuntu 24.04 : le disque est confortable, la RAM impose le bfloat16, et
les 2 cœurs rendent la narration environ **2 à 3 fois plus lente** qu'un portable
à 4 cœurs. Un livre de 3 heures y demande de l'ordre de **5 jours** — acceptable
seulement parce que personne n'attend devant.

Monter en gamme change la donne : 8 cœurs et 32 Go permettent le float32 et
divisent le temps par quatre. À comparer honnêtement au coût d'un GPU loué à
l'heure, qui fait le même livre en moins d'une heure.

## Route 2 — un GPU loué à l'heure, pour la vitesse

C'est la seule option qui change l'ordre de grandeur : **RTF ~0,3 contre ~18 à 40
sur processeur**, soit un livre de 3 heures en moins d'une heure de calcul.

Chez [RunPod](https://www.runpod.io/pricing), une RTX 4090 est à environ
**0,34 $/h** en Community Cloud, facturée à la seconde. Un livre entier coûte donc
moins qu'un café. Aucun engagement : on crée l'instance, on narre, on rapatrie,
on détruit.

### Le volume persistant, qui rend la location à l'heure supportable

Créer l'instance **avec un volume réseau** monté sur `/workspace`, une fois pour
toutes. Sans lui, chaque livre recommence par 4,6 Go de modèle à télécharger et
un environnement Python à construire — vingt minutes payées au tarif GPU, à
chaque fois, pour retrouver un état identique au précédent.

`cloud_setup.sh` s'installe de lui-même sur `/workspace` quand il en trouve un,
et y place le cache Hugging Face à côté. Vingt gigaoctets suffisent (~1,4 $/mois)
et le deuxième livre démarre en deux minutes au lieu de vingt.

**Quelle carte ?** Le modèle tient dans 5 Go de VRAM : n'importe quelle carte à
partir de 12 Go convient, et une RTX 4090 est déjà large. Le script choisit la
roue PyTorch d'après la *compute capability* rapportée par le pilote, donc une
carte Blackwell (RTX 5090) reçoit bien `cu128` et non `cu124` — avec lequel torch
se charge, voit la carte, puis échoue au premier calcul.

### Les quatre étapes d'un livre

```powershell
# 1. La machine, une fois créée (SSH selon l'IP et le port donnés par RunPod)
ssh root@<ip> -p <port>
curl -fsSL https://raw.githubusercontent.com/Eddyosas008/VoxCPM/claude/repo-analysis-improvement-dg0ies/scripts/cloud_setup.sh | bash

# 2. Depuis votre poste : les voix clonées et le livre
#    (assets/voices/ est hors du dépôt — voir plus bas)
./scripts/gpu_session.ps1 push -RemoteHost <ip> -Port <port> -Book C:\livres\mon_livre.epub

# 3. Sur la machine louée : narrer, sans surveillance
source /workspace/voxcpm/env.sh
nohup python scripts/narrate_book.py mon_livre.epub --device cuda \
    --voice 'Aurore — livre audio' --assemble m4b --export-acx \
    > narration.log 2>&1 &
tail -f narration.log

# 4. Depuis votre poste, quand c'est fini
./scripts/gpu_session.ps1 pull -RemoteHost <ip> -Port <port> -Name mon_livre
```

**Détruire l'instance en partant.** Elle est facturée tant qu'elle existe, même
inactive — et vérifier le contenu rapatrié *avant* de détruire, pas après.

### Les voix clonées ne voyagent pas avec le dépôt

`assets/voices/` est dans le `.gitignore`, délibérément : ce sont des
enregistrements de personnes réelles et le dépôt est public. Un clone frais a
donc les quatorze voix de synthèse et **aucune des voix clonées** — leurs
références pointent vers des fichiers absents, et l'échec ne se verrait qu'à la
génération, sur un GPU facturé. `cloud_setup.sh` le signale à la fin de
l'installation, et `gpu_session.ps1 push` envoie les 4,6 Mo qui manquent.

`rsync` n'existe pas sur Windows : `gpu_session.ps1` s'appuie sur `scp`, livré
avec OpenSSH. Sous Linux ou macOS, `rsync -avz` reste évidemment plus efficace.

## Route 3 — Kaggle, pour ne rien payer

[Kaggle](https://www.kaggle.com/product-feedback/173129) donne **une trentaine
d'heures de GPU par semaine** (P100, ou deux T4), en sessions de **12 heures
maximum**. C'est gratuit, c'est un vrai GPU, et la limite de session n'est pas
bloquante ici : le cache par segment fait qu'une session reprend là où la
précédente s'est arrêtée. Un livre de 3 heures tient en une ou deux sessions.

La contrainte est le disque éphémère : il faut écrire les chapitres dans les
*outputs* du notebook, ou les pousser ailleurs avant la fin de session.

## Atteindre l'interface à distance, sans l'offrir à tout le monde

`app.py` écoute par défaut sur `0.0.0.0`, c'est-à-dire sur toutes les interfaces.
Sur une machine distante, **cela met le modèle à la disposition de quiconque
trouve le port** — et sur un GPU loué, c'est votre facture qui synthétise pour un
inconnu. L'application le signale désormais au démarrage.

Deux façons correctes :

**Le tunnel SSH** — rien n'est exposé, c'est la plus sûre :

```bash
# sur le serveur
./.venv/bin/python app.py --host 127.0.0.1 --port 8808 --device cuda --no-denoiser

# sur votre poste
ssh -N -L 8808:127.0.0.1:8808 root@<ip>
# puis http://127.0.0.1:8808
```

Sous Windows, où le port de la machine louée n'est presque jamais 22 :

```powershell
./scripts/gpu_session.ps1 tunnel -RemoteHost <ip> -Port <port>
```

**Un mot de passe**, si l'accès direct est nécessaire :

```bash
VOXCPM_AUTH='edwin:motdepasse' ./.venv/bin/python app.py \
    --host 0.0.0.0 --port 8808 --device cuda --no-denoiser
```

Le mot de passe passe par la variable d'environnement plutôt que par
`--auth` en ligne de commande, pour qu'il n'atterrisse ni dans l'historique du
shell ni dans la liste des processus.

## Laisser tourner sans surveillance

Sur un VPS, une narration dure des jours : elle doit survivre à la fermeture de la
session SSH.

```bash
nohup ./.venv/bin/python scripts/narrate_book.py livre.epub \
    --voice "..." --outdir output/book_mon_livre > narration.log 2>&1 &
tail -f narration.log
```

Pour l'interface, qui elle doit repartir après un redémarrage du serveur, un
service systemd :

```ini
# /etc/systemd/system/voxcpm.service
[Unit]
Description=VoxCPM narration
After=network.target

[Service]
User=root
WorkingDirectory=/root/voxcpm
Environment=VOXCPM_AUTH=edwin:motdepasse
Environment=VOXCPM_CPU_DTYPE=bfloat16
ExecStart=/root/voxcpm/.venv/bin/python app.py --host 127.0.0.1 --port 8808 --device cpu --no-denoiser
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now voxcpm
journalctl -u voxcpm -f
```

## Couper la chaîne en deux

Le paquet `narration/` ne dépend ni de `torch` ni de `gradio` — c'est délibéré.
Tout ce qui n'est pas la synthèse tourne donc sur n'importe quelle petite machine,
en quelques secondes :

- contrôle qualité et listage des défauts (`repair_segment.py --list`)
- réparation d'un segment (celle-ci a besoin du modèle)
- assemblage M4B avec marqueurs et couverture
- contrôle de conformité et export de dépôt (`export_acx.py`)

L'architecture qui en découle : **le GPU loué ne fait que synthétiser**, quelques
dizaines de minutes, et tout le reste vit sur le VPS ou sur le poste local. C'est
ce qui rend la location à l'heure économique.

## Récapitulatif

| | Vitesse | Coût | Pour quoi |
|---|---|---|---|
| **VPS 2 cœurs** | ~3× plus lent qu'un portable | déjà payé | Permanence, stockage, tout le hors-synthèse |
| **VPS 8 cœurs** | ~4× un portable | abonnement mensuel | Narration sans surveillance, sans louer |
| **GPU à l'heure** | **~60× un portable** | ~0,34 $/h, soit ~0,35 $ le livre | Un livre entier en moins d'une heure |
| **Kaggle** | GPU, sessions de 12 h | gratuit | Essais, et livres entiers avec un peu de patience |

Le même GPU laissé allumé en permanence coûterait ~248 $/mois. À l'usage — un
livre de temps en temps — la location à la demande revient donc environ deux
cents fois moins cher, et c'est le volume persistant qui la rend praticable.
