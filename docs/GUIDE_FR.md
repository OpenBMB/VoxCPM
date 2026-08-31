# Guide d'utilisation VoxCPM (Français)

VoxCPM est un système de synthèse vocale (TTS) **sans tokenizer** qui génère une parole
très naturelle dans **30 langues** (dont le français). Ce guide couvre l'installation et
les différentes façons d'utiliser l'application.

## Sommaire

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Interface web (recommandé)](#interface-web-recommandé)
- [Les trois modes de génération](#les-trois-modes-de-génération)
- [Livres audio](#livres-audio)
- [API REST](#api-rest)
- [Ligne de commande (CLI)](#ligne-de-commande-cli)
- [API Python](#api-python)
- [Docker](#docker)
- [Conseils et dépannage](#conseils-et-dépannage)

---

## Prérequis

| Composant | Version |
|-----------|---------|
| Python    | ≥ 3.10 et < 3.13 |
| PyTorch   | ≥ 2.5.0 |
| GPU       | NVIDIA avec CUDA ≥ 12.0 (~8 Go de VRAM pour VoxCPM2) — fonctionne aussi sur CPU ou Apple Silicon (MPS), plus lentement |

Le modèle **VoxCPM2** (~2B paramètres) est téléchargé automatiquement depuis
Hugging Face au premier lancement (plusieurs Go — prévoyez du temps et de l'espace disque).

## Installation

```bash
# Depuis PyPI
pip install voxcpm

# Ou depuis ce dépôt (mode développement, avec le serveur web)
git clone https://github.com/eddyosas008/voxcpm.git
cd voxcpm
pip install -e ".[server]"
```

## Interface web (recommandé)

Le dépôt fournit **deux interfaces web** :

### 1. VoxCPM Studio (interface légère + API REST)

```bash
pip install -e ".[server]"
python server.py --port 8000
# puis ouvrez http://localhost:8000
```

Interface moderne en **français/anglais** avec les trois modes de génération,
les réglages avancés (CFG, étapes de diffusion, seed), l'historique de session
et le téléchargement des fichiers WAV. Le serveur expose aussi une API REST
documentée sur `http://localhost:8000/docs`.

Options utiles :

```bash
python server.py --device cpu          # forcer le CPU
python server.py --preload             # charger le modèle au démarrage
python server.py --model-id ./chemin/local/VoxCPM2   # modèle local
python server.py --host 0.0.0.0        # exposer sur le réseau (attention : pas d'authentification)
```

### 2. Démo Gradio (interface officielle)

```bash
python app.py --port 8808
# puis ouvrez http://localhost:8808
```

Cette interface propose en plus la **transcription automatique** de l'audio de
référence (nécessite `pip install funasr`).

## Les trois modes de génération

| Mode | Ce qu'il faut fournir | Résultat |
|------|----------------------|----------|
| 🎨 **Création de voix** (Voice Design) | Une description de la voix (« voix féminine jeune, douce, débit posé ») + le texte | Une voix inédite créée à partir de la description |
| 🎛️ **Clonage contrôlable** | Un extrait audio de référence (5–15 s) + le texte, avec en option une consigne de style | La voix du clip est clonée ; le style peut être ajusté |
| 🎙️ **Clonage ultime** | L'extrait audio **et** sa transcription exacte + le texte | Reproduction fidèle de chaque nuance vocale (timbre, rythme, émotion) |

**Conseils pour la description de voix** : indiquez le genre, l'âge, le ton, l'émotion
et le débit. Fonctionne en français, anglais, chinois… Exemples :

- *« Voix masculine grave et posée, ton de narrateur de documentaire »*
- *« Jeune femme enjouée, débit rapide, ton enthousiaste »*

## Livres audio

Ce fork ajoute une chaîne de production complète pour la narration longue en
français : import d'un `.txt` ou d'un `.epub`, préparation du texte (nombres,
abréviations, chiffres romains lus correctement), découpage avec pauses selon la
ponctuation, mastering aux normes des plateformes de livres audio, reprise après
interruption au segment près, et assemblage en M4B/MP3 avec marqueurs de chapitres.

Trois points d'entrée :

```bash
# Onglet « 📚 Livre audio » de la démo Gradio
python app.py --port 8808 --no-denoiser

# Narrer un livre entier en ligne de commande (.txt ou .epub)
python scripts/narrate_book.py livre.epub --voice "Narrateur profond & calme" --assemble m4b

# Assembler des chapitres déjà générés
python scripts/assemble_audiobook.py output/book_mon_livre --title "Mon Livre" --check

# Préparer le dossier à déposer chez un distributeur (ACX/Audible, Amazon…)
python scripts/export_acx.py output/book_mon_livre
```

**→ Pour narrer ailleurs que sur son poste** — VPS, GPU loué à l'heure, Kaggle,
et comment atteindre l'interface à distance sans l'exposer : [docs/CLOUD.md](CLOUD.md).

**→ Le guide détaillé est dans [docs/NARRATION.md](NARRATION.md)** : vitesse selon le
matériel, import EPUB, réglages par usage (fiction, documentaire, méditation,
podcast), lexique de prononciation personnalisé, et normes de sonie.

## API REST

Le serveur (`python server.py`) expose :

### `POST /api/tts` — génération complète (multipart/form-data)

```bash
# Création de voix
curl -X POST http://localhost:8000/api/tts \
  -F "text=Bonjour, ceci est une démonstration de VoxCPM en français." \
  -F "control=Voix féminine chaleureuse, débit naturel" \
  -F "seed=42" \
  -o sortie.wav

# Clonage à partir d'un extrait audio
curl -X POST http://localhost:8000/api/tts \
  -F "text=Cette phrase sera prononcée avec la voix clonée." \
  -F "reference_audio=@ma_voix.wav" \
  -o clone.wav

# Clonage ultime (audio + transcription)
curl -X POST http://localhost:8000/api/tts \
  -F "text=Reproduction fidèle de la voix." \
  -F "prompt_audio=@ma_voix.wav" \
  -F "prompt_text=Transcription exacte de l'extrait audio." \
  -F "reference_audio=@ma_voix.wav" \
  -o clone_ultime.wav
```

Paramètres : `cfg_value` (1.0–3.0, défaut 2.0), `inference_timesteps` (4–30, défaut 10),
`seed`, `normalize`, `denoise`, `response_format` (`wav`, `flac`, `ogg`).

### `POST /v1/audio/speech` — compatible OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="non-requis")
response = client.audio.speech.create(
    model="voxcpm",
    voice="voix féminine douce et posée",   # description libre de la voix
    input="Bonjour le monde !",
    response_format="wav",
)
response.write_to_file("sortie.wav")
```

### `GET /api/health` — état du serveur et du modèle

## Ligne de commande (CLI)

```bash
# Création de voix
voxcpm design --text "Bonjour tout le monde" \
  --control "voix masculine chaleureuse" --output sortie.wav

# Clonage
voxcpm clone --text "Texte à prononcer" \
  --reference-audio ma_voix.wav --output clone.wav

# Clonage ultime
voxcpm clone --text "Texte à prononcer" \
  --prompt-audio ma_voix.wav --prompt-text "transcription de l'extrait" \
  --reference-audio ma_voix.wav --output clone.wav

# Traitement par lots (une ligne du fichier = un fichier audio)
voxcpm batch --input examples/input.txt --output-dir sorties/
```

## API Python

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

wav = model.generate(
    text="(voix féminine douce)Bonjour, bienvenue dans VoxCPM !",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
```

Streaming :

```python
for chunk in model.generate_streaming(text="Synthèse en continu…"):
    ...  # traiter chaque morceau d'audio au fil de l'eau
```

## Docker

```bash
docker build -t voxcpm-server .
docker run --gpus all -p 8000:8000 -v voxcpm-cache:/root/.cache voxcpm-server
# puis ouvrez http://localhost:8000
```

Le volume `voxcpm-cache` conserve les poids du modèle entre les redémarrages.

## Conseils et dépannage

- **Le premier lancement est long** : les poids du modèle (~plusieurs Go) sont téléchargés,
  puis le modèle est compilé (`torch.compile`). Les requêtes suivantes sont rapides.
- **Mémoire insuffisante** : utilisez `--no-optimize` pour désactiver la compilation,
  ou le modèle plus petit `--model-id openbmb/VoxCPM1.5`.
- **Résultat instable en création de voix** : c'est connu — relancez la génération
  1 à 3 fois (avec un seed différent) pour obtenir la voix souhaitée.
- **Qualité du clonage** : utilisez un extrait propre de 5 à 15 secondes, sans musique
  de fond ; activez le **débruitage** si l'enregistrement est bruité.
- **Apple Silicon** : `--device mps` (ou `auto`).
- **Éthique** : le clonage de voix ne doit jamais servir à l'usurpation d'identité,
  la fraude ou la désinformation. Signalez clairement tout contenu généré par IA.

---

Documentation complète (en anglais) : [voxcpm.readthedocs.io](https://voxcpm.readthedocs.io/en/latest/)
