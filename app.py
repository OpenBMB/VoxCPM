import os
import re
import sys
import json
import dataclasses
import time
import logging
import random
import numpy as np
import soundfile as sf
import gradio as gr
from typing import Any, List, Optional, Tuple
from pathlib import Path

try:
    from funasr import AutoModel
except ImportError:  # funasr is optional — only needed for auto-transcription (ASR)
    AutoModel = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import resolve_runtime_device

# Audiobook production chain. These modules depend only on numpy/soundfile, so
# they stay importable (and testable) without the engine.
from narration import assemble as assembly
from narration import audio as audio_tools
from narration import cache as cache_tools
from narration import epub as epub_reader
from narration import chunking, credits, delivery, quality, repair, text_en, text_fr
from narration import voices as voices_catalogue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    "`Target Text`: *\"I never asked you to stay… It's not like I care or anything. "
    "But… why does it still hurt so much now that you're gone?\"*  \n\n"
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_USAGE_INSTRUCTIONS_FR = (
    "**VoxCPM2 — Trois modes de génération vocale :**\n\n"
    "🎨 **Création de voix** — Créer une voix inédite  \n"
    "Aucun audio de référence requis. Décrivez les caractéristiques de la voix souhaitée "
    "(genre, âge, timbre, émotion, débit…) dans **Description de la voix / style**, et VoxCPM2 "
    "façonnera une voix unique à partir de votre seule description.\n\n"
    "🎛️ **Clonage contrôlé** — Cloner une voix avec un guidage de style optionnel  \n"
    "Téléversez un extrait audio de référence, puis utilisez **Description de la voix / style** pour "
    "orienter l'émotion, le débit et le style global tout en préservant le timbre d'origine.\n\n"
    "🎙️ **Clonage ultime** — Reproduire chaque nuance vocale par continuation audio  \n"
    "Activez le **Mode clonage ultime** et fournissez (ou faites transcrire automatiquement) le texte "
    "de l'audio de référence. Le modèle traite l'extrait comme un préfixe déjà prononcé et le **continue** "
    "de façon fluide, en préservant fidèlement chaque détail vocal. "
    "Note : ce mode désactive la Description de la voix / style."
)

_EXAMPLES_FOOTER_FR = (
    "---\n"
    "**💡 Exemples de description de voix :**  \n"
    "Essayez les descriptions suivantes pour explorer différentes voix :  \n\n"
    "**Exemple 1 — Jeune fille douce et mélancolique**  \n"
    '`Description`: *"Une jeune fille à la voix douce et suave. '
    'Parle lentement, avec un ton mélancolique et légèrement boudeur."*  \n'
    "`Texte cible`: *\"Je ne t'ai jamais demandé de rester… Ce n'est pas comme si ça me faisait "
    "quelque chose. Mais… pourquoi est-ce que ça fait encore aussi mal maintenant que tu es parti ?\"*  \n\n"
    "**Exemple 2 — Surfeur décontracté**  \n"
    '`Description`: *"Voix masculine jeune et relâchée, légèrement nasillarde, '
    'débit traînant, très décontractée et cool."*  \n'
    '`Texte cible`: *"Mec, t\'as vu cette série de vagues ? La houle est totalement démente aujourd\'hui. '
    "J'ai enchaîné les tubes toute la matinée — c'est juste, genre, parfait, tu vois ce que je veux dire ?\"*"
)

_BOOK_INTRO_EN = (
    "### 📚 Narrate a whole book\n\n"
    "Load a `.txt` file, pick a voice in the **Studio** tab, then narrate. Chapters are "
    "separated by a line containing only `---`.\n\n"
    "- Every chapter is written to `output/book_<name>/` as it finishes, so nothing is lost "
    "if you stop midway.\n"
    "- Every segment is cached: restarting resumes at the segment it stopped on, not at the "
    "beginning of the chapter.\n"
    "- **On CPU this is slow** (roughly 40× slower than real time). Narrate a chapter or two "
    "to check the voice before committing to a whole book."
)

_BOOK_INTRO_FR = (
    "### 📚 Narrer un livre entier\n\n"
    "Chargez un fichier `.txt`, choisissez une voix dans l'onglet **Studio**, puis lancez la "
    "narration. Les chapitres sont séparés par une ligne contenant uniquement `---`.\n\n"
    "- Chaque chapitre est écrit dans `output/book_<nom>/` dès qu'il est terminé : rien n'est "
    "perdu si vous arrêtez en cours de route.\n"
    "- Chaque segment est mis en cache : relancer reprend au segment interrompu, pas au début "
    "du chapitre.\n"
    "- **Sur CPU c'est lent** (environ 40× le temps réel). Narrez un ou deux chapitres pour "
    "valider la voix avant de lancer un livre entier."
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "seed_label": "Seed",
        "seed_info": "Seed used for reproducible generation. Updated with the actual successful seed after generation.",
        "random_seed_label": "Random Seed",
        "random_seed_info": "Generate a new seed before each inference run.",
        "preset_lang_label": "🌐 Language",
        "preset_voices_label": "🎭 Preset narration voices",
        "preset_voices_info": "Pick a voice to auto-fill the description and seed.",
        "preview_btn_label": "🔊 Preview this voice",
        "chunking_label": "Split long texts (audiobooks)",
        "chunking_info": "Automatically split long texts into sentence chunks and stitch the audio together.",
        "chunk_size_label": "Max characters per chunk",
        "chunk_size_info": "Target size of each chunk when splitting long texts (whole sentences are kept together).",
        "load_txt_label": "📄 Load a .txt or .epub file",
        "prepare_text_label": "Prepare French text",
        "prepare_text_info": "Read numbers, abbreviations and Roman numerals as a narrator would (1789, M. Dupont, XIVe).",
        "master_label": "Audiobook mastering",
        "master_info": "Punctuation-aware pauses, trimmed segment edges, click-free joins and one loudness pass.",
        "tab_studio": "🎙️ Studio",
        "tab_book": "📚 Audiobook",
        "book_intro": _BOOK_INTRO_EN,
        "book_file_label": "📄 Load the book (.txt or .epub)",
        "book_epub_loaded": "**{chapters} chapter(s) imported** from « {title} »{author}. "
                            "Check the text below — front matter and a publisher's table of "
                            "contents are imported like anything else, and are yours to delete.",
        "book_text_label": "Book text — separate chapters with a line containing only ---",
        "book_title_label": "Book title",
        "book_author_label": "Author / narrator",
        "book_language_label": "Language of the book",
        "book_language_info": "Chooses the text preparation (numbers, abbreviations) and the wording of the credits.",
        "book_narrator_label": "Narrator named in the credits",
        "book_narrator_info": "Left empty, the credits state that the reading is a synthetic voice — which is what distributors require.",
        "book_credits_label": "Opening and closing credits",
        "book_credits_info": "Distributors (ACX/Audible, Amazon, Apple Books) reject a recording that does not announce its title, author and narrator at both ends.",
        "book_plan_btn": "🔍 Analyse without generating",
        "book_plan_label": "Plan",
        "book_generate_btn": "📖 Narrate the book",
        "book_assemble_btn": "📦 Assemble the audiobook",
        "book_check_btn": "✅ Check against distribution standards",
        "book_format_label": "Format",
        "book_bitrate_label": "Bitrate",
        "book_bitrate_info": "64k AAC is what Audible itself streams; speech gains little above it. Raise it for an archive copy.",
        "book_status_label": "Progress",
        "book_audio_label": "Last finished chapter",
        "book_file_output_label": "Assembled file",
        "book_settings_title": "⚙️ Narration settings",
        "book_target_rms_label": "Loudness target (dBFS)",
        "book_target_rms_info": "Audiobook platforms expect RMS between -23 and -18 dBFS.",
        "book_pause_sentence_label": "Pause after a sentence (s)",
        "book_pause_paragraph_label": "Pause after a paragraph (s)",
        "book_polish_label": "Studio chain",
        "book_polish_info": "High-pass, de-esser, compressor and limiter over each chapter before its level is set — what a listener hears, beyond the levels a distributor checks.",
        "book_repair_title": "🔧 Repair a flagged segment",
        "book_repair_info": "Re-generate a single defective segment and restitch its chapter "
                            "from the cache. The other segments are never re-synthesized.",
        "book_scan_btn": "🔍 Scan this book for defects",
        "book_defect_label": "Segment to repair",
        "book_repair_btn": "🔧 Re-generate this segment",
        "book_qc_label": "Quality re-rolls per segment",
        "book_qc_info": "A segment that comes back truncated, silent or babbling is generated "
                        "again with a derived seed. 0 only reports the defects.",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "fr": {
        "reference_audio_label": "🎤 Audio de référence (optionnel — pour le clonage)",
        "show_prompt_text_label": "🎙️ Mode clonage ultime (clonage guidé par le texte)",
        "show_prompt_text_info": "Transcrit automatiquement l'audio de référence pour reproduire chaque nuance vocale. La Description de la voix / style sera désactivée quand ce mode est actif.",
        "prompt_text_label": "Transcription de l'audio de référence (remplie via ASR, modifiable)",
        "prompt_text_placeholder": "La transcription de votre audio de référence apparaîtra ici …",
        "control_label": "🎛️ Description de la voix / style (optionnel — français, anglais, chinois)",
        "control_placeholder": "ex. Voix masculine chaleureuse / Jeune femme douce / Rapide et enthousiaste",
        "target_text_label": "✍️ Texte à synthétiser — le contenu à dire",
        "generate_btn": "🔊 Générer la voix",
        "generated_audio_label": "Audio généré",
        "advanced_settings_title": "⚙️ Réglages avancés",
        "ref_denoise_label": "Amélioration de l'audio de référence",
        "ref_denoise_info": "Applique un débruitage ZipEnhancer à l'audio de référence avant le clonage",
        "normalize_label": "Normalisation du texte",
        "normalize_info": "Normalise les nombres, dates et abréviations (via wetext)",
        "cfg_label": "CFG (intensité du guidage)",
        "cfg_info": "Plus élevé → plus fidèle à la description / référence ; plus bas → variation plus créative",
        "dit_steps_label": "Étapes de diffusion (LocDiT)",
        "dit_steps_info": "Étapes de flow-matching LocDiT — plus d'étapes → qualité potentiellement meilleure, mais plus lent",
        "seed_label": "Graine (seed)",
        "seed_info": "Graine utilisée pour une génération reproductible. Mise à jour avec la graine réellement utilisée après génération.",
        "random_seed_label": "Graine aléatoire",
        "random_seed_info": "Génère une nouvelle graine avant chaque inférence.",
        "preset_lang_label": "🌐 Langue",
        "preset_voices_label": "🎭 Voix prédéfinies (narration)",
        "preset_voices_info": "Choisissez une voix pour remplir automatiquement la description et le seed.",
        "preview_btn_label": "🔊 Écouter un aperçu de la voix",
        "chunking_label": "Découper les longs textes (livres audio)",
        "chunking_info": "Découpe automatiquement les longs textes en segments de phrases et assemble l'audio.",
        "chunk_size_label": "Caractères max par segment",
        "chunk_size_info": "Taille cible de chaque segment lors du découpage (les phrases entières restent groupées).",
        "load_txt_label": "📄 Charger un fichier .txt ou .epub",
        "prepare_text_label": "Préparation du texte français",
        "prepare_text_info": "Fait lire les nombres, abréviations et chiffres romains comme un narrateur (1789, M. Dupont, XIVe).",
        "master_label": "Mastering livre audio",
        "master_info": "Pauses selon la ponctuation, bords des segments nettoyés, jointures sans clic et un seul passage de normalisation.",
        "tab_studio": "🎙️ Studio",
        "tab_book": "📚 Livre audio",
        "book_intro": _BOOK_INTRO_FR,
        "book_file_label": "📄 Charger le livre (.txt ou .epub)",
        "book_epub_loaded": "**{chapters} chapitre(s) importé(s)** depuis « {title} »{author}. "
                            "Vérifiez le texte ci-dessous : les pages de garde et une table des "
                            "matières éditoriale sont importées comme le reste, à vous de les "
                            "supprimer.",
        "book_text_label": "Texte du livre — séparez les chapitres par une ligne contenant seulement ---",
        "book_title_label": "Titre du livre",
        "book_author_label": "Auteur / narrateur",
        "book_language_label": "Langue du livre",
        "book_language_info": "Détermine la préparation du texte (nombres, abréviations) et la formulation du générique.",
        "book_narrator_label": "Narrateur cité au générique",
        "book_narrator_info": "Laissé vide, le générique indique que la lecture est une voix de synthèse — ce que les plateformes exigent.",
        "book_credits_label": "Générique de début et de fin",
        "book_credits_info": "Les distributeurs (ACX/Audible, Amazon, Apple Books) refusent un enregistrement qui n'annonce pas son titre, son auteur et son narrateur aux deux bouts.",
        "book_plan_btn": "🔍 Analyser sans générer",
        "book_plan_label": "Plan",
        "book_generate_btn": "📖 Narrer le livre",
        "book_assemble_btn": "📦 Assembler le livre audio",
        "book_check_btn": "✅ Vérifier la conformité de dépôt",
        "book_format_label": "Format",
        "book_bitrate_label": "Débit",
        "book_bitrate_info": "64k AAC est ce qu'Audible diffuse lui-même ; la parole gagne peu au-dessus. À monter pour une copie d'archive.",
        "book_status_label": "Avancement",
        "book_audio_label": "Dernier chapitre terminé",
        "book_file_output_label": "Fichier assemblé",
        "book_settings_title": "⚙️ Réglages de narration",
        "book_target_rms_label": "Cible de sonie (dBFS)",
        "book_target_rms_info": "Les plateformes de livres audio attendent un RMS entre -23 et -18 dBFS.",
        "book_pause_sentence_label": "Pause après une phrase (s)",
        "book_pause_paragraph_label": "Pause après un paragraphe (s)",
        "book_polish_label": "Chaîne studio",
        "book_polish_info": "Passe-haut, dé-esseur, compresseur et limiteur sur chaque chapitre avant le calage du niveau — ce que l'auditeur entend, au-delà des niveaux que contrôle un distributeur.",
        "book_repair_title": "🔧 Réparer un segment signalé",
        "book_repair_info": "Régénère un seul segment défectueux et reconstruit son chapitre "
                            "à partir du cache. Les autres segments ne sont jamais recalculés.",
        "book_scan_btn": "🔍 Analyser ce livre",
        "book_defect_label": "Segment à réparer",
        "book_repair_btn": "🔧 Régénérer ce segment",
        "book_qc_label": "Réessais qualité par segment",
        "book_qc_info": "Un segment qui revient tronqué, muet ou parti en boucle est régénéré "
                        "avec une graine dérivée. 0 se contente de signaler les défauts.",
        "usage_instructions": _USAGE_INSTRUCTIONS_FR,
        "examples_footer": _EXAMPLES_FOOTER_FR,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "preset_lang_label": "🌐 语言",
        "preset_voices_label": "🎭 预设旁白语音",
        "preset_voices_info": "选择一个语音以自动填充描述和随机种子。",
        "book_repair_title": "🔧 Repair a flagged segment",
        "book_repair_info": "Re-generate a single defective segment and restitch its chapter "
                            "from the cache. The other segments are never re-synthesized.",
        "book_scan_btn": "🔍 Scan this book for defects",
        "book_defect_label": "Segment to repair",
        "book_repair_btn": "🔧 Re-generate this segment",
        "preview_btn_label": "🔊 试听该语音",
        "chunking_label": "拆分长文本（有声书）",
        "chunking_info": "自动将长文本按句子拆分并拼接音频。",
        "chunk_size_label": "每段最大字符数",
        "chunk_size_info": "拆分长文本时每段的目标长度（整句会保持在一起）。",
        "load_txt_label": "📄 加载 .txt 文件",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,  # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, " "designed to generate highly realistic speech."
)

# ---------- Preset voices for narration (Voice Design mode) ----------
# Each entry regenerates the exact same voice when its (description, seed) pair is
# reused. Common defaults for all: CFG=2.0, diffusion steps=10, normalize=True.
# Built-in defaults below. To add/edit/remove voices WITHOUT touching this file,
# create conf/preset_voices.json (same keys) — it overrides the built-in list.
_BUILTIN_PRESET_VOICES = [
    {
        "name": "Narrateur profond & calme",
        "description": "Voix masculine française de narrateur pour livre audio, profonde, calme et posée, timbre chaleureux et rassurant, débit lent et immersif, diction claire et articulée",
        "seed": 4110390676,
        "cfg": 2.0,
        "diffusion_steps": 10,
        "normalize": True,
    },
    {
        "name": "Narratrice douce & naturelle",
        "description": "Voix féminine française de narratrice pour livre audio, douce et naturelle, timbre chaleureux et authentique, débit fluide et posé, diction claire, ton captivant et apaisant",
        "seed": 3227543575,
        "cfg": 2.0,
        "diffusion_steps": 10,
        "normalize": True,
    },
    {
        "name": "Conteur jeune & dynamique",
        "description": "Voix masculine française de jeune conteur d'environ vingt-cinq ans pour livre audio, dynamique et expressive, ton vivant et engageant, débit naturel, idéale pour la narration d'histoires",
        "seed": 2151638728,
        "cfg": 2.0,
        "diffusion_steps": 10,
        "normalize": True,
    },
    {
        "name": "Narratrice chaleureuse & conversationnelle",
        "description": "Voix féminine française d'âge mûr pour livre audio, chaleureuse et engageante, style conversationnel et charmant, ton bienveillant et proche de l'auditeur, diction naturelle",
        "seed": 3023399458,
        "cfg": 2.0,
        "diffusion_steps": 10,
        "normalize": True,
    },
    {
        "name": "Narrateur documentaire velouté",
        "description": "Voix masculine française de narrateur de documentaire, veloutée et posée, ton professionnel empreint de mystère et d'émerveillement, diction soignée, idéale pour nature, science et histoire",
        "seed": 1468538221,
        "cfg": 2.0,
        "diffusion_steps": 10,
        "normalize": True,
    },
    {
        "name": "Narrateur moderne & professionnel",
        "description": "Voix masculine française moderne, claire et profonde, ton assuré et régulier, débit confiant et professionnel, idéale pour la narration, les podcasts et les livres audio contemporains",
        "seed": 3515672692,
        "cfg": 2.0,
        "diffusion_steps": 10,
        "normalize": True,
    },
    {
        "name": "Méditation guidée (grave & lente)",
        "description": "Voix masculine française très grave et profonde pour méditation guidée, extrêmement lente et douce, ton chaud, apaisant et enveloppant, chuchoté et relaxant, longues pauses, respiration calme, idéale pour la détente et la relaxation",
        "seed": 560505514,
        "cfg": 2.0,
        "diffusion_steps": 10,
        "normalize": True,
    },
]

# Optional external override: conf/preset_voices.json (a JSON list of objects with
# the same keys). Lets non-developers curate the voice list without editing code.
_REPO_ROOT = Path(__file__).parent
_PRESET_VOICES_JSON = _REPO_ROOT / "conf" / "preset_voices.json"


def _load_preset_voices() -> List[dict]:
    """The voice catalogue. Reading it lives in narration.voices, which loads
    without torch or gradio — a voice list is data, not interface."""
    return voices_catalogue.load_presets(
        _PRESET_VOICES_JSON, _REPO_ROOT, _BUILTIN_PRESET_VOICES
    )


PRESET_VOICES = _load_preset_voices()
for _v in PRESET_VOICES:  # every voice has a language (defaults to French)
    _v.setdefault("lang", "fr")

# Label of the "leave everything free" option (current default behavior).
PRESET_CUSTOM_LABEL = "Personnalisé / manuel"
_PRESET_BY_NAME = {v["name"]: v for v in PRESET_VOICES}

# Distinct languages present, and human labels for the language selector. The
# selector only appears in the UI when more than one language is available.
_PRESET_LANGS = sorted({v["lang"] for v in PRESET_VOICES})
_LANG_LABELS = {"fr": "Français", "en": "English", "zh": "中文", "es": "Español", "de": "Deutsch", "it": "Italiano"}


def _lang_label(code: str) -> str:
    return _LANG_LABELS.get(code, code)


def _voice_names_for_lang(lang: Optional[str]) -> List[str]:
    """Preset voice names for a language (all voices when lang is None)."""
    return [v["name"] for v in PRESET_VOICES if lang is None or v["lang"] == lang]

# ---------- Long-text chunking (audiobooks) ----------
# Segmentation and the pause plan live in narration.chunking; the audio side
# (trimming, de-clicking, loudness) lives in narration.audio.
_CHUNK_MAX_CHARS = chunking.DEFAULT_MAX_CHARS
_CHUNK_SILENCE_SEC = chunking.PauseProfile().sentence

# Where a book narrated from the UI keeps its chapters and its resume cache.
_BOOKS_DIR = Path(__file__).parent / "output"
_LEXICON_PATH = Path(__file__).parent / "conf" / "pronunciation_fr.json"

# Short fixed phrase used to preview a preset voice on demand.
_PREVIEW_TEXT = "Bonjour, ceci est un aperçu de cette voix pour la narration de votre livre audio."
_PREVIEW_DIR = Path(__file__).parent / "assets" / "voice_previews"

# Previews are mastered like a chapter, for two reasons. Raw generations land
# anywhere between -32 and -17 dBFS depending on the voice — a 15 dB spread, wide
# enough that the quietest presets are barely audible on laptop speakers. And
# comparing voices at different levels is not a comparison: the louder one always
# sounds better. Shorter edge silences than a chapter, so a preview starts
# playing at once.
_PREVIEW_MASTERING = audio_tools.MasteringSettings(lead_sec=0.1, tail_sec=0.2)


def master_preview(sample_rate: int, wav):
    """Trim, de-click and level a preview to the narration target."""
    return audio_tools.stitch([(wav, 0.0)], sample_rate, _PREVIEW_MASTERING)

# Every generation is also archived here with a descriptive filename.
_OUTPUT_DIR = Path(__file__).parent / "output"


def _sanitize_filename(name: str) -> str:
    """Turn a voice name into a safe filename fragment."""
    name = re.sub(r"[^\w]+", "_", (name or "").strip(), flags=re.UNICODE)
    return name.strip("_")[:60] or "custom"


def _save_output_wav(wav_np: np.ndarray, sr: int, seed: Optional[int], voice_name: str) -> str:
    """Write the generated audio to output/ with a descriptive name; return the path."""
    import soundfile as sf

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    seed_part = f"seed{seed}" if seed is not None else "seedrandom"
    out = _OUTPUT_DIR / f"narration_{_sanitize_filename(voice_name)}_{seed_part}_{stamp}.wav"
    sf.write(str(out), wav_np, sr)
    logger.info(f"Saved generated audio -> {out}")
    return str(out)


def _split_text_into_chunks(text: str, max_chars: int = _CHUNK_MAX_CHARS) -> List[str]:
    """Greedily pack whole sentences into chunks no longer than ``max_chars``.

    Thin wrapper kept for the existing single-shot path and any external caller;
    the segmentation itself now lives in ``narration.chunking``, alongside the
    pause plan that long-form narration needs.
    """
    return chunking.split_text_into_chunks(text, max_chars)

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------


class VoxCPMDemo:
    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        device: str = "auto",
        load_denoiser: bool = True,
    ) -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")
        # The ZipEnhancer denoiser is a ModelScope model only needed to clean a
        # reference audio clip. For narration (Voice Design, no reference audio)
        # it is never used — disable it with --no-denoiser to skip a slow/blocking
        # ModelScope download and start much faster (useful on CPU-only machines).
        self.load_denoiser = load_denoiser

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[Any] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id} (denoiser={'on' if self.load_denoiser else 'off'})")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=self.optimize,
            device=self.device,
            load_denoiser=self.load_denoiser,
        )
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> "AutoModel":
        if AutoModel is None:
            raise RuntimeError(
                "funasr is not installed — automatic transcription of the reference audio "
                "is unavailable. Install it with `pip install funasr`, or type the "
                "transcript manually in the prompt text field."
            )
        if self.asr_model is not None:
            return self.asr_model
        logger.info(f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}")
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
        )
        wav = current_model.generate(**generate_kwargs)
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)


# ---------- UI ----------


def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _coerce_seed(seed_value) -> Optional[int]:
        if seed_value is None or seed_value == "":
            return None
        return int(seed_value)

    def _resolve_voice(preset_name, description, seed_value, cfg, steps, normalize):
        """Voice parameters to use, preferring the selected preset over the boxes.

        A preset carries its own description, seed, CFG and steps. Reading those
        from the text boxes instead lets the two drift apart — and a missing seed
        turns "play this voice" into a from-scratch generation, which is tens of
        minutes on a CPU.
        """
        preset = (
            _PRESET_BY_NAME.get(preset_name)
            if preset_name and preset_name != PRESET_CUSTOM_LABEL
            else None
        )
        if preset is None:
            return (description or ""), _coerce_seed(seed_value), cfg, steps, normalize
        return (
            preset.get("description", description or ""),
            _coerce_seed(preset.get("seed")),
            preset.get("cfg", cfg),
            preset.get("diffusion_steps", steps),
            preset.get("normalize", normalize),
        )

    def _preset_reference(preset_name) -> Tuple[Optional[str], str]:
        """The recording a preset clones, and its transcript. ("", "") if none.

        Kept apart from _resolve_voice rather than widening its tuple: that
        function is called from half a dozen places, and a cloned voice only
        concerns the two that actually synthesize.
        """
        preset = (
            _PRESET_BY_NAME.get(preset_name)
            if preset_name and preset_name != PRESET_CUSTOM_LABEL
            else None
        )
        return voices_catalogue.reference_of(preset)

    def _warn_about_reference(reference: Optional[str], reference_text: str) -> None:
        """Warn, before generating, when a recording and its transcript disagree.

        Warns rather than refuses. The bounds behind this are heuristics on a
        thin sample, and from the interface there would be no way past a wrong
        verdict — a deliberate take that measures oddly would simply become
        unusable. Being told is the whole value; being stopped is not.
        """
        if not reference:
            return
        try:
            wav, sample_rate = sf.read(reference, dtype="float32", always_2d=False)
        except Exception as error:  # noqa: BLE001 - synthesis will fail on it too
            logger.warning("Could not inspect reference %s: %s", reference, error)
            return

        report = quality.inspect_reference(wav, sample_rate, reference_text or "")
        for issue in report.issues:
            logger.warning("Reference %s: %s", Path(reference).name, issue)
            gr.Warning(f"Audio de référence : {issue.detail}")

    def _prepare_seed(use_random_seed: bool, seed_value):
        if use_random_seed:
            return random.randint(0, 2**32 - 1)
        return _coerce_seed(seed_value)

    def _on_random_seed_toggle(checked):
        return gr.update(interactive=not checked)

    def _on_lang_change(lang):
        """Restrict the voice dropdown to the chosen language and reset to Custom."""
        return gr.update(
            choices=[PRESET_CUSTOM_LABEL] + _voice_names_for_lang(lang),
            value=PRESET_CUSTOM_LABEL,
        )

    def _on_preset_change(preset_name):
        """Fill the Voice Design fields from a preset. 'Personnalisé' = no-op."""
        preset = _PRESET_BY_NAME.get(preset_name)
        if preset is None:  # "Personnalisé / manuel" → keep fields as-is (current behavior)
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        return (
            gr.update(value=preset["description"]),             # control_instruction
            gr.update(value=preset["seed"], interactive=True),  # seed_value (editable)
            gr.update(value=False),                             # random_seed → unchecked
            gr.update(value=preset["cfg"]),                     # cfg_value
            gr.update(value=preset["diffusion_steps"]),         # dit_steps
            gr.update(value=preset["normalize"]),               # DoNormalizeText
        )

    def _read_uploaded_text(file_path: str) -> Tuple[str, Optional[epub_reader.EpubBook]]:
        """Text of an uploaded .txt or .epub, plus the book when there was one."""
        if epub_reader.is_epub(file_path):
            text, book = epub_reader.load_book_text(file_path)
            logger.info(
                f"Loaded EPUB '{book.title}' — {len(book.chapters)} chapter(s), "
                f"{book.characters} chars from {file_path}"
            )
            return text, book
        return Path(file_path).read_text(encoding="utf-8").strip(), None

    def _load_text_file(file_path: Optional[str]) -> str:
        """Read a .txt or .epub file and return its text to fill a text box."""
        if not file_path:
            return gr.update()
        try:
            content, _ = _read_uploaded_text(file_path)
            logger.info(f"Loaded text file ({len(content)} chars) from {file_path}")
            return content
        except Exception as e:
            logger.warning(f"Could not read text file {file_path}: {e}")
            raise gr.Error(f"Impossible de lire le fichier : {e}")

    def _load_book_file(file_path: Optional[str]):
        """Fill the book tab from an upload — text, and an EPUB's own metadata.

        Title and author are only overwritten when the file carries them, so an
        EPUB with empty metadata never wipes what the user typed.
        """
        if not file_path:
            return gr.update(), gr.update(), gr.update(), gr.update()
        try:
            content, book = _read_uploaded_text(file_path)
        except Exception as e:
            logger.warning(f"Could not read book file {file_path}: {e}")
            raise gr.Error(f"Impossible de lire le fichier : {e}")

        if book is None:
            return content, gr.update(), gr.update(), gr.update()

        status = I18N("book_epub_loaded").format(
            chapters=len(book.chapters),
            title=book.title or Path(file_path).stem,
            author=f" — {book.author}" if book.author else "",
        )
        # What was taken out of the book is said out loud, never assumed
        # unwanted: the reader is the one who decides it was boilerplate.
        if book.removed:
            status += "\n\n" + "\n".join(f"- 🗑️ {note}" for note in book.removed)

        # The book directory is derived from the title we are about to fill in,
        # so the cover can be put where the assembly will look for it. A title
        # edited afterwards moves that directory; the cover is then simply not
        # found, which costs a picture and nothing else.
        try:
            cover = epub_reader.extract_cover(file_path, _book_dir(book.title))
            if cover:
                status += f"\n\n- 🖼️ Couverture importée (`{cover.name}`)"
        except OSError as error:  # a read-only or full disk, nothing worse
            logger.warning(f"Could not extract the EPUB cover: {error}")

        return content, book.title or gr.update(), book.author or gr.update(), status

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        seed_value,
        enable_chunking: bool,
        chunk_max_chars: int,
        preset_name: str = "",
        prepare_text: bool = False,
        master_audio: bool = True,
        progress=gr.Progress(),
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        seed = _coerce_seed(seed_value)

        # A cloned preset brings its own recording. A file dropped in the form
        # wins: the user picked it deliberately and just now.
        preset_reference, preset_reference_text = _preset_reference(preset_name)
        if preset_reference and not ref_wav:
            ref_wav = preset_reference
            actual_control = ""
            actual_prompt_text = actual_prompt_text or preset_reference_text
            _warn_about_reference(ref_wav, actual_prompt_text)
        voice_name = preset_name if preset_name and preset_name != PRESET_CUSTOM_LABEL else "custom"

        if prepare_text:
            text = text_fr.normalize_french(text, lexicon=text_fr.load_lexicon(_LEXICON_PATH))

        common = dict(
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            seed=seed,  # same seed for every chunk → consistent voice
        )

        # Only chunk plain Voice Design / control text — cloning modes keep a single pass.
        segments = (
            chunking.split_into_segments(text, int(chunk_max_chars)) if enable_chunking else []
        )
        if len(segments) <= 1 or ref_wav or actual_prompt_text:
            sr, wav_np, last_successful_seed = demo.generate_tts_audio(text_input=text, **common)
        else:
            logger.info(f"Chunked synthesis: {len(segments)} segments.")
            sr = None
            rendered: List[Tuple[np.ndarray, float]] = []
            last_successful_seed = seed
            for i, segment in enumerate(progress.tqdm(segments, desc="Synthèse des segments")):
                logger.info(f"  segment {i + 1}/{len(segments)}")
                sr, wav_chunk, last_successful_seed = demo.generate_tts_audio(
                    text_input=segment.text, **common
                )
                rendered.append((wav_chunk, segment.pause_after))
            if master_audio:
                # Punctuation-aware pauses, de-clicked joins, one loudness pass.
                wav_np = audio_tools.stitch(rendered, sr, audio_tools.MasteringSettings())
            else:
                wav_np = audio_tools.concatenate(
                    (wav for wav, _ in rendered), sr, gap_sec=_CHUNK_SILENCE_SEC
                )

        out_path = _save_output_wav(wav_np, sr, last_successful_seed, voice_name)
        return out_path, last_successful_seed

    def _preview_voice(description, seed_value, cfg, steps, normalize, preset_name=None):
        """Play the stored sample of the selected voice, generating it if absent.

        A preset is asked for its own description and seed rather than reading
        the text boxes. Those boxes can be empty or half-edited, and when the
        seed is missing there is no cache key, so what looks like "play this
        voice" silently becomes a from-scratch generation — roughly forty
        minutes on a CPU, with nothing on screen to say so.
        """
        description, seed, cfg, steps, normalize = _resolve_voice(
            preset_name, description, seed_value, cfg, steps, normalize
        )

        cache_path = None
        if seed is not None:
            _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = _PREVIEW_DIR / f"preview_{seed}.wav"
            if cache_path.is_file():
                return str(cache_path)

        if not (description or "").strip():
            raise gr.Error(
                "Aucune voix à écouter : choisissez une voix prédéfinie dans la liste, "
                "ou décrivez la voix souhaitée."
            )

        # Nothing cached, so this really is a generation. On CPU that is tens of
        # minutes; saying so beats a button that appears to do nothing.
        if not demo.device.startswith("cuda"):
            gr.Warning(
                "Aucun aperçu enregistré pour cette voix : génération en cours, "
                "comptez plusieurs dizaines de minutes sur ce processeur. "
                "scripts/pregenerate_previews.py permet de les préparer à l'avance."
            )
        sr, wav_np, _ = demo.generate_tts_audio(
            text_input=_PREVIEW_TEXT,
            control_instruction=description or "",
            cfg_value_input=cfg,
            do_normalize=normalize,
            inference_timesteps=int(steps),
            seed=seed,
        )
        wav_np = master_preview(sr, wav_np)
        if cache_path is not None:
            try:
                import soundfile as sf
                sf.write(str(cache_path), wav_np, sr)
                return str(cache_path)
            except Exception as e:
                logger.warning(f"Could not cache preview ({e}); returning in-memory audio.")
        return (sr, wav_np)

    # ---------- Audiobook tab ----------

    def _chapter_title(chapter: str, index: int) -> str:
        """First non-empty line of a chapter, used as its marker title."""
        for line in chapter.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:80]
        return f"Chapitre {index}"

    def _book_dir(title: str) -> Path:
        """Where a book's chapters and its resume cache live."""
        return _BOOKS_DIR / f"book_{_sanitize_filename(title or 'livre')}"

    def _book_credits(title, author, narrator, enabled: bool, language: str = "fr"):
        """The credits a distributor requires, or None when switched off."""
        if not enabled:
            return None
        return credits.BookCredits(
            title=(title or "").strip(),
            author=(author or "").strip(),
            narrator=(narrator or "").strip(),
            language=language,
        )

    def _book_prepared_chapters(
        book_text: str, prepare: bool, book_credits=None, language: str = "fr"
    ) -> List[str]:
        """Chapters as they will be narrated, credits included.

        The credits are chapters like any other on purpose: they then take the
        same French preparation, voice, seed and mastering as the book.
        """
        chapters = chunking.split_chapters(book_text)
        if book_credits is not None and chapters:
            chapters = [book_credits.opening()] + chapters + [book_credits.closing()]
        if not prepare:
            return chapters
        lexicon = text_fr.load_lexicon(_LEXICON_PATH)
        normalize = (
            text_en.normalize_english if language == "en" else text_fr.normalize_french
        )
        return [normalize(chapter, lexicon=lexicon) for chapter in chapters]

    def _book_chapter_title(
        chapter: str, index: int, count: int, has_credits: bool, language: str = "fr"
    ) -> str:
        """Marker title — the credits are named rather than quoted."""
        opening_title, closing_title = credits.titles_for(language)
        if has_credits and index == 1:
            return opening_title
        if has_credits and index == count:
            return closing_title
        return _chapter_title(chapter, index)

    def _book_profile(pause_sentence: float, pause_paragraph: float) -> chunking.PauseProfile:
        default = chunking.PauseProfile()
        return chunking.PauseProfile(
            clause=min(default.clause, float(pause_sentence)),
            sentence=float(pause_sentence),
            paragraph=float(pause_paragraph),
        )

    def _book_plan(
        book_text,
        chunk_max_chars_value,
        prepare,
        pause_sentence,
        pause_paragraph,
        title="",
        author="",
        narrator="",
        with_credits=True,
        language="fr",
    ):
        """Show what would be generated, without loading the model."""
        book_credits = _book_credits(title, author, narrator, with_credits, language)
        chapters = _book_prepared_chapters(book_text, prepare, book_credits, language)
        if not chapters:
            return "*Aucun texte à analyser.*"

        profile = _book_profile(pause_sentence, pause_paragraph)
        rows, total_segments, total_chars = [], 0, 0
        for index, chapter in enumerate(chapters, 1):
            segments = chunking.split_into_segments(chapter, int(chunk_max_chars_value), profile)
            characters = chunking.total_characters(segments)
            total_segments += len(segments)
            total_chars += characters
            rows.append(f"| {index} | {len(segments)} | {characters} |")

        # ~14 characters of prose per second of finished narration, and roughly
        # 40x real time to synthesize on CPU — both rough, but the difference
        # between "an afternoon" and "several days" is worth knowing up front.
        minutes = total_chars / 14.0 / 60.0
        preview = ""
        first = chunking.split_into_segments(chapters[0], int(chunk_max_chars_value), profile)
        if first:
            preview = f"\n\n**Premier segment tel qu'il sera lu :**\n\n> {first[0].text}"

        return (
            f"**{len(chapters)} chapitre(s) · {total_segments} segment(s) · {total_chars} caractères**\n\n"
            f"Durée de narration estimée : **~{minutes:.0f} min** "
            f"(soit ~{minutes * 40 / 60:.1f} h de calcul sur CPU)\n\n"
            "| Chapitre | Segments | Caractères |\n|---|---|---|\n" + "\n".join(rows) + preview
        )

    def _book_narrate(
        book_text,
        title,
        author,
        control_instruction,
        cfg_value,
        dit_steps,
        do_normalize,
        seed_value,
        chunk_max_chars_value,
        prepare,
        target_rms,
        pause_sentence,
        pause_paragraph,
        preset_name,
        qc_retries,
        narrator="",
        with_credits=True,
        polish_on=True,
        language="fr",
        progress=gr.Progress(),
    ):
        """Narrate every chapter, writing each one to disk as soon as it is done.

        Yields after each chapter so the UI shows progress on a job that runs for
        hours, and so a finished chapter is listenable before the book is.
        """
        book_credits = _book_credits(title, author, narrator, with_credits, language)
        chapters = _book_prepared_chapters(book_text, prepare, book_credits, language)
        if not chapters:
            raise gr.Error("Aucun texte à narrer. Chargez un fichier .txt ou collez le texte.")

        description, seed, cfg_value, dit_steps, do_normalize = _resolve_voice(
            preset_name, control_instruction, seed_value, cfg_value, dit_steps, do_normalize
        )
        # A cloned voice answers "which voice?" with a recording; it needs no
        # description, so the demand for one only applies to a designed voice.
        reference, reference_text = _preset_reference(preset_name)
        if not reference and not description.strip():
            raise gr.Error(
                "Choisissez une voix dans la liste ci-dessus, ou décrivez-en une "
                "dans l'onglet Studio."
            )
        # Said now rather than never: a mismatched recording truncates every
        # segment, and a book is hours of CPU before the first one is heard.
        _warn_about_reference(reference, reference_text)

        outdir = _book_dir(title)
        outdir.mkdir(parents=True, exist_ok=True)
        profile = _book_profile(pause_sentence, pause_paragraph)
        mastering = audio_tools.MasteringSettings(
            target_rms_db=float(target_rms), polish=bool(polish_on)
        )
        voice_spec = cache_tools.VoiceSpec(
            description=description,
            seed=seed,
            cfg=float(cfg_value),
            steps=int(dit_steps),
            normalize=bool(do_normalize),
            model_id=demo._model_id,
            # Hashed by content, so a chapter cloned from one recording never
            # collides in the cache with the same words from another.
            reference=cache_tools.VoiceSpec.hash_reference(reference),
            reference_text=reference_text,
        )
        cache = cache_tools.ChunkCache(outdir / ".cache")

        # The plan is what makes a later repair possible: without it the cut
        # into segments — and so which cache entry holds which sentence — is
        # lost the moment this run ends. Written before any audio, so an
        # interrupted narration is still repairable.
        chapter_plans = [
            repair.PlannedChapter(
                index=index,
                title=_book_chapter_title(
                    chapter, index, len(chapters), book_credits is not None, language
                ),
                segments=tuple(
                    repair.PlannedSegment(segment.text, segment.pause_after)
                    for segment in chunking.split_into_segments(
                        chapter, int(chunk_max_chars_value), profile
                    )
                ),
            )
            for index, chapter in enumerate(chapters, 1)
        ]
        repair.BookPlan(
            voice=dataclasses.asdict(voice_spec),
            mastering=dataclasses.asdict(mastering),
            chapters=tuple(chapter_plans),
        ).save(outdir)

        voice_label = preset_name if preset_name and preset_name != PRESET_CUSTOM_LABEL else "voix personnalisée"
        lines = [
            f"### Narration en cours\n",
            f"Voix : **{voice_label}** · graine `{seed}` · dossier `{outdir.name}`\n",
        ]
        last_chapter_path = None
        qc_flagged: List[Tuple[str, quality.SegmentReport]] = []
        qc_inspected = 0
        yield "\n".join(lines), None

        for planned in chapter_plans:
            index = planned.index
            out = outdir / f"chapitre_{index:03d}.wav"
            if out.is_file():
                lines.append(f"- ⏭️ Chapitre {index}/{len(chapters)} — déjà généré, ignoré")
                last_chapter_path = str(out)
                yield "\n".join(lines), last_chapter_path
                continue

            # The very segments recorded in the plan, so a repair addresses the
            # same cache entries this run wrote.
            segments = planned.segments
            if not segments:
                lines.append(f"- ⚠️ Chapitre {index}/{len(chapters)} — vide, ignoré")
                yield "\n".join(lines), last_chapter_path
                continue

            rendered: List[Tuple[np.ndarray, float]] = []
            sr = None
            for position, segment in enumerate(
                progress.tqdm(segments, desc=f"Chapitre {index}/{len(chapters)}"), 1
            ):
                key = cache.key(segment.text, voice_spec)
                cached = cache.get(key)
                if cached is not None:
                    sr, wav_chunk = cached
                    report = quality.inspect_segment(wav_chunk, sr, segment.text)
                else:
                    def render(current_seed, _segment=segment):
                        sample_rate, wav_out, _ = demo.generate_tts_audio(
                            text_input=_segment.text,
                            control_instruction="" if reference else description,
                            reference_wav_path_input=reference,
                            prompt_text=reference_text,
                            cfg_value_input=cfg_value,
                            do_normalize=do_normalize,
                            inference_timesteps=int(dit_steps),
                            denoise=False,
                            seed=current_seed,
                        )
                        return sample_rate, wav_out

                    result = quality.render_checked(
                        segment.text, render, base_seed=seed, max_attempts=int(qc_retries) + 1
                    )
                    sr, wav_chunk, report = result.sample_rate, result.wav, result.report
                    cache.put(key, sr, wav_chunk, text=segment.text)

                qc_inspected += 1
                if not report.ok:
                    # Surfaced as it happens rather than only in the final
                    # summary: on a run that lasts hours, a defect worth
                    # stopping for should not wait until the end to be seen.
                    qc_flagged.append((f"ch{index:03d}/seg{position:03d}", report))
                    lines.append(
                        f"  - {'❌' if report.fatal else '⚠️'} chapitre {index}, segment "
                        f"{position}/{len(segments)} — {report.describe()}"
                    )
                    yield "\n".join(lines), last_chapter_path
                rendered.append((wav_chunk, segment.pause_after))

            chapter_audio = audio_tools.stitch(rendered, sr, mastering)
            sf.write(str(out), chapter_audio, sr, subtype="PCM_16")
            report = audio_tools.acx_report(chapter_audio, sr)
            last_chapter_path = str(out)
            lines.append(
                f"- ✅ Chapitre {index}/{len(chapters)} — {report['duration_sec'] / 60:.1f} min, "
                f"RMS {report['rms_db']:.1f} dBFS → `{out.name}`"
            )
            yield "\n".join(lines), last_chapter_path

        lines.append(f"\n**Terminé.** {cache.stats.describe()}")
        if qc_flagged:
            summary = quality.summarize(qc_flagged)
            (outdir / "qc_report.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            codes = ", ".join(f"{code}×{count}" for code, count in sorted(summary["by_code"].items()))
            lines.append(
                f"\n**Contrôle qualité :** {summary['flagged']} segment(s) signalé(s) "
                f"({codes}), dont {summary['fatal']} non réparé(s) — détail dans "
                f"`{outdir.name}/qc_report.json`."
            )
        elif qc_inspected:
            lines.append(
                f"\n**Contrôle qualité :** {qc_inspected} segment(s) inspecté(s), aucun défaut."
            )
        lines.append(f"\nChapitres dans `{outdir}` — utilisez « Assembler » pour un fichier unique.")
        yield "\n".join(lines), last_chapter_path

    def _book_scan_defects(title):
        """List the segments of a finished book that the quality pass flags."""
        outdir = _book_dir(title)
        try:
            plan = repair.BookPlan.load(outdir)
        except (FileNotFoundError, ValueError) as error:
            raise gr.Error(str(error))

        cache = cache_tools.ChunkCache(outdir / ".cache")
        flagged = repair.flagged_segments(repair.inspect_book(plan, cache))
        if not flagged:
            return (
                gr.update(choices=[], value=None),
                "**Contrôle qualité :** aucun segment à réparer dans ce livre.",
                None,
            )

        choices = [
            f"{label} — {', '.join(issue.code for issue in report.issues)}"
            for label, report in flagged
        ]
        rows = "\n".join(
            f"| `{label}` | {'❌ fatal' if report.severity == quality.FATAL else '⚠️ suspect'} "
            f"| {report.duration_sec:.1f}s | {', '.join(i.detail for i in report.issues)} |"
            for label, report in flagged
        )
        return (
            gr.update(choices=choices, value=choices[0]),
            f"**{len(flagged)} segment(s) signalé(s)**\n\n"
            "| Segment | Gravité | Durée | Détail |\n|---|---|---|---|\n" + rows,
            None,
        )

    def _book_repair_segment(title, choice, qc_retries):
        """Re-roll one flagged segment and restitch only its chapter."""
        if not choice:
            raise gr.Error("Choisissez d'abord un segment à réparer.")

        outdir = _book_dir(title)
        try:
            plan = repair.BookPlan.load(outdir)
            chapter_index, position = repair.parse_label(choice.split(" — ")[0])
        except (FileNotFoundError, ValueError) as error:
            raise gr.Error(str(error))

        cache = cache_tools.ChunkCache(outdir / ".cache")
        spec = plan.voice_spec()
        segment = plan.segment(chapter_index, position)

        def render(seed):
            sample_rate, wav_out, _ = demo.generate_tts_audio(
                text_input=segment.text,
                control_instruction=spec.description,
                cfg_value_input=spec.cfg,
                do_normalize=spec.normalize,
                inference_timesteps=int(spec.steps),
                seed=seed,
            )
            return sample_rate, wav_out

        gr.Info(f"Régénération de {choice.split(' — ')[0]} — une seule génération, pas le chapitre.")
        result = repair.reroll_segment(plan, chapter_index, position, cache, render)

        lines = [
            f"### Réparation de `{result.label}` (essai {result.attempt}, graine `{result.seed}`)\n",
            f"- Texte : « {segment.text[:120]}{'…' if len(segment.text) > 120 else ''} »",
            f"- Avant : {result.previous.describe() if result.previous else '(rien en cache)'}",
            f"- Après : {result.report.describe()}",
        ]
        if not result.improved:
            # Kept the old take on purpose; say so, or the user would think the
            # repair silently did nothing.
            lines.append(
                "\n**Le nouvel essai est moins bon : l'ancien est conservé.** "
                "Relancez pour tirer une autre version."
            )
            return "\n".join(lines), None

        rebuilt = repair.rebuild_chapter(plan, chapter_index, cache, outdir)
        if not rebuilt.ok:
            lines.append(
                f"\n⚠️ Chapitre non reconstruit : segment(s) absent(s) du cache — "
                f"{', '.join(rebuilt.missing)}. Relancez la narration pour les régénérer."
            )
            return "\n".join(lines), None

        lines.append(
            f"\n✅ Chapitre {chapter_index} reconstruit à partir du cache "
            f"({rebuilt.duration_sec / 60:.1f} min) → `{rebuilt.path.name}`"
        )
        return "\n".join(lines), str(rebuilt.path)

    def _book_assemble(title, author, output_format, bitrate=""):
        """Join the generated chapters into one chaptered file."""
        outdir = _book_dir(title)
        chapter_files = sorted(outdir.glob("chapitre_*.wav"))
        if not chapter_files:
            raise gr.Error(f"Aucun chapitre trouvé dans {outdir}. Lancez d'abord la narration.")

        target = outdir / f"{outdir.name}_complet.{output_format}"
        # Written by the EPUB import, if the book carried one.
        covers = sorted(outdir.glob("couverture.*"))
        result = assembly.assemble(
            chapter_files,
            target,
            title=title or outdir.name,
            author=author or "",
            bitrate=bitrate or None,
            cover_path=covers[0] if covers else None,
        )
        message = [
            f"### Assemblage\n",
            f"- {len(result.chapters)} chapitre(s) · **{result.duration_sec / 60:.1f} min**",
            f"- {result.message}",
        ]
        if result.pending_command:
            import subprocess

            message.append(
                "\nffmpeg n'est pas installé. Le WAV complet et les marqueurs de chapitres sont "
                "prêts ; lancez ensuite :\n\n```\n"
                + subprocess.list2cmdline(result.pending_command)
                + "\n```"
            )
        delivered = result.output_path or result.wav_path
        return "\n".join(message), str(delivered)

    def _book_check_delivery(title):
        """Check the finished chapters against what a distributor accepts.

        Reading only — no encoding, because ffmpeg may not be installed and
        because the answer worth having before spending an evening on an
        upload is *whether* the files pass, not the MP3s themselves.
        """
        outdir = _book_dir(title)
        chapter_files = sorted(outdir.glob("chapitre_*.wav"))
        if not chapter_files:
            raise gr.Error(f"Aucun chapitre trouvé dans {outdir}. Lancez d'abord la narration.")

        titles = []
        titles_path = outdir / "titles.txt"
        if titles_path.is_file():
            # utf-8-sig: a titles.txt edited on Windows carries a byte order mark.
            titles = [
                line.strip()
                for line in titles_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]

        profile = delivery.ACX_PROFILE
        limit = delivery.max_seconds_for(profile)
        rows, failing, total_sec = [], 0, 0.0
        for index, path in enumerate(chapter_files, 1):
            data, sample_rate = sf.read(str(path), dtype="float32")
            parts = delivery.split_for_delivery(data, sample_rate, profile)
            name = titles[index - 1] if index <= len(titles) else path.stem
            for part_number, part in enumerate(parts, 1):
                report = delivery.check_delivered(part, sample_rate, profile)
                total_sec += report["duration_sec"]
                suffix = f" (partie {part_number})" if len(parts) > 1 else ""
                reasons = delivery.failures(report)
                if reasons:
                    failing += 1
                rows.append(
                    f"| {name[:40]}{suffix} | {report['duration_sec'] / 60:.1f} | "
                    f"{report['rms_db']:.1f} | {report['head_room_sec']:.2f} | "
                    f"{report['tail_room_sec']:.2f} | "
                    f"{'✅' if not reasons else '❌ ' + ' ; '.join(reasons)} |"
                )

        header = [
            f"### Conformité {profile.name} — {len(rows)} fichier(s), "
            f"{total_sec / 60:.0f} min\n",
            f"MP3 {profile.bitrate_kbps} kbps CBR · {profile.sample_rate} Hz · "
            f"mono · ≤ {limit / 60:.0f} min par fichier\n",
            "| Fichier | min | RMS dBFS | tête s | queue s | Verdict |",
            "|---|---|---|---|---|---|",
        ]
        footer = (
            f"\n**{failing} fichier(s) hors norme.**"
            if failing
            else "\n**Tous les fichiers satisfont la norme.**"
        )
        footer += (
            "\n\nPour produire le dossier à déposer (MP3 encodés + extrait commercial) :\n\n"
            f"```\npython scripts/export_acx.py {outdir}\n```"
        )
        return "\n".join(header + rows) + footer

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        with gr.Tabs():
            with gr.Tab(I18N("tab_studio")):
                gr.Markdown(I18N("usage_instructions"))

                with gr.Row():
                    with gr.Column():
                        reference_wav = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label=I18N("reference_audio_label"),
                        )
                        show_prompt_text = gr.Checkbox(
                            value=False,
                            label=I18N("show_prompt_text_label"),
                            info=I18N("show_prompt_text_info"),
                            elem_classes=["switch-toggle"],
                        )
                        prompt_text = gr.Textbox(
                            value="",
                            label=I18N("prompt_text_label"),
                            placeholder=I18N("prompt_text_placeholder"),
                            lines=2,
                            visible=False,
                        )
                        _default_lang = _PRESET_LANGS[0] if _PRESET_LANGS else None
                        preset_lang = gr.Dropdown(
                            choices=[(_lang_label(c), c) for c in _PRESET_LANGS],
                            value=_default_lang,
                            label=I18N("preset_lang_label"),
                            visible=len(_PRESET_LANGS) > 1,  # only show when there is a choice to make
                        )
                        preset_voice = gr.Dropdown(
                            choices=[PRESET_CUSTOM_LABEL] + _voice_names_for_lang(_default_lang),
                            value=PRESET_CUSTOM_LABEL,
                            label=I18N("preset_voices_label"),
                            info=I18N("preset_voices_info"),
                        )
                        preview_btn = gr.Button(I18N("preview_btn_label"), size="sm")
                        preview_audio = gr.Audio(label=I18N("preview_btn_label"), visible=False)
                        control_instruction = gr.Textbox(
                            value="",
                            label=I18N("control_label"),
                            placeholder=I18N("control_placeholder"),
                            lines=2,
                        )
                        text = gr.Textbox(
                            value=DEFAULT_TARGET_TEXT,
                            label=I18N("target_text_label"),
                            lines=3,
                        )
                        load_txt_btn = gr.UploadButton(
                            I18N("load_txt_label"),
                            file_types=[".txt", ".epub"],
                            size="sm",
                        )

                        with gr.Accordion(I18N("advanced_settings_title"), open=False):
                            DoDenoisePromptAudio = gr.Checkbox(
                                value=False,
                                label=I18N("ref_denoise_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("ref_denoise_info"),
                            )
                            DoNormalizeText = gr.Checkbox(
                                value=False,
                                label=I18N("normalize_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("normalize_info"),
                            )
                            prepare_text = gr.Checkbox(
                                value=False,
                                label=I18N("prepare_text_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("prepare_text_info"),
                            )
                            master_audio = gr.Checkbox(
                                value=True,
                                label=I18N("master_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("master_info"),
                            )
                            enable_chunking = gr.Checkbox(
                                value=True,
                                label=I18N("chunking_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("chunking_info"),
                            )
                            chunk_max_chars = gr.Slider(
                                minimum=100,
                                maximum=600,
                                value=_CHUNK_MAX_CHARS,
                                step=20,
                                label=I18N("chunk_size_label"),
                                info=I18N("chunk_size_info"),
                            )
                            cfg_value = gr.Slider(
                                minimum=1.0,
                                maximum=3.0,
                                value=2.0,
                                step=0.1,
                                label=I18N("cfg_label"),
                                info=I18N("cfg_info"),
                            )
                            dit_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label=I18N("dit_steps_label"),
                                info=I18N("dit_steps_info"),
                            )
                            with gr.Row():
                                seed_value = gr.Number(
                                    value=random.randint(0, 2**32 - 1),
                                    precision=0,
                                    label=I18N("seed_label"),
                                    info=I18N("seed_info"),
                                    interactive=False,
                                )
                                random_seed = gr.Checkbox(
                                    value=True,
                                    label=I18N("random_seed_label"),
                                    elem_classes=["switch-toggle"],
                                    info=I18N("random_seed_info"),
                                )

                        run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

                    with gr.Column():
                        audio_output = gr.Audio(label=I18N("generated_audio_label"))
                        gr.Markdown(I18N("examples_footer"))

            with gr.Tab(I18N("tab_book")):
                gr.Markdown(I18N("book_intro"))

                with gr.Row():
                    with gr.Column():
                        book_upload = gr.UploadButton(
                            I18N("book_file_label"), file_types=[".txt", ".epub"], size="sm"
                        )
                        book_text = gr.Textbox(
                            value="",
                            label=I18N("book_text_label"),
                            lines=14,
                            placeholder="Chapitre premier…\n\n---\n\nChapitre deux…",
                        )
                        with gr.Row():
                            book_title = gr.Textbox(value="", label=I18N("book_title_label"))
                            book_author = gr.Textbox(value="", label=I18N("book_author_label"))
                        with gr.Row():
                            book_language = gr.Dropdown(
                                choices=[("Français", "fr"), ("English", "en")],
                                value="fr",
                                label=I18N("book_language_label"),
                                info=I18N("book_language_info"),
                            )
                            book_narrator = gr.Textbox(
                                value="",
                                label=I18N("book_narrator_label"),
                                info=I18N("book_narrator_info"),
                            )
                            book_with_credits = gr.Checkbox(
                                value=True,
                                label=I18N("book_credits_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("book_credits_info"),
                            )

                        # The book's own voice picker. Deliberately not a mirror
                        # of the Studio one: this dropdown is what the book is
                        # narrated with, so choosing a voice for a book never
                        # means leaving the tab.
                        with gr.Row():
                            book_preset_voice = gr.Dropdown(
                                # Every voice, not the Studio tab's language
                                # filter: a book picked here should never be
                                # silently restricted by a control it cannot see.
                                choices=[PRESET_CUSTOM_LABEL] + _voice_names_for_lang(None),
                                value=PRESET_CUSTOM_LABEL,
                                label=I18N("preset_voices_label"),
                                info=I18N("preset_voices_info"),
                                scale=3,
                            )
                            book_preview_btn = gr.Button(
                                I18N("preview_btn_label"), size="sm", scale=1
                            )
                        book_preview_audio = gr.Audio(
                            label=I18N("preview_btn_label"), visible=False
                        )

                        with gr.Accordion(I18N("book_settings_title"), open=False):
                            book_prepare = gr.Checkbox(
                                value=True,
                                label=I18N("prepare_text_label"),
                                info=I18N("prepare_text_info"),
                                elem_classes=["switch-toggle"],
                            )
                            book_target_rms = gr.Slider(
                                minimum=-30.0,
                                maximum=-12.0,
                                value=audio_tools.MasteringSettings().target_rms_db,
                                step=0.5,
                                label=I18N("book_target_rms_label"),
                                info=I18N("book_target_rms_info"),
                            )
                            book_pause_sentence = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=chunking.PauseProfile().sentence,
                                step=0.05,
                                label=I18N("book_pause_sentence_label"),
                            )
                            book_polish = gr.Checkbox(
                                value=True,
                                label=I18N("book_polish_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("book_polish_info"),
                            )
                            book_pause_paragraph = gr.Slider(
                                minimum=0.0,
                                maximum=3.0,
                                value=chunking.PauseProfile().paragraph,
                                step=0.05,
                                label=I18N("book_pause_paragraph_label"),
                            )
                            book_qc_retries = gr.Slider(
                                minimum=0,
                                maximum=3,
                                value=1,
                                step=1,
                                label=I18N("book_qc_label"),
                                info=I18N("book_qc_info"),
                            )

                        with gr.Row():
                            book_plan_btn = gr.Button(I18N("book_plan_btn"), size="sm")
                            book_run_btn = gr.Button(I18N("book_generate_btn"), variant="primary")

                    with gr.Column():
                        book_status = gr.Markdown(value="")
                        book_audio = gr.Audio(label=I18N("book_audio_label"))
                        with gr.Row():
                            book_format = gr.Dropdown(
                                choices=["m4b", "mp3", "wav"],
                                value="m4b",
                                label=I18N("book_format_label"),
                                scale=1,
                            )
                            book_bitrate = gr.Dropdown(
                                choices=["64k", "96k", "128k", "192k", "256k"],
                                value="64k",
                                label=I18N("book_bitrate_label"),
                                info=I18N("book_bitrate_info"),
                                scale=1,
                            )
                            book_assemble_btn = gr.Button(I18N("book_assemble_btn"), scale=2)
                            book_check_btn = gr.Button(I18N("book_check_btn"), scale=2)

                        with gr.Accordion(I18N("book_repair_title"), open=False):
                            gr.Markdown(I18N("book_repair_info"))
                            book_scan_btn = gr.Button(I18N("book_scan_btn"), size="sm")
                            book_defects = gr.Markdown(value="")
                            book_defect_choice = gr.Dropdown(
                                choices=[], value=None, label=I18N("book_defect_label")
                            )
                            book_repair_btn = gr.Button(
                                I18N("book_repair_btn"), variant="primary", size="sm"
                            )
                            book_repair_status = gr.Markdown(value="")
                        book_output_file = gr.File(label=I18N("book_file_output_label"))

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        random_seed.change(
            fn=_on_random_seed_toggle,
            inputs=[random_seed],
            outputs=[seed_value],
        )

        preset_lang.change(
            fn=_on_lang_change,
            inputs=[preset_lang],
            outputs=[preset_voice],
        )

        preset_voice.change(
            fn=_on_preset_change,
            inputs=[preset_voice],
            outputs=[
                control_instruction,
                seed_value,
                random_seed,
                cfg_value,
                dit_steps,
                DoNormalizeText,
            ],
        )

        load_txt_btn.upload(
            fn=_load_text_file,
            inputs=[load_txt_btn],
            outputs=[text],
        )

        preview_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[preview_audio],
            show_progress=False,
        ).then(
            fn=_preview_voice,
            inputs=[
                control_instruction,
                seed_value,
                cfg_value,
                dit_steps,
                DoNormalizeText,
                preset_voice,
            ],
            outputs=[preview_audio],
            show_progress=True,
        )

        run_btn.click(
            fn=_prepare_seed,
            inputs=[random_seed, seed_value],
            outputs=[seed_value],
            show_progress=False,
        ).then(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                seed_value,
                enable_chunking,
                chunk_max_chars,
                preset_voice,
                prepare_text,
                master_audio,
            ],
            outputs=[audio_output, seed_value],
            show_progress=True,
            api_name="generate",
        )

        book_upload.upload(
            fn=_load_book_file,
            inputs=[book_upload],
            outputs=[book_text, book_title, book_author, book_status],
        )

        book_plan_btn.click(
            fn=_book_plan,
            inputs=[
                book_text,
                chunk_max_chars,
                book_prepare,
                book_pause_sentence,
                book_pause_paragraph,
                book_title,
                book_author,
                book_narrator,
                book_with_credits,
                book_language,
            ],
            outputs=[book_status],
            show_progress=False,
        )

        # The voice comes from the Studio tab, so the seed is settled the same
        # way as for a single generation before narration starts.
        book_run_btn.click(
            fn=_prepare_seed,
            inputs=[random_seed, seed_value],
            outputs=[seed_value],
            show_progress=False,
        ).then(
            fn=_book_narrate,
            inputs=[
                book_text,
                book_title,
                book_author,
                control_instruction,
                cfg_value,
                dit_steps,
                DoNormalizeText,
                seed_value,
                chunk_max_chars,
                book_prepare,
                book_target_rms,
                book_pause_sentence,
                book_pause_paragraph,
                book_preset_voice,
                book_qc_retries,
                book_narrator,
                book_with_credits,
                book_polish,
                book_language,
            ],
            outputs=[book_status, book_audio],
            show_progress=True,
            api_name="narrate_book",
        )

        book_preview_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[book_preview_audio],
            show_progress=False,
        ).then(
            fn=_preview_voice,
            inputs=[
                control_instruction,
                seed_value,
                cfg_value,
                dit_steps,
                DoNormalizeText,
                book_preset_voice,
            ],
            outputs=[book_preview_audio],
            show_progress=True,
            api_name="preview_book_voice",
        )

        book_scan_btn.click(
            fn=_book_scan_defects,
            inputs=[book_title],
            outputs=[book_defect_choice, book_defects, book_repair_status],
            show_progress=True,
            api_name="scan_book_defects",
        )

        book_repair_btn.click(
            fn=_book_repair_segment,
            inputs=[book_title, book_defect_choice, book_qc_retries],
            outputs=[book_repair_status, book_audio],
            show_progress=True,
            api_name="repair_book_segment",
        )

        book_check_btn.click(
            fn=_book_check_delivery,
            inputs=[book_title],
            outputs=[book_status],
            show_progress=True,
            api_name="check_delivery",
        )

        book_assemble_btn.click(
            fn=_book_assemble,
            inputs=[book_title, book_author, book_format, book_bitrate],
            outputs=[book_status, book_output_file],
            show_progress=True,
            api_name="assemble_book",
        )

    return interface


def parse_auth(value: Optional[str]) -> Optional[Tuple[str, str]]:
    """``user:password`` into a pair Gradio can check, or None.

    A colon in the password is fine — only the first one separates.
    """
    if not value:
        return None
    user, separator, password = value.partition(":")
    if not separator or not user or not password:
        raise ValueError("--auth expects user:password")
    return user, password


def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
    load_denoiser: bool = True,
    auth: Optional[Tuple[str, str]] = None,
):
    demo = VoxCPMDemo(model_id=model_id, device=device, load_denoiser=load_denoiser)
    interface = create_demo_interface(demo)
    # Bound to every interface and unauthenticated, this hands a stranger the
    # machine's whole synthesis capacity — and, on a rented GPU, the bill.
    if server_name not in ("127.0.0.1", "localhost") and auth is None:
        logger.warning(
            "Listening on %s without --auth: anyone who can reach this port can "
            "use the model. Bind to 127.0.0.1 and reach it through an SSH tunnel, "
            "or set --auth user:password.",
            server_name,
        )
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
        auth=auth,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address. Use 127.0.0.1 to restrict access to the local machine; "
             "the default 0.0.0.0 exposes the unauthenticated UI/API to the network (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--no-denoiser",
        action="store_true",
        help="Skip loading the ZipEnhancer (ModelScope) denoiser. It is only used to "
             "clean reference audio for cloning; disabling it speeds up startup and "
             "avoids a slow/blocking download — recommended for narration on CPU.",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=os.environ.get("VOXCPM_AUTH", ""),
        metavar="USER:PASSWORD",
        help="Require a login. Essential whenever the port is reachable from "
             "outside the machine — a rented GPU left open is someone else's "
             "synthesis on your bill. Also read from VOXCPM_AUTH, so the "
             "password need not appear in the command line or in shell history.",
    )
    args = parser.parse_args()
    try:
        auth = parse_auth(args.auth)
    except ValueError as error:
        raise SystemExit(str(error))
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
        load_denoiser=not args.no_denoiser,
        auth=auth,
    )
