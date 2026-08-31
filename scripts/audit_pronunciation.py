"""Trouver les mots que le moteur a mal prononcés, sans les écouter.

Le problème d'échelle. Corriger la prononciation demande de savoir quels mots
sonnent faux ; le savoir demande d'écouter ; et personne n'écoutera trois cents
livres. Il faut un arbitre qui ne soit pas une oreille.

Cet arbitre est la reconnaissance vocale. On fait relire par une machine ce
qu'une autre machine vient de dire, et on compare au texte de départ. Là où la
transcription s'écarte de la source, la prononciation est suspecte : Whisper
n'invente pas « Guébrou » s'il a entendu « Gebru ».

Rien n'est à re-narrer pour cela. Le cache de segments garde côte à côte le
texte demandé et l'audio produit — exactement les deux termes de la
comparaison.

**Ce que la méthode ne voit pas.** Whisper corrige ce qu'il entend d'après le
sens : si le moteur dit « ce livres », il transcrira « ces livres », parce que
la grammaire le lui souffle. Les mots grammaticaux échappent donc à l'audit, et
c'est l'oreille qui les attrape — comme « ces » l'a été. En revanche les noms
propres, les sigles, les mots étrangers et les nombres n'ont pas de filet
grammatical : là, la divergence est franche et l'audit les trouve.

    python scripts/audit_pronunciation.py output/book_mon_livre --sample 120
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

MODELE = "openai/whisper-large-v3-turbo"


def mots(texte: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ÿ''-]+", texte or "")


def pliable(mot: str) -> str:
    """Forme comparable : sans accent, sans casse, sans trait d'union.

    Whisper ponctue et accentue à sa façon ; une différence d'accent n'est pas
    une différence de prononciation, et compter les deux ferait crouler le
    rapport sous du bruit.
    """
    plat = unicodedata.normalize("NFKD", mot.lower())
    plat = "".join(c for c in plat if not unicodedata.combining(c))
    return plat.replace("-", "").replace("'", "").replace("'", "")


def charger_cache(directory: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Les paires (texte demandé, audio produit) que le cache garde."""
    cache = directory / ".cache"
    if not cache.is_dir():
        return []
    paires = []
    for j in sorted(cache.glob("*.json")):
        w = j.with_suffix(".wav")
        if not w.exists():
            continue
        try:
            texte = json.loads(j.read_text(encoding="utf-8")).get("text")
        except (OSError, ValueError):
            continue
        if texte:
            paires.append((texte, w))
    return paires


def transcrire(paires, device: str):
    """Faire relire l'audio par Whisper, segment par segment."""
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # Le pipeline() de transformers 5 décode l'audio via torchcodec, dont les
    # DLL réclament un ffmpeg partagé. Le cache est en WAV : soundfile suffit,
    # et rien ne dépend d'un binaire installé.
    proc = WhisperProcessor.from_pretrained(MODELE)
    modele = WhisperForConditionalGeneration.from_pretrained(MODELE).to(device).eval()

    for i, (texte, chemin) in enumerate(paires, 1):
        x, sr = sf.read(str(chemin), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:  # Whisper n'accepte que 16 kHz
            n = int(len(x) * 16000 / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
        # Whisper se charge en demi-précision : lui donner du float32 lève
        # « Input type (float) and bias type (c10::Half) should be the same ».
        entrees = proc(x, sampling_rate=16000, return_tensors="pt").input_features
        entrees = entrees.to(device=device, dtype=modele.dtype)
        with torch.no_grad():
            ids = modele.generate(entrees, language="fr", task="transcribe", max_new_tokens=440)
        yield texte, proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if i % 20 == 0:
            print(f"    {i}/{len(paires)} segments relus", flush=True)


def recoller_sigles(jetons: list[str]) -> list[str]:
    """« T. D. A. H. » redevient « TDAH ».

    Un sigle correctement épelé revient de la transcription en lettres
    séparées. C'est la bonne prononciation, écrite autrement ; le compter comme
    une faute noierait le rapport sous les sigles qui vont bien.
    """
    sortie: list[str] = []
    tampon: list[str] = []
    for j in jetons + [""]:
        if len(j) == 1 and j.isalpha():
            tampon.append(j)
            continue
        if len(tampon) >= 2:
            sortie.append("".join(tampon))
        else:
            sortie.extend(tampon)
        tampon = []
        if j:
            sortie.append(j)
    return sortie


#: Mots que la méthode ne peut pas juger. Whisper corrige ce qu'il entend
#: d'après le sens, donc « ce livres » revient « ces livres » : un mot
#: grammatical n'apparaît dans le rapport que par accident de transcription, et
#: en nombre il le rend illisible. Ceux-là restent l'affaire de l'oreille.
GRAMMATICAUX = set("""
le la les un une des du de d au aux à a et ou ni mais or donc car que qui quoi
dont où ce cet cette ces ceux celle celles il elle ils elles on nous vous je tu
me te se lui leur leurs mon ma mes ton ta tes son sa ses notre nos votre vos
en y est sont était étaient sera seront été être avoir ai as ont avait avaient
pour par sur sous dans vers chez avec sans entre après avant depuis pendant
plus moins très trop peu bien tout tous toute toutes même aussi encore déjà
comme quand si ne pas non oui alors ainsi cela ceci celui
""".split())

#: Les nombres écrits en toutes lettres par le normaliseur reviennent en
#: chiffres de la transcription : « mille neuf cent quatre-vingts » contre
#: « 1980 ». La prononciation est juste, l'orthographe seule diffère.
NOMBRES = set("""
zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize
quatorze quinze seize vingt vingts trente quarante cinquante soixante cent
cents mille milles million millions milliard milliards demi premier première
""".split())


def interessant(mot: str) -> bool:
    """Un mot dont une divergence dit vraiment quelque chose.

    Restreint aux noms propres et aux sigles, et c'est délibéré. Whisper ne
    transcrit pas mot à mot : il paraphrase, supprime une hésitation, reformule
    une tournure. Un mot courant absent de la transcription ne prouve donc rien,
    et l'expérience le confirme — 264 mots signalés sur trois livres, aucun
    revu d'un livre à l'autre, c'est-à-dire que du bruit.

    Un nom propre et un sigle n'ont pas ce filet : la reconnaissance vocale
    n'invente pas « Filiozat » pour « Filliozat » si elle a entendu le nom
    correctement. Là, la divergence est le signal.
    """
    plat = pliable(mot)
    if not plat or len(plat) < 3:
        return False
    if plat in GRAMMATICAUX or plat in NOMBRES:
        return False
    parties = [pliable(p) for p in re.split(r"[-']", mot) if p]
    if parties and all(p in NOMBRES or p in GRAMMATICAUX for p in parties):
        return False
    # Un sigle : au moins deux capitales d'affilée.
    if re.match(r"^[A-ZÀ-Þ]{2,}$", mot):
        return True
    # Un nom propre : capitale initiale, minuscules ensuite. Une majuscule de
    # début de phrase passe aussi, et c'est acceptable — le bruit qu'elle
    # ajoute est borné, là où les mots courants sont sans fin.
    return bool(re.match(r"^[A-ZÀ-Þ][a-zà-ÿ]", mot))


def comparer(source: str, entendu: str) -> list[tuple[str, str]]:
    """Les mots de la source que la transcription ne retrouve pas.

    Comparaison par ensemble plutôt que par alignement : un mot avalé décale
    tout le reste, et on cherche les mots fautifs, pas leur position.
    """
    vus = collections.Counter(pliable(m) for m in recoller_sigles(mots(entendu)))
    manquants = []
    for m in mots(source):
        cle = pliable(m)
        if vus[cle] > 0:
            vus[cle] -= 1
        elif interessant(m):
            manquants.append((m, entendu))
    return manquants


def main() -> int:
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="dossier d'un livre narré (contenant .cache)")
    ap.add_argument("--sample", type=int, default=150,
                    help="nombre de segments à relire (défaut : 150)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min", type=int, default=2,
                    help="occurrences minimales pour figurer au rapport (défaut : 2)")
    ap.add_argument("--json", help="écrire le rapport ici")
    ap.add_argument("--match", type=float, default=0.6, metavar="TAUX",
                    help="part minimale des mots retrouvés pour qu'un segment compte "
                         "(défaut : 0.6). En deçà, la prise est ratée et non mal dite.")
    ap.add_argument("--merge", metavar="FICHIER",
                    help="cumuler dans ce fichier plutôt que d'écrire un rapport isolé. "
                         "Trois cents livres feraient trois cents rapports que personne "
                         "ne lira ; un seul classement, nourri par tous, se relit.")
    args = ap.parse_args()

    d = pathlib.Path(args.directory)
    paires = charger_cache(d)
    if not paires:
        print(f"aucun cache de segments dans {d}. Le livre a-t-il été balayé "
              f"(--keep deliverables) ? L'audit doit tourner avant le balayage.",
              file=sys.stderr)
        return 1

    # Échantillonner régulièrement plutôt qu'au hasard : un livre change de
    # sujet en avançant, et les noms propres n'arrivent pas tous au début.
    # Un segment tronqué a perdu ses mots par troncature, pas par prononciation :
    # le contrôle qualité s'en occupe déjà, et les compter ici ferait remonter
    # des mots parfaitement dits qui n'ont simplement jamais été prononcés.
    rapport_qc = d / "qc_report.json"
    tronques = set()
    if rapport_qc.is_file():
        try:
            details = json.loads(rapport_qc.read_text(encoding="utf-8")).get("details", [])
            tronques = {e["segment"] for e in details
                        if any(i["code"] in ("truncated", "runaway") for i in e.get("issues", []))}
        except (OSError, ValueError):
            pass
    if tronques:
        print(f"{len(tronques)} segment(s) tronqué(s) exclus de l'audit\n")

    pas = max(1, len(paires) // args.sample)
    echantillon = paires[::pas][: args.sample]
    print(f"{len(paires)} segments en cache, {len(echantillon)} relus\n")

    suspects: collections.Counter = collections.Counter()
    exemples: dict[str, str] = {}
    ecartes = 0
    for source, entendu in transcrire(echantillon, args.device):
        manquants = comparer(source, entendu)
        # Un segment dont la transcription s'écarte massivement n'est pas mal
        # prononcé : il est raté, et le contrôle qualité s'en occupe. Le compter
        # ici ferait remonter tous ses mots — « l'enfant », « parent »,
        # « anxiété » — et noierait les vraies trouvailles comme « Filliozat ».
        total = len(mots(source)) or 1
        if len(manquants) / total > 1 - args.match:
            ecartes += 1
            continue
        for mot, contexte in manquants:
            suspects[mot] += 1
            exemples.setdefault(mot, contexte[:110])
    if ecartes:
        print(f"\n{ecartes} segment(s) écarté(s) : transcription trop éloignée "
              f"pour juger d'une prononciation")

    retenus = [(m, n) for m, n in suspects.most_common() if n >= args.min]
    print(f"\n{len(suspects)} mot(s) non retrouvé(s), {len(retenus)} vu(s) au moins {args.min} fois\n")
    for mot, n in retenus[:40]:
        print(f"  {n:>3} × {mot:<26} entendu : « …{exemples[mot][:70]}… »")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({m: {"occurrences": n, "entendu": exemples[m]} for m, n in retenus},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nrapport : {args.json}")

    if args.merge:
        cumul_path = pathlib.Path(args.merge)
        try:
            cumul = json.loads(cumul_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cumul = {}
        for mot, n in suspects.items():
            entree = cumul.setdefault(mot, {"occurrences": 0, "livres": [], "entendu": ""})
            entree["occurrences"] += n
            if d.name not in entree["livres"]:
                entree["livres"].append(d.name)
            entree["entendu"] = entree["entendu"] or exemples.get(mot, "")[:110]
        # Trié par fréquence : un mot vu dans huit livres se corrige avant un
        # mot vu une fois, à temps d'écoute égal.
        ordonne = dict(sorted(cumul.items(), key=lambda kv: -kv[1]["occurrences"]))
        cumul_path.parent.mkdir(parents=True, exist_ok=True)
        cumul_path.write_text(json.dumps(ordonne, ensure_ascii=False, indent=2), encoding="utf-8")
        recurrents = [m for m, v in ordonne.items() if len(v["livres"]) >= 2]
        print(f"\ncumul : {len(ordonne)} mot(s), dont {len(recurrents)} vu(s) dans "
              f"plusieurs livres — {cumul_path}")

    print("\nRien n'est corrigé ici. Les candidats passent par try_pronunciation.py,")
    print("et seul ce qui a été entendu entre dans le lexique.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
