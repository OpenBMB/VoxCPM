"""Essayer plusieurs orthographes d'un mot et écouter laquelle se dit juste.

Le lexique de `conf/pronunciation_fr.json` sait déjà remplacer un mot par une
orthographe qui se prononce mieux. Ce qu'il manquait, c'est le moyen de savoir
*laquelle* — et son propre commentaire le dit : « écoutez d'abord, une
correction inutile ne peut que dégrader ».

Deviner coûte cher. Une correction fausse appliquée à vingt livres, ce sont
vingt livres à refaire, et personne ne s'en aperçoit avant la livraison. Deux
minutes de GPU et une écoute règlent la question.

Le script prend une phrase de test et des candidats, produit un wav par
candidat, nommé pour qu'on sache lequel on écoute, et écrit un `index.txt`.
Il ne décide rien : il donne à entendre.

    python scripts/try_pronunciation.py \
        --phrase "Ces livres-là sont ses préférés, et ces pages-ci aussi." \
        --mot ces --candidats "cés" "sés" "cè" \
        --voice "Aurore — livre audio" --device cuda

Le premier fichier produit est toujours le texte d'origine, non modifié : sans
lui on compare des corrections entre elles sans savoir si l'une d'elles est
seulement meilleure que le défaut de départ.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Lancé par `python scripts/x.py`, sys.path[0] est scripts/, pas la racine :
# même idiome que narrate_book.py et pregenerate_previews.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrase", required=True, help="phrase de test contenant le mot")
    ap.add_argument("--mot", required=True, help="mot dont la prononciation est douteuse")
    ap.add_argument("--candidats", nargs="+", required=True,
                    help="orthographes à essayer à la place du mot")
    ap.add_argument("--voice", required=True, help="voix prédéfinie")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="output/prononciation")
    args = ap.parse_args()

    import app  # lourd (torch) : après argparse, pour que --help reste instantané

    voice = next((v for v in app.PRESET_VOICES if v["name"] == args.voice), None)
    if voice is None:
        noms = ", ".join(v["name"] for v in app.PRESET_VOICES)
        print(f"voix inconnue : {args.voice}\ndisponibles : {noms}", file=sys.stderr)
        return 1

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    def remplacer(phrase: str, mot: str, par: str) -> str:
        return re.sub(rf"\b{re.escape(mot)}\b", par, phrase, flags=re.IGNORECASE)

    # L'original d'abord : c'est le défaut qu'on cherche à battre.
    essais = [("00_original", args.phrase)]
    for i, cand in enumerate(args.candidats, 1):
        sain = re.sub(r"[^0-9A-Za-zÀ-ÿ-]+", "_", cand).strip("_") or f"cand{i}"
        essais.append((f"{i:02d}_{sain}", remplacer(args.phrase, args.mot, cand)))

    demo = app.VoxCPMDemo(device=args.device, load_denoiser=False)
    import soundfile as sf

    lignes = []
    for nom, texte in essais:
        print(f"  {nom} : {texte}", flush=True)
        sr, wav, _ = demo.generate_tts_audio(
            text_input=texte,
            control_instruction=voice.get("description") or "",
            cfg_value_input=voice.get("cfg", 2.0),
            do_normalize=voice.get("normalize", True),
            inference_timesteps=int(voice.get("diffusion_steps", 10)),
            seed=voice["seed"],
            reference_wav_path_input=voice.get("reference") or None,
            prompt_text=voice.get("reference_text") or "",
            denoise=False,
        )
        chemin = out / f"{nom}.wav"
        sf.write(str(chemin), wav, sr)
        lignes.append(f"{chemin.name}\t{texte}")

    (out / "index.txt").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"\n{len(essais)} extrait(s) dans {out}")
    print("Écoutez 00_original d'abord : si le mot y est déjà juste, ne corrigez rien.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
