"""Narrer une file de livres, sans surveillance, et se relever tout seul.

Un livre demande trois heures de GPU. Vingt en demandent soixante-cinq, soit
deux jours et demi pendant lesquels personne ne regarde. Ce script existe pour
que cette absence ne coûte rien.

Trois principes, tous appris à la dure :

**Un livre qui échoue n'arrête pas les autres.** Sa cause est notée, il est
marqué en échec, on passe au suivant. Rien n'est plus coûteux qu'une file de
vingt livres bloquée à la troisième heure par le second.

**Rien n'est refait.** Un livre déjà terminé est sauté ; un livre interrompu
reprend au segment près, parce que le cache de ``narrate_book.py`` survit à
tout. Relancer ce script après une coupure ne coûte que ce qui manquait.

**La réparation fait partie du rendu, pas de l'après.** Une fois le livre
narré, les segments jugés fatals — silencieux, tronqués, emballés — sont
re-générés un par un avec un seed dérivé, et leur chapitre est recousu depuis
le cache. Renarrer le chapitre entier pour une phrase serait absurde.

    python scripts/narrate_queue.py queue/queue.json --device cuda
"""
from __future__ import annotations

import argparse
import os
import shutil
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
PYTHON = sys.executable

sys.path.insert(0, str(REPO))

from narration import couverture  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


#: Le nom sous lequel chaque voix est créditée dans les génériques. Un
#: catalogue lu par la même voix mérite qu'on la nomme, comme un éditeur
#: crédite une voix virtuelle — la mention « voix de synthèse » reste dite en
#: plus du nom, jamais à sa place.
VOICE_NAMES = {
    "Aurore — livre audio": "Aurore Cabonet",
    "Aurore — méditation guidée": "Aurore Cabonet",
    "Alex Somerset": "Gabriel Adam",
}


def child_env() -> dict:
    """L'environnement des étapes, forcé en UTF-8.

    Lancé par ``ssh machine 'commande'``, le runner hérite d'un shell non
    interactif où ``LANG`` n'est pas défini. Python y résout alors l'encodage
    préféré en ASCII, et le premier titre de chapitre accentué fait tomber une
    étape — après trois heures de narration réussie. Les appels sensibles
    nomment déjà leur encodage ; ceci couvre ceux qu'on aurait manqués.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    return env


def run(cmd: list[str], logfile: pathlib.Path | None = None) -> tuple[int, str]:
    """Lancer une étape. Sa sortie va dans un fichier, pas en mémoire."""
    if logfile:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=child_env())
        tail = logfile.read_text(encoding="utf-8", errors="replace")[-600:]
        return p.returncode, tail
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=REPO, env=child_env())
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def commande(b: dict, txt: pathlib.Path, outdir: pathlib.Path, args) -> list[str]:
    """La commande de narration d'un livre — une seule, pour les deux passages.

    Le pré-vol la relance avec ``--dry-run``. Il faut donc qu'il vole *le même*
    plan : quand il se contentait d'une commande réduite (ni ``--assemble``, ni
    ``--cover``, ni titre), il validait un livre que la vraie prise assemblait
    autrement — et la vérification de couverture, qui ne s'arme qu'à
    l'assemblage, ne s'y déclenchait jamais.
    """
    cmd = [PYTHON, "scripts/narrate_book.py", str(txt), "--voice", b["voice"],
           "--device", args.device, "--outdir", str(outdir),
           "--qc-retries", args.qc_retries,
           "--assemble", "m4b", "--export-acx"]
    nom_voix = b.get("voice_name") or VOICE_NAMES.get(b["voice"], "")
    if nom_voix:
        cmd += ["--voice-name", nom_voix]
    # Choix d'éditeur, porté par la file plutôt que codé ici : la mention
    # de voix de synthèse est exigée par les plateformes, et la retirer
    # doit rester une décision visible dans les données.
    if b.get("no_synthetic_disclosure") or args.no_synthetic_disclosure:
        cmd += ["--no-synthetic-disclosure"]
    # Un livre peut avoir ses propres abréviations. Le lexique général est
    # passé d'abord, le sien ensuite : ils s'empilent, il ne le remplace pas.
    lexiques = ["conf/pronunciation_fr.json"] + list(b.get("lexicons") or [])
    for lex in lexiques:
        cmd += ["--lexicon", lex]
    # Sans titre, narrate_book retombe sur le nom du fichier : cinq livres
    # se sont annoncés « livre-un-esprits-reprogrammes » avant qu'on le
    # remarque. Un .txt ne porte pas de métadonnées, donc la file les porte.
    for option, cle in (("--title", "title"), ("--author", "author"),
                        ("--cover", "cover")):
        if b.get(cle):
            cmd += [option, b[cle]]
    return cmd


def couvertures_manquantes(books: list[dict]) -> list[tuple[str, str]]:
    """Les livres de la file dont la couverture ne serait pas déposable.

    Posée avant le premier livre, et non au fil de l'eau : découvrir au
    trente-deuxième livre que le quarantième n'a pas de couverture, c'est
    l'apprendre quatre jours trop tard. Lire un en-tête d'image coûte le prix
    d'un ``ls``, donc toute la file y passe.
    """
    manquantes = []
    for b in books:
        chemin = b.get("cover")
        # La file porte des chemins relatifs, et les sous-processus tournent
        # avec cwd=REPO : on résout comme eux, pour ne pas refuser une
        # couverture que la narration, elle, trouverait.
        if chemin and not pathlib.Path(chemin).is_absolute():
            chemin = REPO / chemin
        verdict = couverture.inspecter(chemin)
        if not verdict.conforme:
            manquantes.append((b["slug"], verdict.raison))
    return manquantes


def livrables_absents(outdir: pathlib.Path, wavs: int) -> list[str]:
    """Ce qui manque à un livre pour être déposable, nommé.

    On regarde les fichiers, pas le code de sortie de qui les fabrique : un
    livre lancé depuis le mauvais dossier s'était déclaré terminé sans qu'un
    seul chapitre soit trouvé. Un MP3 dans ``acx/`` plutôt que le dossier seul,
    parce que sans ffmpeg l'export y laisse des WAV et un ``encoder.txt`` —
    de quoi encoder, pas de quoi déposer.
    """
    acx = outdir / "acx"
    livrables = {
        "M4B": bool(list(outdir.glob("*_complet.m4b"))
                    + list(outdir.glob("*_complet.m4a"))),
        "chapitres narrés": wavs > 0,
        "export ACX (MP3)": acx.is_dir() and any(acx.glob("*.mp3")),
    }
    return [nom for nom, present in livrables.items() if not present]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", help="queue.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outroot", default="output")
    ap.add_argument("--qc-retries", default="2")
    ap.add_argument("--only", type=int, help="ne traiter que les N premiers")
    ap.add_argument("--skip-repair", action="store_true")
    ap.add_argument("--audit", type=int, metavar="N", default=0,
                    help="relire N segments par livre avec la reconnaissance vocale et "
                         "cumuler les mots suspects dans queue/prononciation_a_valider.json "
                         "(0 = ne pas auditer)")
    ap.add_argument("--no-synthetic-disclosure", action="store_true",
                    help="Retirer la mention « voix de synthèse » de tous les génériques")
    ap.add_argument("--sans-couverture", action="store_true",
                    help="Narrer même les livres dont la couverture serait "
                         "refusée au dépôt (à n'utiliser que pour un essai)")
    ap.add_argument("--keep", choices=("all", "deliverables"), default="all",
                    help="all : tout garder. deliverables : ne garder que le M4B, "
                         "l'export ACX et le rapport, et effacer les WAV de chapitre "
                         "une fois le livre assemblé (un livre pèse ~3 Go de WAV)")
    args = ap.parse_args()

    qpath = pathlib.Path(args.queue).resolve()
    books = json.loads(qpath.read_text(encoding="utf-8"))
    if args.only:
        books = books[: args.only]

    state_path = qpath.parent / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    def save() -> None:
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    logdir = qpath.parent / "logs"
    logdir.mkdir(exist_ok=True)

    log(f"file de {len(books)} livre(s) — {sum(b['chars'] for b in books)} caractères")

    # Les couvertures d'abord, toutes, et avant la première minute de GPU.
    # Un livre déjà terminé sera sauté : sa couverture ne peut plus rien
    # coûter, et le faire échouer bloquerait une reprise pour rien.
    manquantes = couvertures_manquantes(
        [b for b in books if state.get(b["slug"], {}).get("status") != "done"]
    )
    if manquantes:
        for slug, raison in manquantes:
            log(f"couverture refusée — {slug} : {raison}")
        if not args.sans_couverture:
            log(f"{len(manquantes)} livre(s) sans couverture déposable — file non lancée")
            log("    corrigez la file, ou assumez-le avec --sans-couverture")
            return 2
        log(f"{len(manquantes)} livre(s) narrés sans couverture (--sans-couverture)")

    for i, b in enumerate(books, 1):
        slug = b["slug"]
        st = state.get(slug, {})
        if st.get("status") == "done":
            log(f"[{i}/{len(books)}] {slug} : déjà terminé, sauté")
            continue

        # Trois heures de narration meurent mal sur un disque plein : le livre
        # est perdu et le suivant l'est aussi. S'arrêter avant coûte une
        # relance, pas une nuit.
        free_gb = shutil.disk_usage(REPO).free / 1e9
        if free_gb < 5:
            log(f"seulement {free_gb:.1f} Go libres — arrêt avant {slug}")
            log("    rapatriez les livres produits, puis relancez : la file reprend ici")
            break

        txt = qpath.parent / b["txt"]
        outdir = pathlib.Path(args.outroot) / f"book_{slug.replace('-', '_')}"
        blog = logdir / f"{slug}.log"
        state[slug] = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "voice": b["voice"], "chars": b["chars"]}
        save()

        log(f"[{i}/{len(books)}] {slug} — {b['chars']} car., voix « {b['voice']} »")

        # Une reprise écrit par-dessus la production précédente, mais seulement
        # là où les noms coïncident : un chapitre renommé, un découpage qui
        # bouge d'un fichier, et l'ancien reste. « Le Pouvoir Silencieux » est
        # ressorti avec vingt-huit fichiers ACX pour vingt chapitres — sept
        # rescapés du 8 août, portant des titres à peine différents. Le M4B
        # était sain ; le jeu qu'on livre à une plateforme ne l'était pas.
        #
        # On efface donc les livrables de la prise précédente — et eux seuls.
        # `.cache` reste : c'est lui qui permet à une narration interrompue de
        # reprendre où elle en était plutôt que de recommencer trois heures.
        if outdir.exists():
            efface = 0
            for item in outdir.iterdir():
                if item.name == ".cache":
                    continue
                efface += (item.stat().st_size if item.is_file()
                           else sum(f.stat().st_size for f in item.rglob("*") if f.is_file()))
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            if efface:
                log(f"    livrables précédents effacés ({efface / 1e9:.1f} Go), cache conservé")

        # Pré-vol : il ne charge pas le modèle, donc il coûte des secondes et
        # attrape ce qui ferait échouer trois heures plus tard.
        rc, out = run(commande(b, txt, outdir, args) + ["--dry-run"])
        if rc != 0:
            log(f"    pré-vol refusé — livre écarté")
            state[slug].update(status="failed", stage="dry-run", detail=out[-400:])
            save()
            continue

        t0 = time.time()
        cmd = commande(b, txt, outdir, args)
        rc, tail = run(cmd, blog)
        mins = (time.time() - t0) / 60
        if rc != 0:
            log(f"    narration échouée après {mins:.0f} min — voir {blog.name}")
            state[slug].update(status="failed", stage="narration", minutes=round(mins, 1),
                               detail=tail[-400:])
            save()
            continue
        log(f"    narré en {mins:.0f} min")

        # Réparation ciblée : un segment fatal ne justifie pas de refaire son
        # chapitre, encore moins le livre.
        repaired = fatal = 0
        if not args.skip_repair:
            rc, out = run([PYTHON, "scripts/repair_segment.py", str(outdir), "--list"])
            # Le marqueur est « FATAL » en capitales (repair_segment.py:68).
            # Chercher « fatal » sans distinction de casse compterait aussi la
            # phrase « Aucun segment fatalement défectueux », donc trouverait un
            # défaut dans tout livre sain et chargerait le modèle pour rien.
            fatal = sum(1 for line in out.splitlines() if "FATAL " in line)
            if fatal:
                log(f"    {fatal} segment(s) fatal(s) — réparation")
                rc, out = run([PYTHON, "scripts/repair_segment.py", str(outdir),
                               "--all-fatal", "--device", args.device])
                blog.open("a", encoding="utf-8").write(out)
                # « réparé » disait jusqu'ici « fatal trouvé », ce qui promettait
                # des corrections qui n'avaient pas eu lieu : repair_segment
                # conserve l'ancienne prise quand la nouvelle est pire, et il le
                # dit. Compter ce qu'il dit plutôt que ce qu'on espérait.
                conserves = out.count("le nouvel essai est moins bon")
                repaired = max(0, fatal - conserves)
                if conserves:
                    log(f"    {repaired} amélioré(s), {conserves} inchangé(s) "
                        f"(la nouvelle prise était pire)")
                elif rc != 0:
                    log(f"    réparation incomplète (code {rc})")

        # L'audit doit passer AVANT le balayage : il lit le cache de segments,
        # que --keep deliverables efface. Il cumule dans un classement unique —
        # trois cents rapports isolés ne seraient jamais relus, un seul l'est.
        if args.audit:
            log(f"    audit de prononciation ({args.audit} segments)")
            run([PYTHON, "scripts/audit_pronunciation.py", str(outdir),
                 "--sample", str(args.audit), "--device", args.device,
                 "--merge", str(qpath.parent / "prononciation_a_valider.json")], blog)

        wavs = len(list(outdir.glob("*.wav"))) if outdir.is_dir() else 0

        # « terminé » doit vouloir dire « déposable », parce que « terminé » ne
        # se rejoue pas : `state.json` saute ces livres-là pour toujours. La
        # narration a rendu zéro, mais rendre zéro n'a jamais garanti qu'un
        # fichier existe — un livre lancé depuis le mauvais dossier s'était
        # ainsi déclaré terminé sans un seul chapitre trouvé. On regarde donc
        # les livrables eux-mêmes plutôt que le code de sortie de qui les
        # fabrique.
        absents = livrables_absents(outdir, wavs)
        if absents:
            log(f"    narration rendue sans livrable — manque : {', '.join(absents)}")
            log("    livre marqué en échec plutôt que terminé, pour qu'une "
                "reprise le retrouve")
            state[slug].update(status="failed", stage="livraison",
                               minutes=round(mins, 1), chapters_wav=wavs,
                               detail="livrables manquants : " + ", ".join(absents))
            save()
            continue

        # Un livre laisse ~3 Go de WAV de chapitre derrière lui. Vingt livres
        # saturent le volume au quatrième, et une file qui meurt d'un disque
        # plein a produit dix-neuf échecs pour une cause qui n'a rien à voir
        # avec la narration. Les WAV ne partent qu'une fois le M4B et l'export
        # ACX écrits : le livrable existe avant que le master ne disparaisse.
        freed = 0
        if args.keep == "deliverables" and wavs:
            m4b = list(outdir.glob("*.m4b")) + list(outdir.glob("*.m4a"))
            acx = outdir / "acx"
            if m4b and acx.is_dir() and any(acx.iterdir()):
                for w in outdir.glob("*.wav"):
                    freed += w.stat().st_size
                    w.unlink()
                # Le gros morceau est le cache de segments : 1680 fichiers,
                # 1,6 Go par livre, dans un dossier en point que le premier
                # balayage ne voyait pas — il cherchait « seg_*.wav » alors que
                # le cache nomme ses entrées autrement. À 1,6 Go le livre, vingt
                # livres réclament 32 Go sur un volume qui en fait 30.
                #
                # Le prix à payer : sans ce cache, repair_segment.py ne peut
                # plus retoucher le livre. C'est acceptable parce que le
                # contrôle qualité et la réparation ont déjà eu lieu, juste
                # au-dessus, et que le M4B et l'export ACX sont écrits.
                cache_dir = outdir / ".cache"
                if cache_dir.is_dir():
                    freed += sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                    shutil.rmtree(cache_dir, ignore_errors=True)
                log(f"    {freed/1e9:.1f} Go de WAV effacés (M4B et ACX conservés)")
            else:
                log(f"    WAV conservés : M4B ou export ACX manquant, rien n'est effacé")

        state[slug].update(status="done", minutes=round(mins, 1), chapters_wav=wavs,
                           fatal_found=fatal,
                           repaired=repaired, freed_gb=round(freed / 1e9, 2),
                           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        save()
        log(f"    terminé — {wavs} chapitre(s), {repaired} segment(s) réparé(s)")

    done = sum(1 for v in state.values() if v.get("status") == "done")
    failed = [k for k, v in state.items() if v.get("status") == "failed"]
    log(f"file terminée — {done} livre(s) produits, {len(failed)} en échec")
    for k in failed:
        log(f"    échec : {k} ({state[k].get('stage')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
