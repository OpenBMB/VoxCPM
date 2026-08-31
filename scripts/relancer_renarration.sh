#!/bin/bash
# Remet la file entière en chantier, après le correctif de prononciation.
#
# Pourquoi tout refaire : sur les vingt livres narrés, six seulement ont
# tourné avec la version corrigée de narration/text_fr.py. Les quatorze
# autres portent la troncature sur parenthèse courte, les lignes à remplir
# non effacées et « min » dit « mines ». À l'échelle du catalogue visé, un
# défaut se corrige dans la chaîne, pas dans le fichier — donc on refait.
#
# Les livrables de la passe précédente sont sauvegardés en local
# (~/voxcpm-livres : 20 M4B + 20 dossiers ACX, vérifiés octet pour octet)
# avant que ce script ne les écrase.
set -euo pipefail

cd /workspace/voxcpm

# 1. Archiver l'état avant de le remettre à zéro : sans cette copie, on perd
#    la trace de quel livre avait été narré quand, et donc avec quel code.
if [ -f queue/state.json ]; then
  cp queue/state.json queue/state.avant-renarration.json
  echo "état précédent archivé dans queue/state.avant-renarration.json"
fi
echo '{}' > queue/state.json
echo "état remis à zéro — quarante-six livres, dont vingt et un à refaire"

# 2. Faire de la place. Le disque est le seul endroit où un lot de quarante-six
#    peut échouer sans prévenir : chaque livre monte à ~5 Go de WAV avant que
#    la chaîne ne les efface.
#
#    Aucun cache n'est épargné, et c'est délibéré. L'entrée « ce » → « çe »
#    ajoutée au lexique le 2026-08-12 change le texte de presque tous les
#    segments, donc leur clé de cache : mesuré sur l'introduction de
#    livre-01, 0 réutilisation sur 105 segments. Garder ces gigaoctets
#    reviendrait à conserver un index qui ne pointe plus sur rien.
cd output
rm -rf book_* temoin_* essai_* prononciation
cd ..
echo "libre après purge : $(df -h /workspace | tail -1 | awk '{print $4}')"

# 3. Relancer, détaché, avec exactement les options de la passe précédente.
#    Environnement nu et LANG explicite : un shell ouvert par ssh n'hérite
#    d'aucune locale, et ffmpeg renvoie des titres accentués qui font tomber
#    l'assemblage APRÈS une narration réussie.
# Le journal porte l'histoire de ce qui a été narré, quand, et en combien de
# temps — le seul endroit où se lit le coût réel. L'écraser à chaque relance
# efface cette mémoire : on l'archive, et le nouveau s'ajoute à la suite.
if [ -s /workspace/queue.log ]; then
  cat /workspace/queue.log >> /workspace/queue-historique.log
fi

# Le répertoire de travail DOIT être le dépôt.
#
# narrate_queue.py construit ses chemins en relatif — « output/book_<slug> ».
# Lancé depuis /workspace, il cherchait donc /workspace/output, qui n'existe
# pas : ses sous-processus, eux, tournent avec cwd=REPO et écrivaient au bon
# endroit. Résultat, la file voyait zéro chapitre là où vingt existaient, ne
# purgeait rien, et annonçait « terminé — 0 chapitre(s) » sur un livre entier
# et valide. Un livre laissait alors 5,5 Go au lieu de 0,45 : le volume
# saturait au troisième.
cd /workspace/voxcpm

setsid nohup env LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  ./.venv/bin/python \
  scripts/narrate_queue.py \
  queue/queue_catalogue.json \
  --device cuda --keep deliverables --no-synthetic-disclosure --audit 120 \
  > /workspace/queue.log 2>&1 < /dev/null &

sleep 5
echo
echo "file relancée — journal : /workspace/queue.log"
head -3 /workspace/queue.log 2>/dev/null || true
