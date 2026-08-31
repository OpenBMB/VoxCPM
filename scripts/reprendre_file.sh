#!/bin/bash
# Reprend la file là où elle s'est arrêtée, sans rien remettre à zéro.
#
# À distinguer de relancer_renarration.sh, qui efface l'état et les livrables
# pour tout refaire. Ici, un livre marqué « done » dans queue/state.json est
# sauté et son M4B conservé : c'est ce qu'il faut après une interruption —
# panne, correctif déployé à chaud, ou arrêt volontaire.
#
# Le répertoire de travail DOIT être le dépôt. narrate_queue.py construit ses
# chemins en relatif (« output/book_<slug> ») ; lancé d'ailleurs, il ne
# retrouve pas les livres qu'il vient de produire, annonce « 0 chapitre(s) »
# et surtout ne purge plus les WAV — un livre laisse alors 5,5 Go au lieu de
# 0,45 et le volume sature au troisième.
set -euo pipefail

cd /workspace/voxcpm

if pgrep -f "narrate_queue.py" > /dev/null; then
  echo "une file tourne déjà — rien fait" >&2
  exit 1
fi

if [ -s /workspace/queue.log ]; then
  cat /workspace/queue.log >> /workspace/queue-historique.log
fi

setsid nohup env LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  ./.venv/bin/python \
  scripts/narrate_queue.py \
  queue/queue_catalogue.json \
  --device cuda --keep deliverables --no-synthetic-disclosure --audit 120 \
  > /workspace/queue.log 2>&1 < /dev/null &

sleep 5
echo "file reprise — journal : /workspace/queue.log"
head -4 /workspace/queue.log 2>/dev/null || true
