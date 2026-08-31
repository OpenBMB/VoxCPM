#!/bin/bash
# Rapatrie les livres terminés, vérifie la copie, puis libère le pod.
#
# Le volume loué fait 30 Go dont 15 utilisables, et chaque livre pèse ~540 Mo
# une fois purgé de ses WAV : quarante-six livres réclament 25 Go. Sans
# rapatriement en cours de route, la file meurt d'un disque plein vers le
# vingt-septième — après quoi tous les suivants échouent pour une cause qui
# n'a rien à voir avec la narration.
#
# L'ordre compte, et il n'est pas négociable : copier, VÉRIFIER, puis
# seulement effacer. La vérification compare le nombre de fichiers et la
# taille totale, poste par poste. Un livre dont la copie ne correspond pas
# reste sur le pod — mieux vaut un disque qui se remplit qu'un livre perdu.
set -uo pipefail

POD="root@194.26.196.166"
PORT=41114
DIST="/workspace/voxcpm/output"
LOCAL="${1:-/c/Users/PaxHelios/voxcpm-livres}"
# Où atterrit la prise précédente quand une nouvelle la remplace.
ARCHIVE="$LOCAL/avant_correctif"

ssh_pod() { ssh -n -o BatchMode=yes -o ConnectTimeout=15 -p "$PORT" "$POD" "$@"; }

mkdir -p "$LOCAL/acx"

# Un livre n'est rapatriable qu'une fois purgé : la présence de WAV signifie
# que la chaîne n'a pas fini de le traiter.
livres=$(ssh_pod "cd $DIST 2>/dev/null && for d in book_*/; do d=\${d%/}; \
  [ -f \"\$d\"/*.m4b ] 2>/dev/null || continue; \
  [ -z \"\$(ls \$d/*.wav 2>/dev/null)\" ] && echo \$d; done" 2>/dev/null)

[ -z "$livres" ] && { echo "aucun livre prêt à rapatrier"; exit 0; }

rapatries=0
for b in $livres; do
  if [ -f "$LOCAL/${b}_deja" ]; then continue; fi
  echo "── $b"

  # On compare ce qui est comparable : le nombre de fichiers ACX d'un côté et
  # de l'autre, et la taille du M4B à l'octet. « ls *.m4b acx » comptait aussi
  # ses propres en-têtes de section, d'où un décompte faux d'une unité.
  n_acx_dist=$(ssh_pod "ls $DIST/$b/acx | wc -l")
  o_m4b_dist=$(ssh_pod "stat -c%s $DIST/$b/*.m4b")

  # Une narration qui en remplace une autre ne doit pas l'effacer. La version
  # précédente descend d'un cran, numérotée, et reste écoutable : c'est le seul
  # moyen de juger un correctif — on compare deux prises du même chapitre, pas
  # un souvenir et un fichier. Le scp d'après écrase sinon sans rien demander.
  if [ -f "$LOCAL/${b}_complet.m4b" ]; then
    mkdir -p "$ARCHIVE/acx"
    n=1
    while [ -e "$ARCHIVE/${b}_complet_v${n}.m4b" ]; do n=$((n + 1)); done
    mv "$LOCAL/${b}_complet.m4b" "$ARCHIVE/${b}_complet_v${n}.m4b"
    [ -d "$LOCAL/acx/$b" ] && mv "$LOCAL/acx/$b" "$ARCHIVE/acx/${b}_v${n}"
    echo "   version précédente conservée : $(basename "$ARCHIVE")/${b}_complet_v${n}.m4b"
  fi

  mkdir -p "$LOCAL/acx/$b"
  scp -q -P "$PORT" "$POD:$DIST/$b/*.m4b" "$LOCAL/" 2>/dev/null
  scp -q -P "$PORT" "$POD:$DIST/$b/acx/*" "$LOCAL/acx/$b/" 2>/dev/null
  scp -q -P "$PORT" "$POD:$DIST/$b/qc_report.json" "$LOCAL/acx/$b/qc_report.json" 2>/dev/null

  n_acx_loc=$(ls "$LOCAL/acx/$b" 2>/dev/null | grep -cv "^qc_report.json$")
  m4b_loc=$(ls "$LOCAL"/${b}_complet.m4b 2>/dev/null | head -1)
  o_m4b_loc=$([ -n "$m4b_loc" ] && stat -c%s "$m4b_loc" || echo 0)

  if [ "$n_acx_loc" -ne "$n_acx_dist" ] || [ "$o_m4b_loc" != "$o_m4b_dist" ]; then
    echo "   REFUS — rien effacé"
    echo "     ACX : $n_acx_loc copiés / $n_acx_dist attendus"
    echo "     M4B : $o_m4b_loc octets / $o_m4b_dist attendus"
    continue
  fi

  ssh_pod "rm -rf $DIST/$b"
  touch "$LOCAL/${b}_deja"
  rapatries=$((rapatries + 1))
  echo "   $n_acx_loc fichiers ACX + M4B vérifiés à l'octet, libéré sur le pod"
done

echo
echo "rapatriés : $rapatries"
ssh_pod "df -h /workspace | tail -1"
