#!/bin/bash
# POWHEG_DIR must point at the POWHEG run directory (the one holding pwgevents.lhe and powheg.input).
set -euo pipefail
D=${POWHEG_DIR:?set POWHEG_DIR to the POWHEG run directory}
cd "$D"
[ -f pwgevents-ORIGINAL.lhe ] || cp pwgevents.lhe pwgevents-ORIGINAL.lhe
i=0
# id  renscfact facscfact
while read -r ID RS FS; do
  i=$((i+1))
  echo "=== pass $i: id=$ID renscfact=$RS facscfact=$FS  $(date +%H:%M) ==="
  cp powheg.input.orig_backup powheg.input
  cat >> powheg.input <<EOI

compute_rwgt 1
renscfact $RS
facscfact $FS
lhrwgt_id '$ID'
lhrwgt_descr 'muR=$RS muF=$FS'
lhrwgt_group_name 'scale_variation'
lhrwgt_group_combine 'envelope'
EOI
  ./pwhg_main > rwgt_$ID.log 2>&1
  if [ -s pwgevents-rwgt.lhe ]; then
    mv pwgevents-rwgt.lhe pwgevents.lhe
    echo "  pass $i OK -> weights now: $(grep -m1 -A9 '<rwgt>' pwgevents.lhe | grep -c '<wgt')"
  else
    echo "  pass $i FAILED"; tail -3 rwgt_$ID.log; exit 1
  fi
done <<'VARS'
1002 0.5d0 0.5d0
1003 0.5d0 1d0
1004 1d0 0.5d0
1005 1d0 2d0
1006 2d0 1d0
1007 2d0 2d0
VARS
echo "RWGT CHAIN DONE $(date)"
grep -m1 -A9 '<rwgt>' pwgevents.lhe
