#!/usr/bin/env python3
"""health_shrink.py <lambda_export.json> <set_in.json> <set_out.json>

Remedy for a small-qT health failure that keeps the set instead of rebuilding it: the divergence
as rT -> 0 is carried by the leading pure-lnrt tower (moments with no rt^m factor and the highest
lnrt power; check_smallqt.leading_tower).  Drop exactly those members, write the reduced set, and
let the pipeline prune / refit / re-export / re-scan it -- the same shrink mechanism the export
already uses for non-convergence, triggered by the health gate.  Exit 2 if there is nothing to
drop (tower empty or the set would fall below 12 moments), so the caller can fall back."""
import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from check_smallqt import leading_tower, unsafe_mask, DGRID

exp, fin, fout = sys.argv[1:4]
e = json.load(open(exp)); names = list(e['moments']); lam = np.array(e['lambda_physical'], float)
s = json.load(open(fin)); cur = list(s['selected_moments'])
P, cP, members = leading_tower(names, lam, DGRID)
drop = [names[k] for k in members]
print(f"  health-shrink: leading pure-lnrt power P={P}, c_P(d) in [{cP.min():+.4g},{cP.max():+.4g}], "
      f"unsafe on {int(unsafe_mask(P, cP).sum())}/{len(cP)} d-points; tower members: {drop}")
new = [m for m in cur if m not in drop]
if not drop or len(new) == len(cur) or len(new) < 12:
    print(f"  health-shrink: nothing to drop (K {len(cur)} -> {len(new)}); caller should fall back"); sys.exit(2)
s['selected_moments'] = new; s['n_selected'] = len(new)
s['source'] = s.get('source', '') + f" | health-shrink: dropped lnrt^{P} tower {drop}"
json.dump(s, open(fout, 'w'), indent=2, ensure_ascii=False)
print(f"  health-shrink: K {len(cur)} -> {len(new)} -> {fout}")
