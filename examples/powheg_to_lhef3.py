"""Convert POWHEG old-format '#new weight,...' lines into LHEF3 <rwgt> blocks
so Pythia parses them natively. Streaming, order-preserving."""
import sys, shutil
IDS=['1002','1003','1004','1005','1006','1007']
src,dst=sys.argv[1],sys.argv[2]
decl="".join(f"<weight id='{i}'> scale variation {i} </weight>\n" for i in IDS)
nev=0; buf=[]; inev=False; nw=0; bad=0
with open(src) as fi, open(dst,'w') as fo:
    for line in fi:
        if line.startswith("<weightgroup name='scale_variation'"):
            fo.write(line); fo.write(decl); continue
        if line.startswith('<event>'):
            inev=True; buf=[]; fo.write(line); continue
        if inev and line.startswith('#new weight'):
            buf.append(line.split()[2]); continue          # weight value
        if line.startswith('</event>'):
            if len(buf)==len(IDS):
                fo.write('<rwgt>\n')
                for i,v in zip(IDS,buf): fo.write(f"<wgt id='{i}'> {v} </wgt>\n")
                fo.write('</rwgt>\n'); nw+=1
            else: bad+=1
            fo.write(line); inev=False; nev+=1
            if nev % 200000 == 0: print(f"  {nev:,} events",flush=True)
            continue
        fo.write(line)
print(f"events={nev:,}  with {len(IDS)} weights={nw:,}  incomplete={bad:,}")
