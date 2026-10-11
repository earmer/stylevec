import json,sys
OUT='/Volumes/EMC/stylevec_data/audit/weak_ocr_review_20261011'
L=json.load(open(OUT+'/lines.json'))
# usage: addv.py lo hi  then special "idx:class:true:note" args ; others default artifact '。' if ocr=='O' else must be specified
lo,hi=int(sys.argv[1]),int(sys.argv[2]); sp={}
for a in sys.argv[3:]:
    i,c,t,n=a.split(':',3); sp[int(i)]=(c,t,n)
out=[]
for i in range(lo,hi):
    o=L[i]['ocr_text']
    if i in sp: c,t,n=sp[i]
    elif o in ('O','0'): c,t,n='artifact','。','。 read as '+o
    else: raise SystemExit(f'need verdict for {i} {o!r}')
    D='' if t in('-',o) else ' D'
    out.append(f"{i} {c} {t} {n}{D}")
open(OUT+'/verdicts.txt','a').write("\n".join(out)+"\n"); print(len(out),'added')
