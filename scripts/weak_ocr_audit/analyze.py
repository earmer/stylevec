"""Merge verdicts, join lineage/parquet, compute rates, write overlay jsonl + report. Read-only on existing files."""
import json, math, random, collections, pandas as pd, numpy as np
SEED=20261011
A='/Volumes/EMC/stylevec_data/audit/weak_ocr_review_20261011/'
RB='/Users/earmercarey/stylevec/reports/bookraw/'
Q='/Volumes/EMC/stylevec_data/processed/pdf_ocr-reviewed-v1/quality_audit/queue.json'
L=json.load(open(A+'lines.json')); PM=json.load(open(A+'pages_meta.json')); DES=json.load(open(A+'design.json'))['design']
queue=json.load(open(Q)); names={e['book']:e['name'] for e in queue}
V={}
for ln in open(A+'verdicts.txt'):
    p=ln.rstrip('\n').split(' ',3); i=int(p[0]); rest=p[3] if len(p)>3 else ''
    d=rest.endswith(' D'); note=rest[:-2] if d else rest
    V[i]=dict(cls=p[1],true=p[2],note=note,differs=d)
assert sorted(V)==list(range(len(L)))
def reviewer(i):
    if i<570: return 'claude'
    if 580<=i<=599 or 620<=i<=629 or 640<=i<=648: return 'haiku_spotchecked_by_claude'
    return 'haiku_unverified'
# lineage -> block -> paragraph text
frag2block={}; 
for l in map(json.loads,open(RB+'books_scanned_lineage.jsonl')):
    for f in l['source_fragment_ids']: frag2block[f]=l['source_block_id']
df=pd.read_parquet(RB+'books_scanned.parquet',columns=['source_block_id','text']); btext=dict(zip(df.source_block_id,df.text))
TOTAL_ROWS=len(df)
rows=[]
for i,l in enumerate(L):
    v=V[i]; r=dict(kind='line',idx=i,book=l['book'],page=l['page'],line_id=l['id'],ocr_text=l['ocr_text'],ocr_score=l['score'],box=l['box'],
        verdict=v['cls'],image_reading=v['true'],ocr_text_differs_from_image=v['differs'] or v['cls']=='char_error',note=v['note'],
        in_strict_row=l['in_strict_row'],in_exclusion_audit=l['in_exclusion_audit'],page_image=l['page_image'],reviewer=reviewer(i),
        label='model_judgment: image-assisted, not human proofread')
    blk=frag2block.get(l['id']); r['strict_source_block_id']=blk
    if blk:
        t=btext.get(blk,''); r['wrong_text_present_in_strict_row']=bool(l['ocr_text'] and l['ocr_text'] in t) if r['ocr_text_differs_from_image'] else None
    rows.append(r)
# page verdicts
pm={(p['book'],p['page']):p for p in PM}
pg=collections.defaultdict(list)
for r in rows: pg[(r['book'],r['page'])].append(r)
prow=[]
for k,rs in pg.items():
    cl=[r for r in rs if r['verdict']=='char_error']; un=[r for r in rs if r['verdict']=='unclear']
    verdict='char_error' if cl else ('unclear' if un else 'artifact_only')
    m=pm[k]
    prow.append(dict(kind='page',book=k[0],page=k[1],verdict=verdict,weak_lines_reviewed=len(rs),weak_count_flagged=m['weak_count'],
        char_error_lines=len(cl),char_error_lines_in_strict_rows=sum(r['in_strict_row'] for r in cl),unclear_lines=len(un),
        total_ocr_lines=m['total_ocr_lines'],retained_lines=m['retained_lines'],inspected_image=m['image'],
        rationale=f"{len(rs)} weak line(s) reviewed: {len(cl)} char_error, {len(un)} unclear, {len(rs)-len(cl)-len(un)} artifact/source_damage"))
open(RB+f'weak_ocr_review_{SEED}.jsonl','w').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows+prow))
# ---- stats
def wilson(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)
books=sorted(DES)
S={}
for b in books:
    lr=[r for r in rows if r['book']==b]; pr=[p for p in prow if p['book']==b]
    ce=[r for r in lr if r['verdict']=='char_error']; un=[r for r in lr if r['verdict']=='unclear']
    S[b]=dict(N=DES[b]['weak_pages'],n=len(pr),census=DES[b]['census'],lines=len(lr),artifact=sum(r['verdict']=='artifact' for r in lr),
      char_error=len(ce),unclear=len(un),sd=sum(r['verdict']=='source_damage' for r in lr),
      ce_pages=sum(p['verdict']=='char_error' for p in pr),un_pages=sum(p['verdict']=='unclear' for p in pr),
      ocr_lines=sum(p['total_ocr_lines'] for p in pr),ret_lines=sum(p['retained_lines'] for p in pr),
      ce_strict=sum(r['in_strict_row'] for r in ce),un_strict=sum(r['in_strict_row'] for r in un))
# distinct corrupted rows per page for bootstrap
def rowset(pages,field):
    out={}
    for p in pages:
        s=set(frag2block[r['line_id']] for r in pg[(p['book'],p['page'])] if r['verdict']==field and r['in_strict_row'])
        out[p['page']]=len(s)
    return out
rng=np.random.default_rng(SEED); B=10000
est_pt=0; boot=np.zeros(B); est_hi_pt=0; boot_hi=np.zeros(B); detail={}
for b in books:
    pr=[p for p in prow if p['book']==b]; n=len(pr); N=S[b]['N']
    ce=np.array([rowset([p],'char_error')[p['page']] for p in pr]); both=ce+np.array([rowset([p],'unclear')[p['page']] for p in pr])
    f=N/n; est_pt+=f*ce.sum(); est_hi_pt+=f*both.sum(); detail[b]=(f*ce.sum(),f*both.sum())
    if S[b]['census']: boot+=ce.sum(); boot_hi+=both.sum()
    else:
        idx=rng.integers(0,n,size=(B,n)); boot+=f*ce[idx].sum(1); boot_hi+=f*both[idx].sum(1)
lo,hi=np.percentile(boot,[2.5,97.5]); lo2,hi2=np.percentile(boot_hi,[2.5,97.5])
# totals
T=lambda k:sum(S[b][k] for b in books)
tl=T('lines'); ta=T('artifact'); tc=T('char_error'); tu=T('unclear'); tsd=T('sd'); tocr=T('ocr_lines'); tret=T('ret_lines')
ce_rows=[r for r in rows if r['verdict']=='char_error']
ce_strict=[r for r in ce_rows if r['in_strict_row']]; ce_excl=[r for r in ce_rows if not r['in_strict_row'] and r['in_exclusion_audit']]
ce_neither=[r for r in ce_rows if not r['in_strict_row'] and not r['in_exclusion_audit']]
wl_strict=sum(r['in_strict_row'] for r in rows); wl_excl=sum((not r['in_strict_row']) and r['in_exclusion_audit'] for r in rows); wl_nei=len(rows)-wl_strict-wl_excl
D_strict=[r for r in rows if r['in_strict_row'] and r['ocr_text_differs_from_image'] and r['verdict']!='char_error']
D_strict_present=[r for r in D_strict if r['wrong_text_present_in_strict_row']]
ce_strict_present=[r for r in ce_strict if r['wrong_text_present_in_strict_row']]
distinct_rows=len(set(frag2block[r['line_id']] for r in ce_strict))
distinct_rows_present=len(set(frag2block[r['line_id']] for r in ce_strict_present))
# step 6
excl=[json.loads(x) for x in open(RB+'books_scanned_exclusions.jsonl')]
def ex(fid): return [e['reason'] for e in excl if fid in e.get('source_fragment_ids',[])]
step6=dict(f19_1_0=ex('ce71b57bb15e:19:1:0'),f20_0_0=ex('ce71b57bb15e:20:0:0'),f19_0_0_block=frag2block.get('ce71b57bb15e:19:0:0'),f18_2_0_block=frag2block.get('ce71b57bb15e:18:2:0'),
   f20_0_1_block=frag2block.get('ce71b57bb15e:20:0:1'))
json.dump(dict(S=S,tl=tl,ta=ta,tc=tc,tu=tu,tsd=tsd,tocr=tocr,tret=tret,est_pt=est_pt,ci=[lo,hi],est_hi_pt=est_hi_pt,ci_hi=[lo2,hi2],
  ce_strict=len(ce_strict),ce_excl=len(ce_excl),ce_neither=len(ce_neither),wl=[wl_strict,wl_excl,wl_nei],D_strict=len(D_strict),D_strict_present=len(D_strict_present),
  ce_strict_present=len(ce_strict_present),distinct_rows=distinct_rows,distinct_rows_present=distinct_rows_present,step6=step6,TOTAL_ROWS=TOTAL_ROWS,
  detail=detail),open(A+'stats.json','w'),ensure_ascii=False,indent=1,default=float)
print(json.dumps(dict(tl=tl,ta=ta,tc=tc,tu=tu,tsd=tsd,tocr=tocr,tret=tret,est=[est_pt,lo,hi],est_hi=[est_hi_pt,lo2,hi2],ce_strict=len(ce_strict),ce_excl=len(ce_excl),ce_nei=len(ce_neither),
   wl=[wl_strict,wl_excl,wl_nei],Dstrict=len(D_strict),Dpresent=len(D_strict_present),cepresent=len(ce_strict_present),distinct=distinct_rows,distinct_present=distinct_rows_present,step6=step6),ensure_ascii=False,indent=1))
for b in books: print(b,S[b])
