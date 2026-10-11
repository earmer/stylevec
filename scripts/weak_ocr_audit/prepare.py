"""Step 1: stratified sample, render pages @200DPI, crop weak lines, join lineage. Read-only on inputs."""
import json, random, collections, os, sys, fitz
from PIL import Image, ImageDraw, ImageFont
SEED=20261011; PER_BOOK=30
OUT='/Volumes/EMC/stylevec_data/audit/weak_ocr_review_%d'%SEED
Q='/Volumes/EMC/stylevec_data/processed/pdf_ocr-reviewed-v1/quality_audit/queue.json'
RB='/Users/earmercarey/stylevec/reports/bookraw/'
q=json.load(open(Q))
weak=[e for e in q if any(i['kind']=='weak_ocr' for i in e['issues'])]
by=collections.defaultdict(list)
for e in weak: by[e['book']].append(e)
rng=random.Random(SEED); sample=[]; design={}
for b in sorted(by):
    pages=sorted(by[b],key=lambda e:e['page'])
    n=len(pages)
    if n<=PER_BOOK: pick=pages  # census
    else: pick=rng.sample(pages,PER_BOOK)
    design[b]={'weak_pages':n,'sampled':len(pick),'census':n<=PER_BOOK}
    sample+=sorted(pick,key=lambda e:e['page'])
# lineage / exclusions
retained=set()
for l in map(json.loads,open(RB+'books_scanned_lineage.jsonl')): retained.update(l['source_fragment_ids'])
excl=set()
for fn in ['books_scanned_exclusions.jsonl','books_scanned_strict_exclusions.jsonl']:
    for l in map(json.loads,open(RB+fn)): excl.update(l.get('source_fragment_ids',[]))
os.makedirs(OUT+'/pages',exist_ok=True); os.makedirs(OUT+'/crops',exist_ok=True)
docs={}; lines=[]; pages_meta=[]
for e in sample:
    raw=json.load(open(e['raw_json']))
    allines=[(b['id'],ln) for b in raw['blocks'] for ln in b.get('lines',[])]
    wk=[i for i in e['issues'] if i['kind']=='weak_ocr'][0]
    doc=docs.get(e['pdf']) or docs.setdefault(e['pdf'],fitz.open(e['pdf']))
    pg=doc[e['page']-1]; pix=pg.get_pixmap(matrix=fitz.Matrix(200/72,200/72))
    ip=f"{OUT}/pages/{e['book']}_{e['page']:05d}.png"; pix.save(ip)
    nretained=sum(1 for i,_ in allines if i in retained)
    pages_meta.append(dict(book=e['book'],page=e['page'],image=ip,render_wh=[pix.width,pix.height],ocr_wh=[raw['width'],raw['height']],
        total_ocr_lines=len(allines),retained_lines=nretained,weak_count=wk['count'],weak_listed=len(wk['examples'])))
    for ex in wk['examples']:
        lines.append(dict(book=e['book'],page=e['page'],id=ex['id'],ocr_text=ex['text'],score=ex['score'],box=ex['box'],page_image=ip,
          in_strict_row=ex['id'] in retained, in_exclusion_audit=ex['id'] in excl))
json.dump(dict(seed=SEED,per_book=PER_BOOK,design=design),open(OUT+'/design.json','w'),indent=1)
json.dump(pages_meta,open(OUT+'/pages_meta.json','w'),ensure_ascii=False,indent=1)
json.dump(lines,open(OUT+'/lines.json','w'),ensure_ascii=False,indent=1)
print(len(sample),'pages',len(lines),'lines'); print(design)
print('dim mismatches',sum(1 for p in pages_meta if p['render_wh']!=p['ocr_wh']))
print('lines retained',sum(l['in_strict_row'] for l in lines),'excl',sum(l['in_exclusion_audit'] for l in lines))
