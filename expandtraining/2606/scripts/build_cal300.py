import json,random,re,sqlite3,collections,urllib.request,numpy as np,pandas as pd
R='outputs/traits/';rd=lambda p:[json.loads(l) for l in open(p)]
rng=random.Random(300)
CL=['本大爷','本座','老子','小的','小人','在下','老朽','晚辈','奴家','禀大人']
clean=lambda r:not any(c in r['text'] and c not in r['source'] for c in CL)
P=[]  # (subset, tag, A, B)
# ---- (a) 60 zh rewrites
book=[r for r in rd(R+'../bookstyle/verify.jsonl') if r['lang']=='zh' and r['kept'] and clean(r) and 20<=len(r['source'])<=220]
tone=[r for r in rd(R+'../tonetwin/verify.jsonl') if r['lang']=='zh' and r['kept'] and clean(r)]
plan_b={'terse':3,'ornate':3,'classical':3,'trendy':2,'casual_chat':3,'ceremonial':3,'slang':2,'teasing':2,'deadpan':2,'childlike':3,'spontaneous':3,'theatrical':2,'rough':2,'cold':2,'warm':3}
plan_t={'tentative':3,'categorical':2,'sarcastic':3,'earnest':2,'cheerful':2,'gloomy':2,'outburst':2,'flat':2}
for src,plan,tag in((book,plan_b,'book'),(tone,plan_t,'tone')):
    for st,n in plan.items():
        c=[r for r in src if r['style']==st]; assert len(c)>=n,(st,len(c))
        for r in rng.sample(c,n): P.append(('zh_rewrite',f'{tag}:{st}',r['source'],r['text']))
p28=json.load(open(R+'pilot28/pairs.json'))[10:]
for t,a,b in p28: P.append(('zh_rewrite',f'pilot28:{t}',a,b))
assert len(P)==60,len(P)
# ---- (b) 45 douban
def embed(texts):
    out=[]
    for i in range(0,len(texts),64):
        req=urllib.request.Request('http://localhost:11434/api/embed',json.dumps({'model':'embeddinggemma:latest','input':texts[i:i+64]}).encode(),{'Content-Type':'application/json'})
        out+=json.load(urllib.request.urlopen(req,timeout=300))['embeddings']
    E=np.array(out);return E/np.linalg.norm(E,axis=1,keepdims=True)
bucket=lambda s:'pos' if s>=4 else('neu' if s==3 else 'neg')
MARK=['不','没','别','但','却','虽','然而','可是','并非','才不']
hasm=lambda t:any(m in t for m in MARK)
def mine(df,thr,sel_big,sel_rand):
    E=embed(list(df.Comment));cand=[]
    for m,g in df.groupby('Movie_Name_CN'):
        ix=g.index.values;S=E[ix]@E[ix].T
        for a in range(len(ix)):
            for b in range(a+1,len(ix)):
                if S[a,b]>=thr:
                    i,j=ix[a],ix[b]
                    if bucket(df.Star[i])==bucket(df.Star[j]) and hasm(df.Comment[i])==hasm(df.Comment[j]) and df.Comment[i]!=df.Comment[j]:
                        cand.append((S[a,b],i,j))
    cand.sort(key=lambda x:-x[0]);used=set();keep=[]
    for c,i,j in cand:
        if i in used or j in used: continue
        used|={i,j};keep.append((c,i,j))
    print('candidates',len(cand),'cap1',len(keep))
    feat=lambda t:(len(t),len(re.findall('[！!？?]',t)))
    def spread(x):
        a,b=feat(df.Comment[x[1]]),feat(df.Comment[x[2]]);return abs(np.log(a[0]/b[0]))+0.5*abs(a[1]-b[1])
    keep.sort(key=lambda x:-spread(x));big=keep[:sel_big];rest=keep[sel_big:];rest=rng.sample(rest,sel_rand)
    return [(df.Comment[i],df.Comment[j],round(float(c),3),df.Movie_Name_CN[i],int(df.Star[i]),int(df.Star[j])) for c,i,j in big+rest]
d=pd.read_csv('../../data/datasets/douban_short/DMSC.csv',usecols=['Movie_Name_CN','Username','Comment','Star']).dropna();d['Comment']=d.Comment.astype(str).str.strip()
d1=d[d.Comment.str.len().between(15,80)].drop_duplicates('Comment').drop_duplicates(['Movie_Name_CN','Username'])
tops=d1.Movie_Name_CN.value_counts().index[:4]
s1=pd.concat([d1[d1.Movie_Name_CN==m].sample(700,random_state=7) for m in tops],ignore_index=True)
full=mine(s1,0.80,20,10)
rows=[]
for m in ['疯狂动物城','大圣归来']:
    sub=d[d.Movie_Name_CN==m].sample(frac=1,random_state=5).drop_duplicates('Username');seen=set();n=0
    for u,t,s in zip(sub.Username,sub.Comment,sub.Star):
        cl=[c.strip() for c in re.split(r'[，。！？；…,.!?;~～/\\、\s\-]+',t)];cl=[c for c in cl if 6<=len(c)<=25 and c not in seen]
        if not cl:continue
        c=rng.choice(cl);seen.add(c);rows.append((m,u,c,int(s)));n+=1
        if n==1250:break
s2=pd.DataFrame(rows,columns=['Movie_Name_CN','Username','Comment','Star'])
cl=mine(s2,0.82,8,7)
assert len(full)==30 and len(cl)==15,(len(full),len(cl))
for a,b,c,m,sa,sb in full: P.append(('zh_douban',f'full|{m}|{sa}/{sb}|cos{c}',a,b))
for a,b,c,m,sa,sb in cl: P.append(('zh_douban',f'clause|{m}|{sa}/{sb}|cos{c}',a,b))
# ---- (c) 45 en/ja
ms=collections.defaultdict(dict)
for r in rd(R+'extra/msynth.jsonl'): ms[(r['feature'],r['pair'])][r['side']]=r['text']
SUB=re.compile(r'\b(that|which|who|whom|whose|although|though|because|while|whereas|if|when|since|unless|until|after|before|as)\b',re.I)
plan_m={'Complex / Simple':6,'Active / Passive':3,'With nominalizations / Without nominalizations':3,'Fluent sentence / Disfluent sentence':3,'Cognitive process / Perceptual process':3,'Self-focused / Third-person singular':3,'Certain / Uncertain':2,'Polite / Impolite':2,'With sarcasm / Without sarcasm':2,'With Humor / Without Humor':2,'With metaphor / Without metaphor':2,'Formal / Informal':2,'Long average word length / Short average word length':2}
assert sum(plan_m.values())==35
for f,n in plan_m.items():
    c=[v for (ff,_),v in ms.items() if ff==f and len(v)==2]
    if f=='Complex / Simple':
        k=lambda v:len(SUB.findall(v['positive']))-len(SUB.findall(v['negative']))
        c=sorted([v for v in c if k(v)>=1],key=lambda v:-k(v));print('complex candidates (>=1 more subordinator)',len(c),'top diffs',[k(v) for v in c[:6]])
        for v in c[:6]: P.append(('en_ja',f'msynth:{f}',v['positive'],v['negative']))
        continue
    for v in rng.sample(c,n): P.append(('en_ja',f'msynth:{f}',v['positive'],v['negative']))
for lang,tab in(('en','dialogues_en'),('ja','dialogues_ja')):
    c=sqlite3.connect('../../data/genshin/genshin_v2.db')
    rows=c.execute(f"select origin_text,para_text,similarity from {tab} where para_text is not null and para_text!='' and length(origin_text) between 30 and 140 order by random() limit 400").fetchall()
    for o,p,s in rng.sample([r for r in rows if r[0]!=r[1]],5): P.append(('en_ja',f'genshin_{lang}',o,p))
assert len(P)==150,len(P)
out=[dict(pair=f'pair_{i:03d}',subset=s,tag=t,A=a,B=b) for i,(s,t,a,b) in enumerate(P)]
json.dump(out,open(R+'cal300/pairs.json','w'),ensure_ascii=False,indent=1)
print(collections.Counter(o['subset'] for o in out),len({o['A'] for o in out}|{o['B'] for o in out}))
