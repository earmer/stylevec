"""Local mining prototype: content-matched natural pairs for Weibo, CPED, Reddit. No API calls (embeddinggemma via local Ollama, LoRA r8 attention readout)."""
import sys,json,re,random,urllib.request,numpy as np,pandas as pd,collections
sys.path.insert(0,'expandtraining/2606/scripts');sys.path.insert(0,'expandtraining/2606/src')
from config import resolve_repo_path
from hongloumeng_trait_test import predict_pieces
from trait_predictor import OUT_DIR,load_run
rng=random.Random(8)
def embed(texts):
    out=[]
    for i in range(0,len(texts),64):
        req=urllib.request.Request('http://localhost:11434/api/embed',json.dumps({'model':'embeddinggemma:latest','input':texts[i:i+64]}).encode(),{'Content-Type':'application/json'})
        out+=json.load(urllib.request.urlopen(req,timeout=300))['embeddings']
    E=np.array(out,dtype=np.float32);return E/np.linalg.norm(E,axis=1,keepdims=True)
ZM=['不','没','别','但','却','虽','然而','可是','并非','才不'];EM=[' not',"n't",' no ',' never','but ','however','although']
zh_tok=lambda t:set(t);en_tok=lambda t:set(re.findall(r"[a-z']+",t.lower()))
def mine(df,key,textcol,tok,marks,thr=0.70,jmax=0.35,diff=('author',),maxn=6000):
    df=df.reset_index(drop=True);E=embed(list(df[textcol]));cand=[]
    groups=df.groupby(key).indices if key else {None:np.arange(len(df))}
    for g,ix in groups.items():
        if len(ix)<2:continue
        S=E[ix]@E[ix].T
        for a,b in zip(*np.where(np.triu(S,1)>=thr)):
            i,j=ix[a],ix[b];ta,tb=df[textcol][i],df[textcol][j]
            if ta==tb or any(df[c][i]==df[c][j] for c in diff):continue
            ja=len(tok(ta)&tok(tb))/max(1,len(tok(ta)|tok(tb)))
            if ja>=jmax:continue
            if any(m in ta.lower() for m in marks)!=any(m in tb.lower() for m in marks):continue
            cand.append((float(S[a,b]),ja,i,j))
    cand.sort(key=lambda x:-x[0]);used=set();keep=[]
    for c,ja,i,j in cand:
        if i in used or j in used:continue
        used|={i,j};keep.append((c,ja,i,j))
    return df,cand,keep
res={}
# ---- Weibo: original posts, no thread id -> mine globally across different users
w=pd.read_csv('data/datasets/weibo_users/0.数据/weibo.csv',encoding='utf-8-sig').rename(columns={'0':'text'}).dropna(subset=['text'])
w['text']=w.text.astype(str).str.strip();w=w[w['是否原创'].astype(str)=='True']
print('WEIBO rows',len(w),'users',w.uid.nunique(),'original')
w=w[w.text.str.len().between(15,80)&~w.text.str.contains('http|@|转发')].drop_duplicates('text')
w=w.sample(min(12000,len(w)),random_state=1)
res['weibo']=mine(w,None,'text',zh_tok,ZM,diff=('uid',));
# ---- CPED: group by (dialogue act, emotion); different dialogue + speaker
c=pd.read_csv('data/datasets/CPED/data/CPED/train_split.csv',encoding='utf-8-sig')
print('CPED rows',len(c),'dialogues',c.Dialogue_ID.nunique(),'speakers',c.Speaker.nunique(),'scenes',c.Scene.nunique(),'DA',c.DA.nunique(),'Emotion',c.Emotion.nunique())
c=c[c.Utterance.astype(str).str.len().between(8,40)].drop_duplicates('Utterance').rename(columns={'Utterance':'text'})
c['grp']=c.DA.astype(str)+'|'+c.Emotion.astype(str);c['dlg']=c.Dialogue_ID;c['author']=c.Speaker
c=c.sample(min(6000,len(c)),random_state=1)
res['cped']=mine(c,'grp','text',zh_tok,ZM,diff=('dlg','author'))
# ---- Reddit: group by subreddit; different author
r=pd.DataFrame([json.loads(l) for l in open('data/episodes/reddit_train.jsonl')])
print('REDDIT rows',len(r),'authors',r.author.nunique(),'subreddits',r.topic.nunique())
r=r[r.text.str.len().between(50,200)].drop_duplicates('text')
top=r.topic.value_counts().index[:12];r=pd.concat([g.sample(min(500,len(g)),random_state=1) for _,g in r[r.topic.isin(top)].groupby('topic')],ignore_index=True)
res['reddit']=mine(r,'topic','text',en_tok,EM,diff=('author',))
_,pred=load_run(resolve_repo_path(OUT_DIR)/'lora_r8_delta1.0_s42_attention')
for name,(df,cand,keep) in res.items():
    ts=sorted({df.text[i] for _,_,i,j in keep}|{df.text[j] for _,_,i,j in keep});P=dict(zip(ts,predict_pieces(pred,ts,16)))
    dy=lambda i,j:float(np.linalg.norm(P[df.text[i]]-P[df.text[j]]))
    n=np.array([dy(i,j) for _,_,i,j in keep])
    print(f'\n##### {name.upper()}: {len(df)} sampled lines | cap=1 pairs (cos>=.70, jac<.35, polarity ok, different author/dialogue): >=.70: {len(keep)}, >=.75: {sum(k[0]>=.75 for k in keep)}, >=.80: {sum(k[0]>=.80 for k in keep)}, >=.85: {sum(k[0]>=.85 for k in keep)} | ||dy|| median {np.median(n):.2f} p10 {np.quantile(n,.1):.2f} p90 {np.quantile(n,.9):.2f}')
    hi=[k for k in keep if k[0]>=.80];pool=hi if len(hi)>=8 else keep
    print('   (8 random pairs drawn from cos>=.80)' if len(hi)>=8 else f'   (only {len(hi)} pairs reach .80; 8 random pairs drawn from the cos>=.70 set, labelled by cos)')
    for k,(cs,ja,i,j) in enumerate(rng.sample(pool,min(8,len(pool))),1):
        extra=f" [{df.grp[i]} / {df.grp[j]}]" if name=='cped' else (f" [r/{df.topic[i]}]" if name=='reddit' else '')
        print(f"{k}. cos {cs:.3f} jac {ja:.2f} ||dy|| {dy(i,j):.2f}{extra}\n   A: {df.text[i]}\n   B: {df.text[j]}")
    json.dump([dict(a=df.text[i],b=df.text[j],cos=cs,jac=ja) for cs,ja,i,j in keep],open(f'/tmp/_unused','w')) if False else None
    json.dump([dict(a=df.text[i],b=df.text[j],cos=cs,jac=ja) for cs,ja,i,j in keep],open(f'expandtraining/2606/outputs/traits/mined_{name}_proto.json','w'),ensure_ascii=False)
