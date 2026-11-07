import json,numpy as np
O='outputs/traits/cal300/';P=json.load(open(O+'pairs.json'))
R=[json.loads(l) for l in open(O+'raw.jsonl')];by={(r['id'],r['judge']):r for r in R}
sc=lambda r:{k:v['score'] for k,v in r['reply']['answers'].items()}
N=list(sc(R[0]));ids=[p['pair']+'_'+s for p in P for s in 'AB'];assert len(R)==600 and all((i,j) in by for i in ids for j in('jev','span02'))
A=np.array([[sc(by[i,'jev'])[n] for n in N] for i in ids]);B=np.array([[sc(by[i,'span02'])[n] for n in N] for i in ids])
sub=np.array([p['subset'] for p in P]);subl=np.repeat(sub,2)
c=[r['reply']['usage']['cost'] for r in R if r['judge']=='span02'];ji=[r['reply']['usage'] for r in R if r['judge']=='jev']
print('span-02 billed total $%.5f  per line $%.2e  per10k $%.3f | Jev tokens in %d out %d (mean %.0f/%.0f per line)'%(sum(c),np.mean(c),np.mean(c)*1e4,sum(u['input_tokens'] for u in ji),sum(u['output_tokens'] for u in ji),np.mean([u['input_tokens'] for u in ji]),np.mean([u['output_tokens'] for u in ji])))
print('latency jev %.2fs span %.2fs'%(np.mean([r['latency'] for r in R if r['judge']=='jev']),np.mean([r['latency'] for r in R if r['judge']=='span02'])))
f=lambda x,y:np.corrcoef(x,y)[0,1] if x.std()*y.std()>0 else float('nan')
dA=A[1::2]-A[0::2];dB=B[1::2]-B[0::2]
fold=np.arange(150)%5;np.random.RandomState(0).shuffle(fold);fl=np.repeat(fold,2)
print('\ntrait | Jev m/sd | span m/sd | SDdy J/S | line r | delta r | rawMAE | cvMAE | cv line r | a b')
rows={}
for j,n in enumerate(N):
    pred=np.zeros(300)
    for k in range(5):
        tr=fl!=k;a,b=np.polyfit(B[tr,j],A[tr,j],1);pred[~tr]=a*B[~tr,j]+b
    a,b=np.polyfit(B[:,j],A[:,j],1);rows[n]=(f(A[:,j],B[:,j]),f(dA[:,j],dB[:,j]))
    print(f"{n:21s}| {A[:,j].mean():.2f}/{A[:,j].std(ddof=1):.2f} | {B[:,j].mean():.2f}/{B[:,j].std(ddof=1):.2f} | {dA[:,j].std(ddof=1):.2f}/{dB[:,j].std(ddof=1):.2f} | {rows[n][0]:.2f} | {rows[n][1]:.2f} | {np.abs(A[:,j]-B[:,j]).mean():.2f} | {np.abs(A[:,j]-pred).mean():.2f} | {f(A[:,j],pred):.2f} | {a:.2f} {b:+.2f}")
print('MEAN line r %.2f delta r %.2f'%(np.mean([v[0] for v in rows.values()]),np.mean([v[1] for v in rows.values()])))
S=('zh_rewrite','zh_douban','en_ja');md={s:np.array([p['subset']==s for p in P]) for s in S};ml={s:subl==s for s in S}
print('\nsubset means (line r / delta r):')
for s in S: print(s,'%.2f / %.2f'%(np.nanmean([f(A[ml[s],j],B[ml[s],j]) for j in range(17)]),np.nanmean([f(dA[md[s],j],dB[md[s],j]) for j in range(17)])))
print('\ntrait | rewrite | douban | en_ja   (line r/delta r) ; SD(dy) Jev')
for j,n in enumerate(N): print(f"{n:21s}"+' | '.join(f"{f(A[ml[s],j],B[ml[s],j]):.2f}/{f(dA[md[s],j],dB[md[s],j]):.2f} sd{dA[md[s],j].std(ddof=1):.2f}" for s in S))
json.dump(dict(N=N,A=A.tolist(),B=B.tolist(),ids=ids),open(O+'scores.json','w'))
