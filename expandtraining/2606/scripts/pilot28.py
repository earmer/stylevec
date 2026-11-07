import sys,os,json,time,urllib.request,random,collections
sys.path.insert(0,'src'); import traits_v21 as V; import traits as T
for l in open('../../.env'):
    if l.startswith('TYPESAFE_KEY='): os.environ['TYPESAFE_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
    if l.startswith('OPENROUTER_KEY='): os.environ['OPENROUTER_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
O='outputs/traits/pilot28/'; SC='/private/tmp/claude-501/-Users-earmercarey-stylevec/85c4f0fb-18b3-45be-93f5-002c1738e9b6/scratchpad/'
P=json.load(open('outputs/traits/pilot20/pairs.json'))
s2=json.load(open(SC+'step2.json')); fin={json.loads(l)['id']:json.loads(l) for l in open(SC+'step3_raw/final.jsonl')}
pick=lambda st,pre:[r for r in s2 if r['style']==st and r['source'].startswith(pre)][0]
for pre in ['呵呵，我看我还是','欸，不容易想象吗']: r=pick('arrogant',pre); P.append(['arrogant',r['source'],r['text']])
r=pick('humble','好的，谢谢您'); P.append(['humble',r['source'],r['text']])
g=fin['humble-g1']; P.append(['humble',g['source'],g['polished']])
assert len(P)==14
json.dump(P,open(O+'pairs.json','w'),ensure_ascii=False,indent=1)
Q=V.questions(); assert len(Q)==17
def post(url,body,key):
    h={"Content-Type":"application/json","Authorization":"Bearer "+key};t=time.time()
    for a in range(5):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(url,json.dumps(body).encode(),h),timeout=90)),time.time()-t
        except Exception:
            if a==4: raise
            time.sleep(2**a)
for pi,(src,a,b) in enumerate(P):
    for side,txt in(('a',a),('b',b)):
        for jn,url,body,key in[('jev',T.JEV_URL,{"state":txt,"model":T.JEV_MODEL,"questions":Q},os.environ['TYPESAFE_API_KEY']),
                               ('span02',T.OPENROUTER_URL,{"model":"respan/span-02","state":[txt],"questions":Q},os.environ['OPENROUTER_API_KEY'])]:
            rep,dt=post(url,body,key)
            open(O+'raw.jsonl','a').write(json.dumps(dict(pair=pi,src=src,side=side,text=txt,judge=jn,latency=dt,reply=rep),ensure_ascii=False)+'\n')
print('done')
