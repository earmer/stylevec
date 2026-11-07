import sys,os,json,time,threading,urllib.request,concurrent.futures as cf
sys.path.insert(0,'src');import traits_v21 as V,traits as T
for l in open('../../.env'):
    if l.startswith('TYPESAFE_KEY='): os.environ['TYPESAFE_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
    if l.startswith('OPENROUTER_KEY='): os.environ['OPENROUTER_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
O='outputs/traits/cal300/';P=json.load(open(O+'pairs.json'));Q=V.questions();lock=threading.Lock()
done=set()
if os.path.exists(O+'raw.jsonl'):
    for l in open(O+'raw.jsonl'): r=json.loads(l);done.add((r['id'],r['judge']))
jobs=[(p['pair']+'_'+s,t,j) for p in P for s,t in(('A',p['A']),('B',p['B'])) for j in('jev','span02') if (p['pair']+'_'+s,j) not in done]
print('todo',len(jobs),flush=True)
def go(job):
    i,txt,j=job
    url,body,key=(T.JEV_URL,{"state":txt,"model":T.JEV_MODEL,"questions":Q},os.environ['TYPESAFE_API_KEY']) if j=='jev' else (T.OPENROUTER_URL,{"model":"respan/span-02","state":[txt],"questions":Q},os.environ['OPENROUTER_API_KEY'])
    h={"Content-Type":"application/json","Authorization":"Bearer "+key};t0=time.time()
    for a in range(6):
        try: rep=json.load(urllib.request.urlopen(urllib.request.Request(url,json.dumps(body).encode(),h),timeout=90));break
        except Exception:
            if a==5: return
            time.sleep(2**a)
    with lock: open(O+'raw.jsonl','a').write(json.dumps(dict(id=i,judge=j,text=txt,latency=time.time()-t0,reply=rep),ensure_ascii=False)+'\n')
with cf.ThreadPoolExecutor(4) as ex: list(ex.map(go,jobs))
print('done',flush=True)
