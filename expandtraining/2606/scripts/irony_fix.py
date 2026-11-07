import sys,os,json,time,urllib.request
sys.path.insert(0,'src');import traits_v21 as V,traits as T
for l in open('../../.env'):
    if l.startswith('TYPESAFE_KEY='): os.environ['TYPESAFE_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
    if l.startswith('OPENROUTER_KEY='): os.environ['OPENROUTER_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
O='outputs/traits/irony_fix/';C='outputs/traits/cal300/';Q=V.questions()
ids=['pair_124_A','pair_124_B','pair_133_A','pair_133_B','pair_112_A','pair_112_B']
old={}
for l in open(C+'raw.jsonl'): r=json.loads(l);old[(r['id'],r['judge'])]=r
for i in ids:
    txt=old[(i,'jev')]['text']
    for j in('jev','span02'):
        url,body,key=(T.JEV_URL,{"state":txt,"model":T.JEV_MODEL,"questions":Q},os.environ['TYPESAFE_API_KEY']) if j=='jev' else (T.OPENROUTER_URL,{"model":"respan/span-02","state":[txt],"questions":Q},os.environ['OPENROUTER_API_KEY'])
        h={"Content-Type":"application/json","Authorization":"Bearer "+key}
        for a in range(5):
            try: rep=json.load(urllib.request.urlopen(urllib.request.Request(url,json.dumps(body).encode(),h),timeout=90));break
            except Exception:
                if a==4: raise
                time.sleep(2**a)
        open(O+'raw.jsonl','a').write(json.dumps(dict(id=i,judge=j,text=txt,reply=rep),ensure_ascii=False)+'\n')
        o=old[(i,j)]['reply']['answers']['irony']['score'];n=rep['answers']['irony']['score']
        print(i,j,'irony old %.2f -> new %.2f'%(o,n),'cost',rep.get('usage',{}).get('cost'))
