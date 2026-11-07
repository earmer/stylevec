"""20-line pilot: unified 20-trait rubric (level 2 = everyday baseline), Jev vs respan/span-02. Raw replies saved at once."""
import sys,os,json,random,time,urllib.request,collections
sys.path.insert(0,'src'); import traits as T
for l in open('../../.env'):
    if l.startswith('TYPESAFE_KEY='): os.environ['TYPESAFE_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
    if l.startswith('OPENROUTER_KEY='): os.environ['OPENROUTER_API_KEY']=l.split('=',1)[1].strip().strip('"\'')
O='outputs/traits/pilot20/'; rd=lambda p:[json.loads(l) for l in open(p)]
NEWC={
'archaic':["Heavy internet, trendy or slangy language","Casual modern language","Standard modern prose, including common idioms and set phrases","Classical literary flavour: classical words, 成语 beyond everyday use","Largely Classical Chinese / archaic grammar and vocabulary (之乎者也, thee/thou)"],
'nonstandard':["Stiff, textbook or officially standardised language","Careful standard language","Ordinary everyday standard language, with common colloquialisms","Some slang, dialect words or personal verbal tics","Strongly nonstandard: heavy dialect, slang, catchphrases or a distinctive verbal tic"],
'hedging':["Absolute, categorical assertions or commands","Emphatic, confident","Plain, direct statement as people ordinarily speak","Softened: 吧, 呢, 可能, maybe, I guess","Evasive or very tentative: avoids committing (或许…吧, 我也说不准)"],
'irony':["Gushing, openly heartfelt","Earnest, sincere","Plain and literal, no irony signal","Dry or wry","Openly sarcastic or mocking"],
'ornament':["Bare, telegraphic, minimal words","Very plain, functional","Ordinary everyday wording, perhaps one common idiom","Clear imagery, figurative language or set phrases","Heavily ornamented: parallelism, poetic imagery, metaphor, allusion"],
'intensity':["Flat, detached, emotionless","Calm and composed","Ordinary everyday feeling, plainly expressed","Clearly emotional: exclamations, strong words","Outburst: very intense, broken or repeated exclamations"],
'verbosity':["Terse to the point of curtness","Very brief","Ordinary length for what is said","Elaborate, adds detail and qualifiers","Long-winded, rambling or digressive"],
'theatricality':["Deadpan, flat","Understated","Ordinary natural expressiveness","Dramatic, expressive flourish","Theatrical or grandiloquent: declamation, stage-like flourish"],
'spokenness':["Formal written prose: polished, edited sentences","Composed, written-like","Ordinary plain language that reads equally well spoken or written","Speech-like: loose syntax, interjections","Spontaneous speech: fillers (嗯, 呃, 啊), self-corrections, trailing off, fragments"]}
NEWT={'syntactic_complexity':("Syntactic complexity of the sentence structure.",["Minimal fragments, telegraphic","Simple short clauses","Ordinary everyday sentence structure","Multi-clause with subordinate connectors and modifiers","Deeply nested syntax, long pre-nominal modifiers"]),
'abstraction':("Concrete versus abstract content of the wording.",["Purely concrete sensory or physical details and actions","Mostly concrete events and objects","Everyday balance of concrete and general","Analytical or conceptual focus","Dense abstract theory, nominalizations"]),
'ego_focus':("Perspective: impersonal versus centred on the speaker's own self.",["Purely impersonal, third-person detached narration","Mostly external, objective focus","Everyday balanced framing","Noticeably personal first-person perspective","Strongly self-centred: the speaker's own feelings and stance dominate"])}
V2={'formality':("Register of the wording, from heavy slang and dialect to ceremonial or academic language (not the topic).",["Heavy slang, internet speak or dialect","Casual, relaxed chat","Standard everyday language","Careful formal written prose","Ceremonial, official or academic register"]),
'warmth':("Rough or cold versus warm, gentle and caring toward the listener.",["Rough, coarse, cold or hostile","Reserved, firm or distant","Everyday neutral, even-tempered","Kind, gentle, considerate","Very warm, tender, affectionate"]),
'intensity':("Emotional intensity of the delivery, from flat to dramatic declamation, whatever the emotion.",["Flat, deadpan, emotionless","Calm, restrained","Ordinary everyday emotional level","Clearly emotional, expressive","Intense outburst or dramatic declamation"]),
'deference':("How the speaker ranks themselves against the listener in address and self-reference.",["Condescending or commanding: talks down, orders","Blunt, unceremonious","Peer-to-peer, ordinary address","Polite: please, softened requests, respectful address","Very humble, self-lowering, deferential"]),
'swagger':("How the speaker presents their own standing: self-effacing versus boastful.",["Self-deprecating, insecure","Modest","Ordinary, unremarkable self-presentation","Self-assured, confident","Boastful, arrogant, self-aggrandizing"]),
'valence':("Emotional tone of the delivery: negative versus positive feeling.",["Hostile, bitter or despairing","Mildly negative: annoyed, worried, sad","Neutral, even","Mildly positive: friendly, pleased","Strongly positive: delighted, approving, praising"]),
'playfulness':("Serious versus playful, teasing or joking delivery.",["Grave, solemn","Serious, matter-of-fact","Ordinary, neither serious nor playful","Light and friendly, a little humour","Clearly teasing, joking or mischievous"]),
'age_voice':("How old the voice sounds from its wording alone.",["Elderly voice: old-fashioned wording","Mature adult","Ordinary adult","Youthful, lively","Childlike or cutesy: baby talk, reduplication, cute nicknames"]),
'spokenness':("Formal written prose versus spontaneous speech.",["Formal written prose: polished, edited sentences","Composed, written-like","Ordinary plain language that reads equally well spoken or written","Speech-like: loose syntax, interjections","Spontaneous speech: fillers (嗯, 呃, 啊), self-corrections, trailing off, fragments"])}
T.TRAITS={**{k:V2[k] if k in V2 else (v[0],NEWC[k] if k in NEWC else v[1]) for k,v in T.TRAITS.items() if k not in('softness','nonstandard','theatricality')},**NEWT}
T.CLARIFICATIONS['ornament']="A short, plain line with no imagery or rhetoric is ordinary plain language (level 2 or below), not an absence of signal."
if __name__!='__main__': sys.exit
def pairs():
    random.seed(20); P=[]
    P+=[('douban',a,b) for a,b in [("良心大片，中国动漫的希望！！！","国漫良心 希望你我的梦想都能实现。"),("非常不错啦，故事略欠逻辑，但是人设画面都超赞的，良心之作！！","虽然故事情节有点俗套，都在意料之中，但结构完整，起承转合恰到好处，但实属良心之作"),("国产动画也算是尽力之作了。画面算做到了能做到的最好。但是差距依然明显。剧情也是短板。","有人说跟外国动画电影比还是辣鸡，但是至少能看到是在用心做动画了，算是国产动画电影的鸡头了。剧情是硬伤，特别是结尾有点早泄")]]
    ms=collections.defaultdict(dict)
    for r in rd(O+'../extra/msynth.jsonl'): ms[(r['feature'],r['pair'])][r['side']]=r['text']
    for f in ['Self-focused / Third-person singular','Cognitive process / Perceptual process','Complex / Simple']:
        k=random.choice([k for k in ms if k[0]==f and len(ms[k])==2]); P.append(('msynth:'+f,ms[k]['positive'],ms[k]['negative']))
    v=rd('outputs/bookstyle/verify.jsonl')
    for s in ['casual_chat','ornate']:
        r=random.choice([x for x in v if x['style']==s and x['lang']=='zh' and len(x['source'])<160]); P.append(('book:'+s,r['source'],r['text']))
    g=rd('outputs/traits/trait_pairs_train.jsonl'); 
    for r in random.sample([x for x in g if len(x['a'])<70],2): P.append(('genshin',r['a'],r['b']))
    return P
def post(url,body,key,model_cost=False):
    h={"Content-Type":"application/json","Authorization":"Bearer "+key}; t=time.time()
    for a in range(5):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(url,json.dumps(body).encode(),h),timeout=90)),time.time()-t
        except Exception as e:
            if a==4: raise
            time.sleep(2**a)
if __name__=='__main__':
    P=pairs(); json.dump(P,open(O+'pairs.json','w'),ensure_ascii=False,indent=1)
    Qj=T.questions(); Qs=T.questions(clarified=True)
    assert len(Qj)==17 and all(len(v['criteria'])==5 for v in Qj.values())
    for src,a,b in P:
        for side,txt in (('a',a),('b',b)):
            for name,url,body,key in [('jev',T.JEV_URL,{"state":txt,"model":T.JEV_MODEL,"questions":Qj},os.environ['TYPESAFE_API_KEY']),
                                      ('span02',T.OPENROUTER_URL,{"model":"respan/span-02","state":[txt],"questions":Qs},os.environ['OPENROUTER_API_KEY'])]:
                rep,dt=post(url,body,key)
                with open(O+'raw.jsonl','a') as f: f.write(json.dumps(dict(src=src,side=side,text=txt,judge=name,latency=dt,reply=rep),ensure_ascii=False)+'\n')
    print('done')
