import json,sys
from PIL import Image
OUT='/Volumes/EMC/stylevec_data/audit/weak_ocr_review_20261011'
L=json.load(open(OUT+'/lines.json')); idx=[int(a) for a in sys.argv[1:]]
cs=[]
for i in idx:
    l=L[i]; im=Image.open(l['page_image']).convert('RGB'); x0,y0,x1,y1=l['box']
    c=im.crop((int(x0)-150,int(y0)-40,int(x1)+250,int(y1)+40)); s=min(6,1300/c.width); cs.append(c.resize((int(c.width*s),int(c.height*s)),Image.LANCZOS))
sh=Image.new('RGB',(1300,sum(c.height+10 for c in cs)),'white'); y=0
for c in cs: sh.paste(c,(0,y)); y+=c.height+10
sh.save(OUT+'/zoom.png')
