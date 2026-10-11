import json
from PIL import Image, ImageDraw, ImageFont
OUT='/Volumes/EMC/stylevec_data/audit/weak_ocr_review_20261011'
L=json.load(open(OUT+'/lines.json'))
font=ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',26)
PER=10; W=1400
import os; os.makedirs(OUT+'/sheets',exist_ok=True)
cache={}
def cell(i,l):
    im=cache.get(l['page_image']) or cache.setdefault(l['page_image'],Image.open(l['page_image']).convert('RGB'))
    x0,y0,x1,y1=l['box']; px,py=70,28
    c=im.crop((max(0,int(x0)-px),max(0,int(y0)-py),min(im.width,int(x1)+px),min(im.height,int(y1)+py)))
    d=ImageDraw.Draw(c); d.rectangle((px-2 if x0>=px else x0-2,py-2 if y0>=py else y0-2,(x1-x0)+px+2 if x0>=px else x1+2,(y1-y0)+py+2 if y0>=py else y1+2),outline=(255,0,0))
    s=min(4,(W-20)/c.width); c=c.resize((int(c.width*s),int(c.height*s)),Image.LANCZOS)
    cv=Image.new('RGB',(W,c.height+34),'white'); cv.paste(c,(10,32))
    ImageDraw.Draw(cv).text((10,2),f"#{i}",fill=(0,0,200),font=font)
    ImageDraw.Draw(cv).line((0,cv.height-1,W,cv.height-1),fill=(160,160,160))
    return cv
for s in range(0,len(L),PER):
    cs=[cell(i,l) for i,l in enumerate(L[s:s+PER],s)]
    sh=Image.new('RGB',(W,sum(c.height for c in cs)),'white'); y=0
    for c in cs: sh.paste(c,(0,y)); y+=c.height
    sh.save(f"{OUT}/sheets/sheet_{s//PER:03d}.png")
print(len(L)//PER+1)
