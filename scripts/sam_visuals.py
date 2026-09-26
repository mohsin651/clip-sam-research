"""Small image-only panel helpers; no dataset or annotation access."""
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import matplotlib

def heat(raw):
    span=raw.max()-raw.min();norm=(raw-raw.min())/span if span>0 else np.zeros_like(raw)
    return (matplotlib.colormaps['inferno'](norm)[...,:3]*255).astype(np.uint8)

def overlay(image,mask,color=(0,220,120)):
    arr=np.asarray(image).copy();arr[mask]=(.55*arr[mask]+.45*np.array(color)).astype(np.uint8)
    return arr

def draw_prompt(image,prompt):
    pic=image.copy();d=ImageDraw.Draw(pic);radius=max(3,min(image.size)//70)
    if prompt['box'] is not None:d.rectangle(prompt['box'],outline='#00ffff',width=3)
    if prompt['points'] is not None:
        for x,y in prompt['points']:d.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#ff3030',outline='white',width=2)
    return pic

def panel(cells,path,title,columns=4):
    cw,ch=300,340;top=90;rows=(len(cells)+columns-1)//columns
    image=Image.new('RGB',(cw*columns,top+ch*rows),'#f7f9fc');d=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for i,line in enumerate(textwrap.wrap(title,width=125)):d.text((10,8+22*i),line,fill='#16233c',font=font)
    for i,(label,pixels) in enumerate(cells):
        x=(i%columns)*cw;y=top+(i//columns)*ch
        for j,line in enumerate(textwrap.wrap(label,width=33)):d.text((x+8,y+19*j),line,fill='#16233c',font=font)
        pic=Image.fromarray(pixels) if isinstance(pixels,np.ndarray) else pixels.copy()
        pic=ImageOps.contain(pic.convert('RGB'),(284,284))
        image.paste(pic,(x+8+(284-pic.width)//2,y+45+(284-pic.height)//2))
    path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
