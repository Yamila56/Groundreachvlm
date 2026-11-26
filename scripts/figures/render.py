"""Render the README figures from archived results. Run from any directory."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/figures"
OUT.mkdir(parents=True, exist_ok=True)
teal, blue, gray, gold, purple, red = "#168A83","#3974B8","#64748B","#D8A04B","#8877AE","#C45C58"
ink = "#182230"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"text.color":ink,
    "axes.labelcolor":ink,"xtick.color":gray,"ytick.color":gray,"axes.edgecolor":"#CBD5E1",
    "axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":9,
    "axes.titleweight":"bold","svg.fonttype":"path","svg.hashsalt":"research"})
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def save(fig,name):
    for ext in ("png","svg"):
        fig.savefig(OUT/f"{name}.{ext}",dpi=300,facecolor="white",metadata={"Date":None} if ext=="svg" else None)
    plt.close(fig)
def diagram(height):
    fig=plt.figure(figsize=(8,height))
    ax=fig.add_axes([0,0,1,1],xlim=(0,100),ylim=(0,100))
    ax.axis("off")
    return fig,ax
def label(ax,x,y,s,size=8,color=ink,bold=False,**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight="bold" if bold else "normal",
                   va="center",**kw)
def arrow(ax,x,y,u,v,color=gray,dashed=False):
    ax.annotate("",(u,v),(x,y),arrowprops=dict(arrowstyle="->",color=color,lw=1,
                linestyle="--" if dashed else "-",mutation_scale=9))
def panel(ax,title):
    ax.set_title(title,loc="left",pad=10)

import hashlib
pilot=ROOT/"results/grounding-study/point-token-pilot"
rows=read(pilot/"point-token-confirm/manifest.json")["rows"][:4]
assert [r["truth"] for r in rows]==["N","Y","Y","N"]
scene=ROOT/"assets/figure-inputs"
def load_image(row,field,digest):
    path=scene/Path(row[field]).name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row[digest]
    return Image.open(path).convert("RGB")
def crop(ax,row,field,digest,limits):
    im=load_image(row,field,digest)
    ax.imshow(im.crop(limits),extent=[limits[0],limits[2],limits[3],limits[1]],interpolation="nearest")
    x,y=row["point"]
    ax.plot(x,y,"o",ms=10,mec="white",mfc="none",mew=1.4)
    ax.plot(x,y,"+",ms=6,color="black",mew=.8)
    ax.axis("off")
fig=plt.figure(figsize=(8,3.1),layout="constrained")
gs=fig.add_gridspec(2,3,width_ratios=[1.65,.8,1.15])
a=fig.add_subplot(gs[:,0]); panel(a,"(a)  Resolve the language query")
crop(a,rows[0],"marked_image","marked_sha256",(210,190,442,316))
x,y,u,v=rows[0]["box"];a.add_patch(Rectangle((x,y),u-x,v-y,fill=False,ec="white",lw=1,ls="--"))
a.text(.01,-.09,"Nearest purple box left of the red cylinder",transform=a.transAxes,fontsize=7)
for i,r in enumerate(rows[:2]):
    a=fig.add_subplot(gs[i,1]);crop(a,r,"sp_map","map_sha256",(218,234,268,284))
    a.set_title("(b)  Switch maps" if i==0 else "",loc="left",fontsize=9)
    a.text(.02,.87,"Workspace "+r["side"],transform=a.transAxes,fontsize=7,color="white",bbox=dict(facecolor=ink,pad=1,edgecolor="none"))
    a.text(.02,.08,"Reachable" if r["truth"]=="Y" else "Unreachable",transform=a.transAxes,fontsize=7,weight="bold",bbox=dict(facecolor="white",pad=1,edgecolor="none"))
a=fig.add_subplot(gs[:,2]);panel(a,"(c)  Select a map token")
a.set(xlim=(0,1),ylim=(0,1),aspect="equal");a.axis("off")
for i in range(6):
    for j in range(5):
        a.add_patch(Rectangle((.1+i*.12,.34+j*.10),.112,.091,color=teal if (i,j)==(2,2) else "#DFE5EC"))
a.text(.1,.23,"Point-indexed grid",fontsize=8,weight="bold")
a.text(.1,.09,"Residual adapter → extra slot",fontsize=7,color=blue)
save(fig,"overview")

fig,ax=diagram(2.15)
label(ax,3,90,"From a spatial query to a language-model input",11,bold=True)
label(ax,3,65,"Aligned RGB\nand S-P map",9,bold=True)
arrow(ax,16,64,22,64)
label(ax,23,64,"Frozen visual\nencoder",9,gray)
for i in range(5):
    for j in range(4):
        ax.add_patch(Rectangle((37+i*1.7,52+j*5.2),1.5,4.7,color=teal if (i,j)==(2,1) else "#DFE5EC"))
label(ax,44,51,"Merged grid",7,gray)
arrow(ax,45.5,64,79,64)
label(ax,52,73,"Original visual tokens",8,gray)
label(ax,80,65,"Frozen Qwen3-VL\nlanguage model",9,gray,True)
arrow(ax,89,53,89,36)
label(ax,83,28,"Yes / No logits",8,bold=True)
ax.plot([41,41,23,23],[56,47,47,32],color=teal,lw=.9)
label(ax,3,30,"Selected point token",8,teal,True)
arrow(ax,22,30,29,30,teal)
label(ax,30,30,"Linear → GELU → Linear",8,blue,True)
label(ax,30,16,"32-D bottleneck · 133,152 parameters",7,blue)
ax.plot([28,28,57,57],[30,40,40,32],color=blue,lw=.8)
arrow(ax,52,30,55,30,blue);label(ax,56,30,"⊕",11,blue)
arrow(ax,59,30,63,30,teal)
label(ax,64,30,"Extra slot",8,teal)
arrow(ax,73,32,80,56,teal)
save(fig,"adaptation")

fig,axes=plt.subplots(2,2,figsize=(7,2.85),layout="constrained")
for a,r in zip(axes.flat,[rows[0],rows[1],rows[2],rows[3]]):
    im=np.asarray(load_image(r,"sp_map","map_sha256").crop((216,225,436,290)),dtype=int)
    mask=np.ptp(im,axis=2)>=83
    a.imshow(np.where(mask[:,:,None],[133,193,184],[229,233,237]).astype("uint8"),extent=[216,436,290,225],interpolation="nearest")
    a.plot(*r["point"],"o",ms=9,mfc="white",mec=ink);a.plot(*r["point"],"+",color=ink,ms=5);a.axis("off")
    a.text(.98,.08,"Reachable" if r["truth"]=="Y" else "Unreachable",ha="right",transform=a.transAxes,fontsize=8,weight="bold",bbox=dict(facecolor="white",edgecolor="none",pad=2))
    if r["id"].endswith("0"):a.set_title("Workspace "+r["side"],fontsize=10)
    if r["side"]=="A":a.text(0,-.15,r["target"].capitalize(),fontsize=8,transform=a.transAxes)
save(fig,"counterfactual")
sets=["automatic","cross-check-29","cross-check-17","automatic-permuted"]
metrics=[read(pilot/f"point-token-{name}/metrics.json") for name in sets]
fig,axes=plt.subplots(2,2,figsize=(8,4.6),layout="constrained")
for a,title,key,den in zip(axes.flat,["(a)  Certified questions","(b)  Target switches","(c)  Map switches"],[None,"target_flip","workspace_flip"],[128,16,32]):
    panel(a,title)
    for idx,(method,color,mark) in enumerate([("pixel_readout",gray,"x"),("adapted_mean",purple,"s"),("adapted_point",teal,"o")]):
        vals=[(m[method][key] if key else m[method])["correct"] for m in metrics]
        a.scatter(np.arange(4)+(idx-1)*.14,vals,color=color,marker=mark,s=26,label=["Pixel rule","Mean","Point"][idx])
    a.set(xticks=range(4),xticklabels=["A · 17","A · 29","B · 17","B · 29"],ylim=(-den*.08,den*1.1),ylabel=f"Correct / {den}")
    a.grid(axis="y",color="#E2E8F0",lw=.5);a.set_axisbelow(True)
axes[0,0].legend(fontsize=6,frameon=False,ncol=3,loc="lower center")
a=axes[1,1];panel(a,"(d)  Point shift · set B, seed 29")
jitter=read(pilot/"point-token-jitter/metrics.json")
for i,m in enumerate([metrics[-1],jitter]):
    for off,key,col,mark in [(-.06,"pixel_readout",gray,"x"),(.06,"adapted_point",teal,"o")]:
        val=m[key]["correct"];a.scatter(i+off,val,color=col,marker=mark);a.text(i+off,val+(1 if off<0 else -2),str(val),ha="center",fontsize=7,color=col)
a.set(xticks=[0,1],xticklabels=["Original","8 px shift"],ylabel="Correct / 128",ylim=(111,130),xlim=(-.35,1.35))
save(fig,"main_results")
