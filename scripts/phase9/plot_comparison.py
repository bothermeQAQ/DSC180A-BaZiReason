"""Paired percentage-point effects of the two controlled training changes and P vs G."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("results/phase9")
t = json.loads((out/"transfer_comparison.json").read_text())["endpoints"]
b = json.loads((out/"preservation.json").read_text())["bridge"]
fig, axes = plt.subplots(1,3, figsize=(10,3.0), sharey=True)
for ax,(title,e) in zip(axes,[("New supplied rules",t["novel"]),("Familiar rules, unseen pairs",b),("Unaided three-step",t["deep"])]):
    for y,(pair,color) in enumerate([("p-g","#654593"),("p-x","#13695c"),("x-s","#405f88")]):
        d=e["paired"][pair]
        low,high=d["chart_ci"]
        ax.plot([low,high],[y,y],lw=2,color=color)
        ax.scatter([d["pp"]],[y],s=34,color=color,zorder=3)
        ax.text(d["pp"],y+.17,f"{round(d['pp'],8):+.1f}",ha="center",color=color,fontsize=9)
    ax.axvline(0,c="#888888",lw=1,ls="--")
    ax.set(xlim=(-15,41), ylim=(-.45,2.55),title=title,xticks=[-10,0,10,20,30,40], yticks=[0,1,2])
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
axes[0].set_yticklabels(["Planned vs parent (P−G)","Adding the plan (P−X)","Expanding pairs (X−S)"])
fig.supxlabel("Accuracy change in percentage points · paired 95% chart interval",fontsize=10)
fig.tight_layout()
fig.savefig(out/"paired_effects.png",dpi=160,bbox_inches="tight")
