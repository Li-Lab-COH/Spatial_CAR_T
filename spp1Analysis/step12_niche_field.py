"""
THE DECISIVE CONTROL FOR 08 SECTION C.

Section C measures cross-coupling: macrophage SPP1-TAM level vs the pathway score of the
NON-MYELOID cells in a distance band around it. It found the coupling reaches further after RT.

But that statistic has two inputs, and section C only ever interrogated the macrophage side.
If the NICHE FIELD ITSELF (the non-myeloid pathway score, as a spatial field) is smoother /
longer-ranged after radiation -- bigger necrotic and hypoxic patches, coarser architecture --
then the value at 150um is more similar to the value at 25um, and the cross-coupling at 150um
rises AUTOMATICALLY, with nothing whatsoever changing about the macrophages.

Test: the autocorrelation range of the niche field alone, computed only among non-myeloid cells,
with no macrophage involved. Background-matched, per the project's design rule.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, spatialdata as spd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from pathlib import Path
pd.set_option("display.width", 220)

ZARR = "/Users/janzules/Roselab/Spatial/CAR_T/data/zarrFiles/1_C2l_annotations_400"
GMT  = Path("/Users/janzules/Roselab/Spatial/CAR_T/data/references/pathways")
MPP, BANDS = 0.34, [(0,25),(25,50),(50,100),(100,150)]
MAC=["M1_like_Mac","M2_like_Mac","Intermediate_Mac"]
MYE=MAC+["Classical_Mono","Nonclassical_Mono","Monocyte","Macrophage","N1_like_Neu","N2_like_Neu","Neutrophil"]
EXCL=set(MYE)|{"Unknown","Erythrocyte"}
PANEL={"Hypoxia":"HALLMARK_HYPOXIA","ECM remodeling":"REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION",
       "Cholesterol handling":"HALLMARK_CHOLESTEROL_HOMEOSTASIS",
       "Oxidative phosphorylation":"HALLMARK_OXIDATIVE_PHOSPHORYLATION",
       "IFN-γ response":"HALLMARK_INTERFERON_GAMMA_RESPONSE"}

def load_gmt(d):
    o={}
    for f in sorted(Path(d).glob("*.gmt")):
        for l in open(f):
            p=l.rstrip("\n").split("\t")
            if len(p)>2 and p[0] not in o: o[p[0]]=[g for g in p[2:] if g]
    return o
gmt=load_gmt(GMT)

print("loading spatial object ...")
sd=spd.read_zarr(ZARR); ad_=sd.tables["segmentation_counts"]
sc.pp.normalize_total(ad_, target_sum=1e4); sc.pp.log1p(ad_)
for lbl,full in PANEL.items():
    g=[x for x in gmt[full] if x in ad_.var_names]
    sc.tl.score_genes(ad_, g, score_name=lbl, ctrl_size=50, random_state=0)
    print(f"  scored {lbl}: {len(g)} genes")

obs=ad_.obs
rows=[]
tissues=sorted(obs["tissue"].unique())
for ti,t in enumerate(tissues,1):
    m = (obs["tissue"]==t).values
    sub = obs[m]; keep = ~sub["c2l_consolidated"].astype(str).isin(EXCL)
    if keep.sum() < 300: 
        print(f"  skip {t}: only {keep.sum()} niche cells"); continue
    xy = ad_.obsm["spatial"][m][keep.values]*MPP
    if len(xy) > 4000:
        idx = np.random.default_rng(0).choice(len(xy), 4000, replace=False)
    else:
        idx = np.arange(len(xy))
    xy_s = xy[idx]
    nn = NearestNeighbors(radius=150).fit(xy)
    dist, ind = nn.radius_neighbors(xy_s, radius=150)
    for lbl in PANEL:
        v = sub.loc[keep.values, lbl].values.astype(float)
        vs = v[idx]
        z = (v - v.mean())/ (v.std()+1e-12); zs = z[idx]
        for lo,hi in BANDS:
            lag=np.full(len(idx), np.nan)
            for i,(dd,ii) in enumerate(zip(dist,ind)):
                sel=(dd>=lo)&(dd<hi)
                if sel.sum()>=3: lag[i]=z[ii[sel]].mean()
            ok=np.isfinite(lag)
            if ok.sum()<50: continue
            r=np.corrcoef(zs[ok], lag[ok])[0,1]
            rows.append(dict(tissue=t, pathway=lbl, band=f"{lo}-{hi}", r_field=r, n=int(ok.sum())))
    print(f"  [{ti}/{len(tissues)}] {t}: {keep.sum():,} niche cells")

F=pd.DataFrame(rows)
meta=obs.groupby("tissue", observed=True)[["condition","tumor_loc"]].first()
F=F.join(meta, on="tissue")
F["cond"]=F["condition"].astype(str)
F["RT"]=F["cond"].str.startswith("RT")
F.to_csv("/Users/janzules/Roselab/Spatial/CAR_T/results/spp1_analysis/rt_niche_field_autocorr.csv", index=False)

print("\n"+"="*100)
print("NICHE-FIELD AUTOCORRELATION (non-myeloid cells only — no macrophage in this statistic)")
print("="*100)
piv=F.pivot_table(index=["pathway","band"], columns="RT", values="r_field")
piv.columns=["no_RT","RT"]; piv["delta"]=piv.RT-piv.no_RT
print(piv.to_string(float_format=lambda x:f"{x:+.4f}"))

print("\n--- far band (100-150um) only, background-matched as the design requires ---")
far=F[(F.band=="100-150") & (F.cond!="NoTx")]
rows=[]
for cart, rt_arm, ct_arm in [("PSCA","RTCyPSCA","CyPSCA"),("nonspecific","RTCyT72","CyT72")]:
    for p in PANEL:
        a=far[(far.cond==rt_arm)&(far.pathway==p)].r_field
        b=far[(far.cond==ct_arm)&(far.pathway==p)].r_field
        rows.append(dict(cart=cart,pathway=p,rt=a.mean(),no_rt=b.mean(),delta=a.mean()-b.mean(),
                         n_rt=len(a),n_ctl=len(b)))
D=pd.DataFrame(rows).pivot_table(index="pathway",columns="cart",values="delta")
D["mean"]=D.mean(axis=1); D["sign_agree"]=np.sign(D.PSCA)==np.sign(D.nonspecific)
print(D.to_string(float_format=lambda x:f"{x:+.4f}"))
print(f"\n  mean far-band niche-field autocorrelation:  no_RT {far[~far.RT].r_field.mean():+.4f}"
      f"   RT {far[far.RT].r_field.mean():+.4f}"
      f"   delta {far[far.RT].r_field.mean()-far[~far.RT].r_field.mean():+.4f}")
print("""
  READ: if the niche field's OWN 100-150um autocorrelation is substantially higher after RT,
  then section C's 'the SPP1-TAM coupling reaches further' is a restatement of 'the irradiated
  tissue's pathway fields are spatially coarser', and it is not a macrophage finding.""")
