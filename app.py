"""AI Chemistry Laboratory — educational Streamlit cheminformatics + ML app.
Run: streamlit run app.py
"""
import io, json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, QED, rdMolDescriptors, Draw
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import py3Dmol
    HAS_3D = True
except Exception:
    HAS_3D = False

EXAMPLES = {
 "Caffeine":"Cn1c(=O)c2c(ncn2C)n(C)c1=O",
 "Aspirin":"CC(=O)OC1=CC=CC=C1C(=O)O",
 "Paracetamol":"CC(=O)NC1=CC=C(O)C=C1",
 "Ibuprofen":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
 "Nicotine":"CN1CCC[C@H]1c1cccnc1",
 "Benzene":"c1ccccc1", "Ethanol":"CCO", "Glucose":"C(C1C(C(C(C(O1)O)O)O)O)O",
 "Acetic acid":"CC(=O)O", "Acetone":"CC(=O)C", "Urea":"NC(=O)N",
}

st.set_page_config(page_title="AI Chemistry Laboratory", page_icon="🧬", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Manrope,sans-serif}
.stApp{background:linear-gradient(130deg,#f4f7ff,#f8f3ff,#effcff)}
.block-container{max-width:1300px;padding-top:1.2rem}
.hero{padding:26px;border-radius:22px;color:white;background:linear-gradient(110deg,#111827,#4338ca,#0891b2);margin-bottom:20px}
.hero h1{font-size:clamp(2rem,5vw,3rem);font-weight:800;margin:0}
.hero p{margin:6px 0 0;font-size:1rem}
div[data-testid="stMetric"]{background:white;border:1px solid #e5e7eb;border-radius:15px;padding:12px}
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:700;min-height:2.7rem}
</style>
<div class="hero"><h1>🧬 AI Chemistry Laboratory</h1>
<p>Machine learning • Molecular fingerprints • Similarity search • Chemical data science</p></div>
""", unsafe_allow_html=True)

def mol(smiles):
    return Chem.MolFromSmiles(str(smiles).strip()) if smiles and str(smiles).strip() else None

def props(m):
    return {
      "Formula":rdMolDescriptors.CalcMolFormula(m),
      "Molecular weight":round(Descriptors.MolWt(m),3),
      "Exact mass":round(Descriptors.ExactMolWt(m),4),
      "LogP":round(Crippen.MolLogP(m),3),
      "TPSA":round(rdMolDescriptors.CalcTPSA(m),3),
      "H-bond donors":Lipinski.NumHDonors(m),
      "H-bond acceptors":Lipinski.NumHAcceptors(m),
      "Rotatable bonds":Lipinski.NumRotatableBonds(m),
      "Heavy atoms":m.GetNumHeavyAtoms(),
      "Aromatic rings":Lipinski.NumAromaticRings(m),
      "Fraction Csp3":round(rdMolDescriptors.CalcFractionCSP3(m),3),
      "QED":round(QED.qed(m),3),
    }

def fp(m, n=1024):
    bits=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=n)
    arr=np.zeros((n,),dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bits,arr)
    return arr

def png(m):
    mm=Chem.Mol(m); AllChem.Compute2DCoords(mm)
    return Draw.MolToImage(mm,size=(700,470))

def pdf_report(name, smiles, m):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    b=io.BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4)
    styles=getSampleStyleSheet(); story=[
      Paragraph("AI Chemistry Laboratory — Molecular Report",styles["Title"]),
      Paragraph("Molecule: "+name,styles["Normal"]),
      Paragraph("Generated: "+datetime.now().strftime("%d %b %Y %H:%M"),styles["Normal"]),
      Spacer(1,8),Paragraph("Canonical SMILES: "+Chem.MolToSmiles(m),styles["Code"]),Spacer(1,8)]
    im=io.BytesIO(); png(m).save(im,format="PNG"); im.seek(0)
    story += [Image(im,width=125*mm,height=84*mm),Spacer(1,8)]
    rows=[["Descriptor","Value"]]+[[str(k),str(v)] for k,v in props(m).items()]
    t=Table(rows,colWidths=[75*mm,85*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#4338ca")),
      ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),
      ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef2ff")]),("PADDING",(0,0),(-1,-1),6)]))
    story += [t,Spacer(1,10),Paragraph("Educational output only; not validated medical or safety advice.",styles["Italic"])]
    doc.build(story); return b.getvalue()

def demo_data():
    rows=[]
    extra=["CC","CCC","CCCC","CCCCC","CCCCCC","CO","CCO","CCCO","CC(=O)O",
      "C1CCCCC1","c1ccccc1","CCN","CCCN","CC(=O)N","O=C=O","CCCl","CCBr",
      "CC(=O)OC","CC(C)O","O","N","C","CC(=O)OCC","c1ccncc1","c1ccoc1",
      "c1ccsc1","CC(C)C","CC(C)(C)O","O=C(O)C(O)C"]
    for n,s in list(EXAMPLES.items())+[(f"Demo {i+1}",s) for i,s in enumerate(extra)]:
        m=mol(s)
        if m: rows.append({"name":n,"smiles":s,"target":Crippen.MolLogP(m)})
    return pd.DataFrame(rows).drop_duplicates("smiles")

def validate_csv(df):
    df.columns=[str(c).lower() for c in df.columns]
    if not {"smiles","target"}.issubset(df.columns):
        raise ValueError("CSV must contain columns named smiles and target.")
    df=df[["smiles","target"]].copy()
    df["target"]=pd.to_numeric(df["target"],errors="coerce")
    df=df.dropna()
    df["mol"]=df.smiles.astype(str).apply(mol)
    df=df.dropna(subset=["mol"])
    if len(df)<10: raise ValueError("Please provide at least 10 valid molecule rows.")
    return df.reset_index(drop=True)

page=st.sidebar.radio("🧭 Laboratory modules",[
 "Molecule Workbench","AI Property Predictor","Similarity Search",
 "Molecular Clustering","Chemistry Assistant","About & Data"])
st.sidebar.caption("AI Chemistry Laboratory • AIML project")

if page=="Molecule Workbench":
    st.header("🔬 Molecular Workbench")
    a,b=st.columns([1,1])
    with a:
        choice=st.selectbox("Choose example",["Custom SMILES"]+list(EXAMPLES))
        smi=st.text_area("SMILES notation",value="" if choice=="Custom SMILES" else EXAMPLES[choice],height=90)
        name=st.text_input("Molecule name",value=choice if choice!="Custom SMILES" else "My molecule")
        go=st.button("Analyze molecule",type="primary",use_container_width=True)
    with b: st.info("Paste a SMILES string or choose an example, then analyze it.")
    if go:
        m=mol(smi)
        if not m: st.error("Invalid SMILES. Please check the notation.")
        else: st.session_state["current"]={"mol":m,"smiles":smi,"name":name or "Molecule"}
    if "current" in st.session_state:
        d=st.session_state.current; m=d["mol"]
        t1,t2,t3,t4,t5=st.tabs(["2D Structure","3D Model","Descriptors","Drug-likeness","Export"])
        with t1:
            im=png(m); st.image(im,use_container_width=True)
            out=io.BytesIO(); im.save(out,format="PNG")
            st.download_button("Download 2D PNG",out.getvalue(),"molecule.png","image/png")
        with t2:
            if HAS_3D:
                m3=Chem.AddHs(Chem.Mol(m))
                if AllChem.EmbedMolecule(m3,randomSeed=7)==0:
                    AllChem.MMFFOptimizeMolecule(m3)
                    v=py3Dmol.view(width=800,height=480); v.addModel(Chem.MolToMolBlock(m3),"mol")
                    v.setStyle({"stick":{"radius":.16}}); v.setBackgroundColor("#fbfdff"); v.zoomTo()
                    import streamlit.components.v1 as components
                    components.html(v._make_html(),height=500)
                else: st.warning("Could not generate a 3D conformer for this molecule.")
            else: st.info("Install py3Dmol to enable the interactive 3D viewer.")
        with t3:
            p=props(m); x,y,z,w=st.columns(4)
            x.metric("Molecular weight",p["Molecular weight"]); y.metric("LogP",p["LogP"])
            z.metric("TPSA",p["TPSA"]); w.metric("QED",p["QED"])
            st.dataframe(pd.DataFrame(list(p.items()),columns=["Descriptor","Value"]),use_container_width=True,hide_index=True)
        with t4:
            checks=[("Molecular weight ≤ 500",props(m)["Molecular weight"]<=500),
             ("LogP ≤ 5",props(m)["LogP"]<=5),("H-bond donors ≤ 5",props(m)["H-bond donors"]<=5),
             ("H-bond acceptors ≤ 10",props(m)["H-bond acceptors"]<=10)]
            st.metric("Lipinski criteria met",f"{sum(v for _,v in checks)}/4")
            for label,ok in checks: st.write(("✅ " if ok else "⚠️ ")+label)
            st.caption("Lipinski's Rule of Five is a screening heuristic, not proof of drug efficacy or safety.")
        with t5:
            st.download_button("Download PDF molecular report",pdf_report(d["name"],d["smiles"],m),
                               "molecule_report.pdf","application/pdf",type="primary")
            st.download_button("Download descriptors JSON",json.dumps(props(m),indent=2),
                               "descriptors.json","application/json")

elif page=="AI Property Predictor":
    st.header("🤖 Machine-Learning Property Predictor")
    st.warning("Built-in example target is RDKit-calculated LogP, not experimental data. Upload your own curated experimental dataset for a meaningful QSAR workflow.")
    uploaded=st.file_uploader("Optional CSV: columns smiles,target",type=["csv"])
    try:
        if uploaded: data=validate_csv(pd.read_csv(uploaded)); label="uploaded target"
        else:
            raw=demo_data(); raw["mol"]=raw.smiles.apply(mol); data=raw.rename(columns={"target":"target"})
            st.caption(f"Using {len(data)} built-in demo molecules; target = RDKit-calculated LogP.")
        with st.expander("Preview data"): st.dataframe(data[["smiles","target"]].head(30),hide_index=True)
        if st.button("Train Random Forest model",type="primary"):
            X=np.vstack([fp(m) for m in data.mol]); y=data.target.to_numpy(float)
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
            model=RandomForestRegressor(n_estimators=250,random_state=42,n_jobs=-1)
            model.fit(Xtr,ytr); pred=model.predict(Xte)
            st.session_state["ml_model"]=model
            st.session_state["ml_metrics"]={"MAE":mean_absolute_error(yte,pred),"R2":r2_score(yte,pred),"n":len(yte)}
            st.session_state["ml_label"]=label
            st.success("Training complete.")
    except Exception as e: st.error(str(e))
    if "ml_model" in st.session_state:
        q=st.session_state.ml_metrics; x,y,z=st.columns(3)
        x.metric("Test MAE",f"{q['MAE']:.3f}"); y.metric("Test R²",f"{q['R2']:.3f}"); z.metric("Test rows",q["n"])
        st.caption("Random holdout metrics can be unstable for small datasets; use external validation for research.")
        sm=st.text_input("SMILES to predict",value=EXAMPLES["Aspirin"])
        if st.button("Predict property"):
            m=mol(sm)
            if not m: st.error("Invalid SMILES.")
            else:
                val=st.session_state.ml_model.predict(fp(m).reshape(1,-1))[0]
                st.success(f"Estimated {st.session_state.ml_label}: {val:.4f}")
                st.caption("Prediction is an estimate from the training data, not a clinical or safety result.")

elif page=="Similarity Search":
    st.header("🧲 Molecular Similarity Search")
    q=st.text_input("Query molecule SMILES",value=EXAMPLES["Caffeine"])
    file=st.file_uploader("Optional library CSV with name,smiles columns",type=["csv"])
    if file:
        try:
            lib=pd.read_csv(file); lib.columns=[str(c).lower() for c in lib.columns]
            if not {"name","smiles"}.issubset(lib.columns): raise ValueError("CSV needs name and smiles columns.")
        except Exception as e: st.error(str(e)); lib=pd.DataFrame(columns=["name","smiles"])
    else: lib=pd.DataFrame([{"name":k,"smiles":v} for k,v in EXAMPLES.items()])
    threshold=st.slider("Minimum Tanimoto similarity",0.,1.,.15,.05)
    if st.button("Find similar molecules",type="primary"):
        qm=mol(q)
        if not qm: st.error("Invalid query SMILES.")
        else:
            qfp=AllChem.GetMorganFingerprintAsBitVect(qm,2,nBits=2048); rows=[]
            for _,r in lib.iterrows():
                m=mol(r.smiles)
                if m:
                    f=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048)
                    score=DataStructs.TanimotoSimilarity(qfp,f)
                    if score>=threshold: rows.append({"Name":r["name"],"SMILES":r["smiles"],"Tanimoto":round(score,4)})
            result=pd.DataFrame(rows)
            if result.empty: st.info("No molecules reached this threshold.")
            else:
                result=result.sort_values("Tanimoto",ascending=False)
                st.dataframe(result,use_container_width=True,hide_index=True)
                st.download_button("Download similarity results",result.to_csv(index=False),"similarity.csv","text/csv")

elif page=="Molecular Clustering":
    st.header("🧩 Molecular Clustering")
    file=st.file_uploader("Optional CSV with name,smiles columns",type=["csv"])
    if file:
        try:
            df=pd.read_csv(file); df.columns=[str(c).lower() for c in df.columns]
            if "smiles" not in df: raise ValueError("CSV needs smiles column.")
            if "name" not in df: df["name"]=[f"Molecule {i+1}" for i in range(len(df))]
        except Exception as e: st.error(str(e)); df=pd.DataFrame(columns=["name","smiles"])
    else: df=pd.DataFrame([{"name":k,"smiles":v} for k,v in EXAMPLES.items()])
    k=st.slider("Number of clusters",2,6,3)
    if st.button("Cluster molecules",type="primary"):
        valid=[]
        for _,r in df.iterrows():
            m=mol(r.smiles)
            if m: valid.append((r["name"],r.smiles,m))
        if len(valid)<k: st.error(f"Need at least {k} valid molecules.")
        else:
            X=np.vstack([fp(x[2]) for x in valid])
            labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)
            xy=PCA(n_components=2,random_state=42).fit_transform(X.astype(float))
            st.session_state["clusters"]=pd.DataFrame({"Name":[x[0] for x in valid],"SMILES":[x[1] for x in valid],
              "Cluster":labels.astype(str),"PCA 1":xy[:,0],"PCA 2":xy[:,1]})
    if "clusters" in st.session_state:
        result=st.session_state.clusters
        st.dataframe(result,use_container_width=True,hide_index=True)
        st.scatter_chart(result,x="PCA 1",y="PCA 2",color="Cluster")
        st.download_button("Download cluster assignments",result.to_csv(index=False),"clusters.csv","text/csv")

elif page=="Chemistry Assistant":
    st.header("💬 Chemistry Assistant")
    st.caption("Offline rule-based explainer; it is not a generative AI chatbot.")
    question=st.text_area("Ask about chemistry or cheminformatics")
    if st.button("Explain",type="primary"):
        q=question.lower()
        topics=[
         (["smiles"],"SMILES is a text notation that represents atoms, bonds, rings, branches and stereochemistry in a molecule."),
         (["logp","lipophil"],"LogP estimates partitioning between octanol and water for a neutral compound. It is a lipophilicity descriptor, not a direct measure of absorption or safety."),
         (["tpsa","polar surface"],"TPSA is topological polar surface area, a descriptor derived from polar atoms and their bonding environment."),
         (["fingerprint","tanimoto","similarity"],"Molecular fingerprints encode structural features as bit vectors. Tanimoto similarity compares shared on-bits with the union of on-bits."),
         (["lipinski","rule of five"],"Lipinski's Rule of Five is a heuristic: MW ≤500, LogP ≤5, H-bond donors ≤5 and acceptors ≤10. It is not a pass/fail drug test."),
         (["qed","drug likeness","drug-likeness"],"QED is a quantitative estimate of drug-likeness from molecular properties; it does not establish efficacy or safety."),
         (["qsar","machine learning","prediction"],"QSAR models learn relationships between molecular representations and measured endpoints. Good models require curated experimental data, proper validation and applicability-domain checks."),
         (["cluster"],"Clustering groups molecules by similarity of their representations. Here K-Means uses Morgan fingerprint vectors; PCA projects them to two dimensions for display.")
        ]
        answer=next((a for keys,a in topics if any(k in q for k in keys)),
          "Try asking about SMILES, LogP, TPSA, molecular fingerprints, Tanimoto similarity, Lipinski, QED, QSAR or clustering.")
        st.info(answer)

else:
    st.header("📚 About & Dataset Format")
    st.markdown("""
**AI Chemistry Laboratory — Version 2** combines RDKit cheminformatics with scikit-learn machine learning.

- Molecular workbench: SMILES validation, 2D/3D structures, descriptors, Lipinski screening and PDF export.
- ML predictor: Random Forest regression from Morgan fingerprints.
- Similarity search: Tanimoto similarity.
- Clustering: K-Means and PCA visualization.
- Chemistry assistant: offline rule-based explanations.

**For your own predictor dataset**, upload a CSV with `smiles` and `target` columns. Use numeric experimental endpoint values and keep an independent test set for proper validation.

**Scientific limitations:** the built-in training labels are RDKit-computed LogP, not measured values. This application does not provide validated toxicity, efficacy, reaction or clinical predictions. It is an educational project.

**Developer:** K. Balashanmuga • AIML Student • SRM Institute of Science and Technology, Tiruchirappalli
""")
    sample=pd.DataFrame({"smiles":["CCO","CC(=O)O","c1ccccc1","CCN"],"target":[-0.001,0.09,2.13,-0.03]})
    st.dataframe(sample,hide_index=True)
    st.download_button("Download sample CSV",sample.to_csv(index=False),"sample_qsar.csv","text/csv")

st.divider()
st.caption("AI Chemistry Laboratory • Educational AIML project • 2026")
