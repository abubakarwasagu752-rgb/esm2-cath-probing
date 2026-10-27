import subprocess, tempfile, os, pandas as pd
MM = os.path.expanduser("~/mmseqs/bin/mmseqs")
pairs = [("data_cap5","1nh2C00","1x4rA01"),("data_cap10","1vl1A00","1vr6A01"),("data_cap10","5w7xB00","2w3oA00"),
         ("data_cap10","2ksoB00","1v38A00"),("data_cap15","1tf3A02","2rpcA03"),("data_cap15","1v7vA02","4ktpA02")]
for d, q, t in pairs:
    dom = pd.read_csv(f"{d}/domains.tsv", sep="\t", dtype=str).set_index("domain_id")
    with tempfile.TemporaryDirectory() as tmp:
        open(f"{tmp}/q.fa","w").write(f">{q}\n{dom.loc[q,'seq']}\n")
        subprocess.run([MM,"easy-search",f"{tmp}/q.fa",f"{d}/domains.fasta",f"{tmp}/o.tsv",f"{tmp}/w","-e","10","-s","7.5",
                        "--max-seqs","1000","--format-output","query,target,evalue,bits"],check=True,capture_output=True)
        h = pd.read_csv(f"{tmp}/o.tsv", sep="\t", header=None, names=["q","t","e","b"])
        r = h[h.t == t]
        print(d, q, t, "full-DB e-value:", (f"{r.e.iloc[0]:.2e}" if len(r) else "not found"))
