"""One-cell head-only numerical replay. No palette rollout, target scoring or source edit."""
from pathlib import Path
from dataclasses import asdict
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("palette_driver_diag",HERE/"run_palette_history_secondary_v1.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()

def compare(actual, reference):
    actual=np.asarray(actual,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
    d=np.abs(actual-reference);tol=1e-5+1e-5*np.abs(reference);ratio=d/tol
    bad=np.flatnonzero(d>tol);imax=int(np.argmax(d));iratio=int(np.argmax(ratio))
    def point(i):
        return dict(index=int(i),relative_time_s=float(i/10),actual=float(actual[i]),reference=float(reference[i]),
                    absolute_error=float(d[i]),tolerance=float(tol[i]),tolerance_ratio=float(ratio[i]))
    refnorm=float(np.linalg.norm(reference));diffnorm=float(np.linalg.norm(actual-reference))
    return dict(exact_equal=bool(np.array_equal(actual,reference)),all_finite=bool(np.isfinite(actual).all() and np.isfinite(reference).all()),
        allclose_1e5=bool(np.allclose(actual,reference,rtol=1e-5,atol=1e-5)),max_absolute_error=float(d.max()),
        mean_absolute_error=float(d.mean()),rms_error=float(np.sqrt(np.mean(d*d))),difference_l2=diffnorm,
        reference_l2=refnorm,relative_l2=diffnorm/refnorm if refnorm else None,
        normwise_inf_ratio=float(d.max()/np.max(np.abs(reference))) if np.max(np.abs(reference)) else None,
        bad_count=len(bad),bad_points=[point(i) for i in bad],max_abs_point=point(imax),max_componentwise_point=point(iratio))

def main():
    a=argparse.ArgumentParser();a.add_argument("--registration",required=True);a.add_argument("--output",required=True)
    args=a.parse_args();started=time.monotonic()
    reg,eventreg,manifest,transforms,store,baseline,receipts,train,dev=m.load_authorized(args.registration,"predict")
    order=sorted(receipts,key=lambda r:(m.ROLES.index(r["checkpoint_role"]),r["seed"]))
    existing=m.DERIVED_ROOT/"predictions_v1";cells=[]
    for r in order:
        for row in r["records"]:
            filename=f"{r['checkpoint_role']}_s{r['seed']}_{hashlib.sha256(row['recording_id'].encode()).hexdigest()[:20]}.npz"
            cells.append((r,row,filename,(existing/filename).exists()))
    failed=next(i for i,c in enumerate(cells) if not c[3])
    m.require(failed==101 and sum(c[3] for c in cells)==101 and all(c[3] for c in cells[:failed]),"Unexpected partial output inventory")
    receipt,row,filename,_=cells[failed]
    m.require((receipt["checkpoint_role"],receipt["seed"],row["recording_id"])==("final",0,"jkGH7JIHDSk"),"Unexpected failing fixture")
    checkpoint=receipt["checkpoint"];m.ev.validate_training_receipt(checkpoint,eventreg)
    output=Path(args.output);m.require(not output.exists(),"Diagnosis output exists")
    m.space(m.DERIVED_ROOT,5*2**20)
    before={str(p):m.tr.sha256(p) for p in sorted(existing.glob("*.npz"))}
    donor_id=eventreg["donor_map"][row["recording_id"]]
    record=store.visual(row["recording_id"]);donor_record=store.visual(donor_id)
    m.ev.verify_donor_tempos(store,[row["recording_id"],donor_id],eventreg["donor_tempos_bpm"])
    x=m.ev.audio_window(store,record,transforms)
    donor_x=m.ev.audio_window(store,donor_record,transforms)
    dx=m.ev.nonrhythm_donor(x,donor_x)
    preserved=np.array_equal(dx[:,86:],x[:,86:]);replaced=np.array_equal(dx[:,:86],donor_x[:,:86])
    m.require(preserved and replaced,"Donor intervention mismatch")
    sources={}
    for rid in (row["recording_id"],donor_id):
        sources[rid]={}
        for key in ("audio","rhythm","visual","visual_complete"):
            ref=store.rows[rid].get(key)
            if ref is not None:
                m.ev.checked_ref(ref);sources[rid][key]=ref
    arrays={"recipient_input":x,"donor_original_input":donor_x,"donor_intervened_input":dx}
    with np.load(m.ev.checked_ref(row["artifact"]),allow_pickle=False) as z:
        for cond in ("model","donor"):
            for key in ("color_logit","color","accent_logit","generated_color_cues"):
                arrays[f"saved__{cond}__{key}"]=z[f"{cond}__{key}"].copy()
        for key in ("requested_times10_s","source_times10_s"):
            arrays[key]=z["model__"+key].copy()
    m.require(np.array_equal(record.times,arrays["requested_times10_s"]),"Requested clock mismatch")
    m.require(np.array_equal(record.source_times,arrays["source_times10_s"]),"Source clock mismatch")
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,metadata=m.ev.load_bound_model(checkpoint,transforms,transforms_sha256=reg["transforms"]["sha256"],
                                         expected_bindings=checkpoint["bindings"],device="cpu")
    def forward(device,cond,inp):
        with torch.no_grad(),torch.autocast(device_type=device,enabled=False):
            o=model(torch.from_numpy(inp[None]).to(device))
            arrays[f"{device}__{cond}__color_logit"]=o["chromatic_change_logit"][0].float().cpu().numpy()
            arrays[f"{device}__{cond}__color"]=torch.sigmoid(o["chromatic_change_logit"].float())[0].cpu().numpy()
            arrays[f"{device}__{cond}__accent_logit"]=o["accent_logit"][0].float().cpu().numpy()
    forward("cpu","model",x);forward("cpu","donor",dx);forward("cpu","donor_repeat",dx)
    m.require(torch.cuda.is_available(),"Authorized one-GPU numerical replay unavailable")
    model=model.to("cuda").eval()
    torch.cuda.synchronize();gpu_start=time.monotonic()
    forward("cuda","model",x);forward("cuda","donor",dx)
    torch.cuda.synchronize();gpu_seconds=time.monotonic()-gpu_start
    comparisons={};cues={}
    for cond in ("model","donor"):
        comparisons[cond]={}
        for left,right in (("cpu","saved"),("cuda","saved"),("cpu","cuda")):
            comparisons[cond][f"{left}_vs_{right}"]=compare(arrays[f"{left}__{cond}__color_logit"],arrays[f"{right}__{cond}__color_logit"])
        original=m.cue_indices(arrays[f"saved__{cond}__generated_color_cues"])
        cues[cond]={"saved_indices":original.tolist()}
        for device in ("cpu","cuda"):
            got=np.unique(np.r_[0,m.tr._cue_indices(arrays[f"{device}__{cond}__color"][None],.5)[0]]).astype(int)
            cues[cond][device+"_indices"]=got.tolist();cues[cond][device+"_matches_saved"]=bool(np.array_equal(got,original))
    comparisons["donor_cpu_repeat"]=compare(arrays["cpu__donor_repeat__color_logit"],arrays["cpu__donor__color_logit"])
    after={str(p):m.tr.sha256(p) for p in sorted(existing.glob("*.npz"))}
    m.require(before==after,"Partial scientific artifacts changed during diagnosis")
    result=dict(status="diagnostic_complete",scope="ONE failed fixture head-only numerical replay;no palette rollouts or scientific scores",
        failing_cell=dict(order_index=failed,role=receipt["checkpoint_role"],seed=receipt["seed"],recording_id=row["recording_id"],canonical_id=row["canonical_id"],donor_id=donor_id,missing_filename=filename),
        checkpoint=checkpoint,original_prediction=row["artifact"],registration=m.ev.reference(args.registration),
        source_bindings=sources,input_hashes={k:array_hash(v) for k,v in arrays.items() if k.endswith("input")},
        intervention=dict(replaced0to85_exact=replaced,preserved86to93_exact=preserved),
        comparisons=comparisons,cues=cues,preserved_partial_outputs=dict(count=101,unchanged=True,inventory_sha256=m.tr.json_hash(before)),
        environment=dict(torch=torch.__version__,numpy=np.__version__,torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            device=torch.cuda.get_device_name(),cpu_threads=torch.get_num_threads(),float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,mkldnn_enabled=torch.backends.mkldnn.enabled),
        gpu_replay_seconds=gpu_seconds,elapsed_s=time.monotonic()-started,driver=m.ev.reference(HERE/"run_palette_history_secondary_v1.py"),diagnostic_source=m.ev.reference(__file__))
    output.mkdir(parents=True,exist_ok=False)
    m.write_npz(output/"arrays.npz",arrays,m.DERIVED_ROOT)
    result["arrays"]=m.ev.reference(output/"arrays.npz")
    m.write_json(output/"receipt.json",result,m.DERIVED_ROOT)
    print(json.dumps({"fixture":result["failing_cell"],"comparisons":comparisons,"cues_equal":{c:{k:v for k,v in r.items() if k.endswith("matches_saved")} for c,r in cues.items()},"receipt":str(output/"receipt.json")},indent=2))

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
