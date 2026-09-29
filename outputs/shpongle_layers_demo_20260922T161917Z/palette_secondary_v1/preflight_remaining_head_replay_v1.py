"""Bounded remaining-cell HEAD-ONLY backend qualification, never palette science.
Checks exactly84 missing cells after preserved101 plus the separately proved cell101.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("first_replay_diag",HERE/"diagnose_first_donor_replay_v1.py")
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
m=d.m


def forward(model,device,x):
    with torch.no_grad(),torch.autocast(device_type=device,enabled=False):
        out=model(torch.from_numpy(x[None]).to(device))
        logits=out["chromatic_change_logit"][0].float().cpu().numpy()
        probability=torch.sigmoid(out["chromatic_change_logit"].float())[0].cpu().numpy()
    return logits,probability


def inferred_cues(probability):
    return np.unique(np.r_[0,m.tr._cue_indices(probability[None],.5)[0]]).astype(np.int64)


def compare_summary(actual,saved):
    result=d.compare(actual,saved)
    if result["bad_count"]==0:result.pop("bad_points")
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--registration",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args();start=time.monotonic()
    reg,eventreg,manifest,transforms,store,baselines,receipts,train,dev=m.load_authorized(args.registration,"predict")
    original=m.DERIVED_ROOT/"predictions_v1"
    cells=[]
    for receipt in sorted(receipts,key=lambda r:(m.ROLES.index(r["checkpoint_role"]),r["seed"])):
        for row in receipt["records"]:
            fname=f"{receipt['checkpoint_role']}_s{receipt['seed']}_{hashlib.sha256(row['recording_id'].encode()).hexdigest()[:20]}.npz"
            cells.append((receipt,row,fname,(original/fname).exists()))
    m.require(len(cells)==186 and sum(c[3] for c in cells)==101 and all(c[3] for c in cells[:101])
              and not any(c[3] for c in cells[101:]),"Partial scientific inventory changed")
    first_path=m.DERIVED_ROOT/"diagnosis_first_donor_replay_v1/receipt.json"
    first=json.loads(first_path.read_text());m.ev.checked_ref(first["arrays"])
    m.require(first["failing_cell"]["order_index"]==101 and first["driver"]==reg["driver"]
              and first["comparisons"]["donor_cpu_repeat"]["exact_equal"]
              and all(first["comparisons"][c]["cuda_vs_saved"]["exact_equal"] for c in ("model","donor"))
              and all(first["cues"][c][device+"_matches_saved"] for c in ("model","donor") for device in ("cpu","cuda")),
              "Existing first-cell proof is incomplete or mismatched")
    targets=cells[102:];m.require(len(targets)==84,"Expected exactly84 other missing cells")
    output=Path(args.output);m.require(not output.exists(),"New exclusive diagnostic output required")
    m.space(m.DERIVED_ROOT,4*2**20);output.mkdir(parents=True,exist_ok=False)
    before={str(p):m.tr.sha256(p) for p in sorted(original.glob("*.npz"))}
    target_inventory=[dict(index=i+102,role=r["checkpoint_role"],seed=r["seed"],recording_id=row["recording_id"],
        canonical_id=row["canonical_id"],missing_filename=fname,checkpoint=r["checkpoint"],original_prediction=row["artifact"])
        for i,(r,row,fname,_) in enumerate(targets)]
    base=dict(scope="Numerical execution qualification only; no palette rollout, metrics, contexts/audio/weight persistence",
        registration=m.ev.reference(args.registration),source=m.ev.reference(__file__),helper_source=m.ev.reference(HERE/"diagnose_first_donor_replay_v1.py"),
        reused_first_proof=m.ev.reference(first_path),targets=target_inventory,source_driver=reg["driver"],partial_inventory_sha256=m.tr.json_hash(before))
    m.write_json(output/"plan.json",{**base,"status":"head_only_started"},m.DERIVED_ROOT)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];proofs=[];active=None;model=None
    with np.load(m.ev.checked_ref(first["arrays"]),allow_pickle=False) as z:
        first_vectors={c:z[f"cpu__{c}__color_logit"].copy() for c in ("model","donor")}
    first_exception=dict(order_index=101,role="final",seed=0,recording_id="jkGH7JIHDSk",donor_id=first["failing_cell"]["donor_id"],
        checkpoint=first["checkpoint"],original_prediction=first["original_prediction"],input_hashes=first["input_hashes"],
        cpu_color_logit_hashes={c:d.array_hash(v) for c,v in first_vectors.items()},proof=m.ev.reference(first_path),
        cpu_vectors=first["arrays"],vector_keys={c:f"cpu__{c}__color_logit" for c in ("model","donor")})
    try:
        for entry,(receipt,row,fname,_) in zip(target_inventory,targets):
            m.deadline(start)
            checkpoint=receipt["checkpoint"];key=(receipt["checkpoint_role"],receipt["seed"])
            if key!=active:
                if model is not None:del model
                m.ev.validate_training_receipt(checkpoint,eventreg)
                model,_=m.ev.load_bound_model(checkpoint,transforms,transforms_sha256=reg["transforms"]["sha256"],
                    expected_bindings=checkpoint["bindings"],device="cpu");active=key
            rid=row["recording_id"];donor_id=eventreg["donor_map"][rid]
            record=store.visual(rid);donor_record=store.visual(donor_id)
            m.ev.verify_donor_tempos(store,[rid,donor_id],eventreg["donor_tempos_bpm"])
            x=m.ev.audio_window(store,record,transforms);donor_x=m.ev.audio_window(store,donor_record,transforms)
            dx=m.ev.nonrhythm_donor(x,donor_x)
            m.require(np.array_equal(dx[:,86:],x[:,86:]) and np.array_equal(dx[:,:86],donor_x[:,:86]),"Donor intervention differs")
            source_refs={}
            for source_id in (rid,donor_id):
                source_refs[source_id]={}
                for field in ("audio","rhythm","visual","visual_complete"):
                    if field in store.rows[source_id]:
                        ref=store.rows[source_id][field];m.ev.checked_ref(ref);source_refs[source_id][field]=ref
            saved={}
            with np.load(m.ev.checked_ref(row["artifact"]),allow_pickle=False) as z:
                for cond in ("model","donor"):
                    saved[cond]=(z[f"{cond}__color_logit"].copy(),m.cue_indices(z[f"{cond}__generated_color_cues"]))
                m.require(np.array_equal(z["model__requested_times10_s"],record.times)
                          and np.array_equal(z["model__source_times10_s"],record.source_times),"Clock mismatch")
            cpu={"model":forward(model,"cpu",x),"donor":forward(model,"cpu",dx)}
            result={**entry,"donor_id":donor_id,"sources":source_refs,"input_hashes":{
                "recipient_input":d.array_hash(x),"donor_original_input":d.array_hash(donor_x),"donor_intervened_input":d.array_hash(dx)},
                "cpu_vs_saved":{c:compare_summary(cpu[c][0],saved[c][0]) for c in ("model","donor")},
                "cpu_color_logit_hashes":{c:d.array_hash(cpu[c][0]) for c in ("model","donor")},
                "cue_inventory_hashes":{c:m.tr.json_hash(saved[c][1].tolist()) for c in ("model","donor")},
                "cpu_cues_exact":{c:bool(np.array_equal(inferred_cues(cpu[c][1]),saved[c][1])) for c in ("model","donor")}}
            m.require(all(result["cpu_cues_exact"].values()),"CPU cue inventory changed")
            failed=[c for c in ("model","donor") if not result["cpu_vs_saved"][c]["allclose_1e5"]]
            result["original_guard_failed_conditions"]=failed
            if failed:
                repeat={"model":forward(model,"cpu",x),"donor":forward(model,"cpu",dx)}
                stable={c:bool(np.array_equal(cpu[c][0],repeat[c][0]) and np.array_equal(cpu[c][1],repeat[c][1])) for c in ("model","donor")}
                m.require(all(stable.values()),"CPU failed cell not bitwise stable")
                m.require(torch.cuda.is_available(),"One-GPU proof replay unavailable")
                model=model.to("cuda").eval();torch.cuda.synchronize();t=time.monotonic()
                gpu={"model":forward(model,"cuda",x),"donor":forward(model,"cuda",dx)}
                torch.cuda.synchronize();gpu_seconds=time.monotonic()-t
                result.update(cpu_repeat_exact=stable,cuda_vs_saved={c:compare_summary(gpu[c][0],saved[c][0]) for c in ("model","donor")},
                    cuda_cues_exact={c:bool(np.array_equal(inferred_cues(gpu[c][1]),saved[c][1])) for c in ("model","donor")},gpu_replay_seconds=gpu_seconds)
                m.require(all(result["cuda_vs_saved"][c]["exact_equal"] for c in ("model","donor")),"GPU replay not bitwise saved")
                m.require(all(result["cuda_cues_exact"].values()),"GPU cue inventory changed")
                model=model.to("cpu").eval()
                # Persist only compact head vectors, never inputs or contexts.
                vector_path=output/f"proof_vectors_{entry['index']:03d}.npz"
                m.write_npz(vector_path,{c:cpu[c][0] for c in ("model","donor")},m.DERIVED_ROOT)
                result["cpu_vectors"]=m.ev.reference(vector_path)
                proof_path=output/f"proof_{entry['index']:03d}.json"
                m.write_json(proof_path,{**result,"status":"strict_roundoff_proved"},m.DERIVED_ROOT)
                proofs.append(dict(order_index=entry["index"],role=entry["role"],seed=entry["seed"],recording_id=rid,donor_id=donor_id,
                    checkpoint=checkpoint,original_prediction=row["artifact"],input_hashes=result["input_hashes"],
                    cpu_color_logit_hashes=result["cpu_color_logit_hashes"],proof=m.ev.reference(proof_path),
                    cpu_vectors=result["cpu_vectors"],vector_keys={c:c for c in ("model","donor")}))
            result["status"]="strict_roundoff_proved" if failed else "original_guard_pass"
            rows.append(result)
            print(json.dumps(dict(completed=len(rows),total=84,fixture=rid,guard_failed=failed,new_proof_count=len(proofs))),flush=True)
        m.deadline(start)
    except Exception as error:
        m.write_json(output/"blocked.json",{**base,"status":"blocked","reason":str(error),"completed":len(rows),
            "rows":rows,"new_proofs":proofs,"elapsed_s":time.monotonic()-start},m.DERIVED_ROOT)
        raise
    after={str(p):m.tr.sha256(p) for p in sorted(original.glob("*.npz"))}
    m.require(before==after,"Partial science artifacts changed")
    exceptions=[first_exception,*proofs]
    m.write_json(output/"exception_map_proposal.json",dict(status="PROPOSAL_ONLY_NOT_AUTHORIZED",scope="Exact proved cells only;othercells retainoriginal tolerance;CPU science unchanged",
        original_driver=reg["driver"],registration=m.ev.reference(args.registration),map_count=len(exceptions),exceptions=exceptions,
        preflight_plan=m.ev.reference(output/"plan.json"),no_resumed_science=True),m.DERIVED_ROOT)
    m.write_json(output/"complete.json",{**base,"status":"all84_numeric_qualification_complete","completed":84,
        "original_guard_pass_count":84-len(proofs),"new_strict_proof_count":len(proofs),"total_exact_cell_exceptions_including_first":len(exceptions),
        "rows":rows,"exception_map":m.ev.reference(output/"exception_map_proposal.json"),"preserved_partial_count":101,
        "partial_outputs_unchanged":True,"elapsed_s":time.monotonic()-start},m.DERIVED_ROOT)
    print(json.dumps(dict(status="complete",checked=84,new_strict_proofs=len(proofs),total_exception_map=len(exceptions),output=str(output))),flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
