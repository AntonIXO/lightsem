"""Read-only numerical-proof metadata normalization; no model/inference/scoring."""
from pathlib import Path
import importlib.util, json, sys
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("replay_diag_normalize",HERE/"diagnose_first_donor_replay_v1.py")
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
m=d.m
ROOT=m.DERIVED_ROOT/"diagnosis_remaining_head_preflight_v1"
source=ROOT/"exception_map_proposal.json"
original=json.loads(source.read_text());rows=[]
for cell in original["exceptions"]:
    proof=json.loads(m.ev.checked_ref(cell["proof"]).read_text())
    isfirst=cell["order_index"]==101
    failed=([head for head in ("model","donor") if not proof["comparisons"][head]["cpu_vs_saved"]["allclose_1e5"]]
            if isfirst else proof["original_guard_failed_conditions"])
    if isfirst:
        m.require(all(proof["comparisons"][h]["cuda_vs_saved"]["exact_equal"] for h in ("model","donor")),"First proof GPU mismatch")
        m.require(proof["comparisons"]["donor_cpu_repeat"]["exact_equal"],"First proof CPU mismatch")
        m.require(all(proof["cues"][h][dev+"_matches_saved"] for h in ("model","donor") for dev in ("cpu","cuda")),"First proof cues mismatch")
    else:
        m.require(proof["status"]=="strict_roundoff_proved" and all(proof["cpu_repeat_exact"].values())
                  and all(proof["cuda_vs_saved"][h]["exact_equal"] for h in ("model","donor"))
                  and all(proof["cpu_cues_exact"].values()) and all(proof["cuda_cues_exact"].values()),"New proof incomplete")
    with np.load(m.ev.checked_ref(cell["original_prediction"]),allow_pickle=False) as saved, np.load(m.ev.checked_ref(cell["cpu_vectors"]),allow_pickle=False) as vectors:
        for head in failed:
            cpu=vectors[cell["vector_keys"][head]];gpu=saved[f"{head}__color_logit"]
            m.require(cpu.dtype==np.float32 and gpu.dtype==np.float32 and cpu.shape==gpu.shape==(640,),"Vector contract")
            m.require(d.array_hash(cpu)==cell["cpu_color_logit_hashes"][head],"CPU vector hash mismatch")
            m.require(not np.allclose(cpu,gpu,rtol=1e-5,atol=1e-5),"Exception has no original guard failure")
            cues=m.cue_indices(saved[f"{head}__generated_color_cues"])
            rows.append(dict(role=cell["role"],seed=cell["seed"],recording_id=cell["recording_id"],order_index=cell["order_index"],
                head=head,checkpoint_sha256=cell["checkpoint"]["sha256"],original_prediction=cell["original_prediction"],
                donor_id=cell["donor_id"],input_hashes=cell["input_hashes"],cpu_color_logit_sha256=d.array_hash(cpu),
                saved_color_logit_sha256=d.array_hash(gpu),cue_inventory_sha256=m.tr.json_hash(cues.tolist()),
                proof=cell["proof"],cpu_vectors=cell["cpu_vectors"],vector_key=cell["vector_keys"][head],proof_status="strict_roundoff_proved"))
require_keys={(r["role"],r["seed"],r["recording_id"],r["head"]) for r in rows}
m.require(len(rows)==len(require_keys)==6,"Expected six unique proved cell/head exceptions")
output=ROOT/"exception_map_heads_proposal_v1.json"
m.write_json(output,dict(schema_version=1,status="PROPOSAL_ONLY_NOT_AUTHORIZED",proof_status="all_listed_heads_strict_roundoff_proved",
    scope="Original allclose remains for every unlisted cell/head; listed CPU vector must match exacthash with source/input/checkpoint/savedGPU/cue bindings;no global tolerance change",
    array_hash_rule="sha256(str(np.ascontiguousarray(a).shape).encode()+str(np.ascontiguousarray(a).dtype).encode()+np.ascontiguousarray(a).tobytes());no dtype coercion",
    cpu_vector_contract="float32[640]",input_contract="float32[2560,94]",cue_hash_rule="training.json_hash(cues.tolist());cues=unique([0]+indices_of_saved_bool_mask)",
    original_driver=original["original_driver"],registration=original["registration"],source_map=m.ev.reference(source),
    preflight_complete=m.ev.reference(ROOT/"complete.json"),normalizer_source=m.ev.reference(__file__),
    rows=rows,head_count=len(rows),cell_count=len({(r["role"],r["seed"],r["recording_id"]) for r in rows}),authorized=False),m.DERIVED_ROOT)
print(json.dumps(m.ev.reference(output),indent=2))
