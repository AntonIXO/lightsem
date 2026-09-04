# vibescience-mcp API notes (learned by hitting the validation errors)

Every one of these cost a round-trip. The server validates hard and its error
messages are good — read them, they name the fix. Recorded here so the next
session does not rediscover them.

## rule_out

```python
c.call("rule_out", {
    "problem_id": ...,
    "claim":  "the claim being ELIMINATED",
    "method": "HOW it was eliminated",   # REQUIRED. 'measurement' is not a field.
    "evidence": "the numbers that killed it",  # REQUIRED
    "topic_tags": [...],                 # ONLY topic_tags. No problem_tags here.
})
```

- `method` is required — "a rule-out with no method is a rumour".
- There is **no `problem_tags` parameter** on `rule_out`, even though tags are
  registered on two axes. So a problem-axis tag like `matched-baseline` cannot
  be attached to a rule-out at all. **Proposed upstream feature**: accept
  `problem_tags` on `rule_out`, since a rule-out *is* a failure mode.

## Tags are a registered vocabulary

Not free text. `cross-modal-transfer` was rejected with a suggestion of
`cross-modal`. Tags also live on **axes** (`topic` vs `problem`) and using one
on the wrong axis is an error. Check with `list_tags(axis=...)`, create with
`register_tag(id=..., axis=...)`.

## Diagnostics must exist before they can be predicted against

`propose_hypothesis(predicted_effects=[{diagnostic_id, ...}])` rejects any
`diagnostic_id` that is not registered. Call `register_diagnostic` first.

## Two different direction enums — do not mix them

| Call | Field | Vocabulary |
|---|---|---|
| `register_diagnostic` | `direction` | `higher_better` / `lower_better` / `neutral` |
| `propose_hypothesis` | `predicted_effects[].direction` | `up` / `down` / `none` |

Passing `up` to `register_diagnostic` fails with `'up' is not a valid DiagDirection`.

## propose_hypothesis

```python
{
  "problem_id": ..., "statement": ...,
  "predicted_effects": [                 # REQUIRED, >=1, list of dicts (not prose)
      {"diagnostic_id": ..., "direction": "up", "magnitude_note": ...}],
  "rationale": ..., "plan": ...,
  "gates": [{"id": ..., "description": ...}],   # id+description, NOT name/criterion
  "topic_tags": [...], "problem_tags": [...],
}
```

## start_experiment

Takes `hypothesis_id`, `git_ref`, `external_run`, `config_note`,
`parent_experiment_id`. There is **no `title`, no `plan`, no `topic_tags`** —
the plan belongs on the hypothesis. Put run parameters in `config_note`.

## record_diagnostics

Parameter is `measurements`, **not** `diagnostics`:
`[{diagnostic_id, before, after}]`.

## record_gate_results

Parameter is `results`: `[{gate_id, passed, evidence}]`. Gates are recorded
separately from the experiment close because **the server deliberately does not
compute gates** — it records what was committed to, so "the direction was right"
can never be reported as "the claim was established".

## close_experiment

Signature is only `(experiment_id, notes)`. **You cannot assert an outcome.**
The verdict is COMPUTED from the preregistered `predicted_effects` versus the
recorded measurements, and a failing blocking gate downgrades `supports` to
`directional_only`. Passing `outcome="supported"` is rejected.

Returns e.g.:
```json
{"verdict": "supports",
 "prediction_match": {"per_diagnostic": {"event-ap-lift-audio-marginal": true},
                      "overall": true},
 "observed_effects": [{"diagnostic_id": "...", "delta": 0.2477, "direction": "up"}]}
```

## Vault separation (LightSem vs optiHealth)

Both server instances self-report a name; if identical, Hermes collapses the
namespaces and one vault silently shadows the other. `log_to_vault.py` pins
`VIBESCIENCE_VAULT=/root/lightsem-vault` and
`VIBESCIENCE_SERVER_NAME=vibescience-lightsem` over stdio JSON-RPC so records
cannot land in the wrong vault regardless of gateway state.

**Verify after every write session:** `/root/vibescience-vault/rule_outs/`
must stay at 14 files.

## stdio client pitfall

Feeding all requests via `subprocess.run(input=...)` closes stdin, and the
server exits on EOF after replying to `initialize` only — later responses are
never produced. Keep stdin OPEN and read responses incrementally.

## register_runbook has no `title`

`register_runbook(id, problem_id, prompt, priority, gates, topic_tags)` — passing `title`
fails with `Unexpected keyword argument`. Put the title on the first line of `prompt`.
`update_runbook(runbook_id, note | prompt)` is the same. (2026-09-04)

## Validation errors are strings, not `{"ok": false}`

A rejected call returns the pydantic error text as a plain string. A helper that only
checks `r.get("ok") is False` on dicts lets the failure through. Assert
`isinstance(r, dict) and r.get("ok") is not False`. This is how `rb-ls2-07` went
unregistered for half a day while PLAN.md and four hypotheses referred to it. (2026-09-04)
