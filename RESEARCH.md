# Inception — LLM Safety Evaluation System

> LLM-as-judge evaluation pipeline for biosecurity and chemical threat severity classification, using calibrated examples and structured JSON output.

## Current TODOs
- [ ] Fix high-priority calibration errors (bio ex 2 delivery/route, chem ex 4&5 grade, bio ex 10 threat/delivery)
- [ ] Fix bio ex 8 severity 7→8 (self-contradictory — analysis says 8, JSON assigns 7)
- [ ] Decide: replace bio ex 10 (1918 flu) with deliberate bioweapon scenario for better severity-10 ceiling
- [ ] Fix bio severity scale non-monotonicity (lvl 4 = dozens dead > lvl 5 = single-digit dead)
- [ ] Add incapacitant to chem agent_class enum (or add routing rule for opioid/CNS agents)
- [ ] Add chem injection delivery example (zero examples currently)
- [ ] Reassign bio ex 6 (WWI Livestock) severity 3→6

## In Progress
- [ ] Kimi-K2.5 judge eval — NVFP4 on 4x B200 hangs during post-loading init (sglang 0.5.9 + modelopt quantization issue). Options: DeepInfra API or wait for 8x B200 for BF16.
- [ ] Danger score aggregation for Qwen3.5-397B eval

## Recently Done
- Qwen3.5-397B-A17B-FP8 full judge eval completed — **34/34 models, 27,200 rows, 0 empty responses** — results in `evaluation/eval_qwen397b_judge/`
- 297+14 context-length failures patched via DeepInfra API (see eval ops notes below)
- Re-ran `ablation/think/tokens_128/GPT-OSS-120B` after uncorrupted input pickle restored
- Replaced bio/chem historical calibration examples with new ones (591bbfe)
- Finalized prompt template (280a8c4)
- Updated severity ladder ordering (79e4ec5, 0aebca9)
- Deep calibration review: docs/calibration-review.md
- Field coverage audit: docs/eval-field-audit.md
- Integrated StrongReject evaluator prompt into step 4 of judge template

## Key Decisions
- Prompt template lives in `evaluation/prompts/` with `loader.py` for dynamic loading
- Judge uses two separate templates: biosecurity + chemical, each with 10 calibration examples
- Schema fields are lowercase, user query field included
- StrongReject evaluator integrated for step 4
- `late_refusal` field is collected but NOT used in danger_score formula (dead field, low priority)

## Eval Ops Notes

### Serving Qwen3.5 on B200
- **vLLM 0.18.0 crashes** on Qwen3.5 (hybrid Mamba+Attention arch) during KV cache profiling. Use **SGLang** instead.
- Launch command requires `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (anaconda libstdc++ too old for FlashInfer MoE kernels).
- Use `--attention-backend triton` (FlashInfer JIT fails with nvcc 12.8 on sm_100a/Blackwell).
- First startup ~30-40 min (DeepGEMM warmup + FlashInfer MoE JIT), subsequent starts use cached kernels at `~/.cache/flashinfer/`.
- See memory file `sglang_qwen35_serving.md` for full launch command.

### Handling context-length BadRequestError in local eval
When running local judge eval (SGLang/vLLM), some samples exceed the server's `--context-length` limit. The error looks like:
```
BadRequestError: Requested token count exceeds the model's maximum context length of 32768 tokens.
```
**Recovery procedure:**
1. Extract failed rows: find all rows in `*_safety_judge.pickle` where `error` contains "context length"
2. Save to a single pickle with `_source_file` and `_source_idx` metadata
3. **Drop stale judge columns** before re-running (response, complied, etc.) — otherwise SafetyEvaluator merge creates `_eval` suffixed duplicates instead of populating the real columns
4. **Reset index** (`reset_index(drop=True)`) — failure rows have scattered indices from different source files; SafetyEvaluator needs sequential 0-N indices for internal merge
5. Re-run via API (e.g. `client_name="deepinfra"`) with same SafetyEvaluator — the API provider has longer context
6. Verify output: check complied/specificity/convincingness are populated, compare distributions against original eval
7. Patch back using `_source_file` + `_source_idx` to write judge columns into the original pickles
8. Fix dtype mismatches when patching: cast specificity/convincingness/severity_level/fatalities to float64

### Data integrity
- Input pickle `results/max_iterations_1/think/architect_initial_max_tokens_128/GPT-OSS-120B.pickle` is **corrupted** (512K, truncated). Needs re-generation.
- `evaluation/inception_data` is a symlink to `results/`

## Known Issues / Blockers
- Bio severity scale is non-monotone (level 4 = dozens dead > level 5 = single-digit dead)
- Chem examples 4 & 5 assign grade_m to Aum Shinrikyo's improvised sarin (should be grade_i)
- Bio ex 10 (1918 flu) uses wrong threat class (C instead of A) and wrong delivery (aerosol for natural P2P spread)
- Bio ex 8 (Rhodesian Anthrax) analysis says severity 8, JSON has 7
- Bio ex 2 (Minnesota Ricin) delivery_method/route contradict each other and history (DMSO plan)
- See docs/calibration-review.md and docs/eval-field-audit.md for full audit
