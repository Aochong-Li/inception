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
(nothing active)

## Recently Done
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

## Known Issues / Blockers
- Bio severity scale is non-monotone (level 4 = dozens dead > level 5 = single-digit dead)
- Chem examples 4 & 5 assign grade_m to Aum Shinrikyo's improvised sarin (should be grade_i)
- Bio ex 10 (1918 flu) uses wrong threat class (C instead of A) and wrong delivery (aerosol for natural P2P spread)
- Bio ex 8 (Rhodesian Anthrax) analysis says severity 8, JSON has 7
- Bio ex 2 (Minnesota Ricin) delivery_method/route contradict each other and history (DMSO plan)
- See docs/calibration-review.md and docs/eval-field-audit.md for full audit
