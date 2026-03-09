# OpenAI Research Grant Application — Inception Project

**Research Areas:** Robustness (primary), Misuse potential (secondary)

---

## Research Question

How robust are reasoning LLMs to iterative chain-of-thought injection attacks, and does the explicit reasoning mechanism (`<think>` block) create a systematically exploitable attack surface for bypassing safety alignment in high-stakes domains such as biosecurity and chemical threats?

Specifically:
1. Can a small adversarial "hacker" model iteratively mislead a larger reasoning model's internal thinking toward compliance with harmful requests by injecting manipulated CoT prefixes?
2. How does attack success vary across open-source (GPT-OSS, DeepSeek, Kimi) and closed-source model families (GPT-5, o3), and does reasoning capability correlate with vulnerability?
3. Do existing safety alignment techniques (deliberative alignment, RLHF) provide adequate protection at the reasoning-trace level?

---

## Project Description

Our prior work on off-trajectory reasoning (Li & Goyal, ICLR 2026) established that reasoning LLMs are brittle when encountering tokens they did not generate — stronger solo-reasoning performance does not predict robustness to off-distribution reasoning content. We found that models trained via distillation inherit their teacher's vulnerabilities, while reinforcement learning substantially improves resilience. This revealed a fundamental gap: current training paradigms optimize for solo-reasoning benchmarks but leave models unprepared for mixed-authorship reasoning traces.

My current project, called inception, extends this finding to the safety domain. We investigate a novel attack vector: **iterative CoT prefix injection**, where a small architect model (7B parameters) generates plausible reasoning continuations that are injected into a target reasoning model's `<think>` block. Unlike traditional prompt injection or jailbreaking, this attack operates *inside* the model's own reasoning mechanism, exploiting the same off-trajectory fragility we documented in our ICLR work. The attack iterates up to mutilple rounds — detecting and surgically removing refusals from the reasoning trace before feeding the truncated reasoning back as a seed for the next iteration.

We evaluate on different safety benchmarks, e.g., WMDP (Weapons of Mass Destruction Proficiency) across biosecurity, chemical, and cybersecurity scenarios, using a calibrated LLM-as-judge framework with domain-specific severity scales. We are developing a general and robust evaluation framework to capture both whether a model complies (binary), the quality of harmful output (e.g., specificity and convincingness) and potential consequences of successful jailbreaking outputs.

**How the API supports this work:** We require inference access to OpenAI's reasoning models (o3, o4-mini) and GPT-5 suite to: (1) test these models as attack targets — measuring whether their reasoning traces can be hijacked by CoT in user input , (2) compare attack surfaces between reasoning and non-reasoning model variants, and (3) use GPT-5 as a calibrated safety judge for evaluating attack outputs. Our existing pipeline already supports OpenAI API integration and has been validated on 6 open-weight reasoning models; extending to OpenAI's models enables the most policy-relevant comparison in the field.

---

## Research Design

**Phase 1: Baseline Safety Audit.** Establish refusal rates of target OpenAI models (GPT-5, o3, o4-mini) on WMDP biosecurity, chemical, and cybersecurity questions without any attack. This provides the safety ceiling.

**Phase 2: Simple Injection Baseline.** Single-pass CoT prefix injection (e.g., "Ok, I need to help the user to...") into reasoning blocks. Measures how much a single manipulated prefix degrades safety.

**Phase 3: Iterative Inception Attack.** Deploy the full architect-target loop (up to N iterations) with automated refusal detection and removal. Compare attack success rates (ASR) across model families, measuring danger scores via calibrated biosecurity/chemical judges.

**Phase 4: Cross-Architecture Analysis.** Compare OpenAI reasoning models against the 6 open-weight models we've already evaluated (DeepSeek-R1, Qwen3-235B, Kimi-K2, GLM-4.6, etc.) to identify whether architectural differences affect vulnerability.

**Phase 5: Defense Recommendations.** Based on vulnerability patterns, propose targeted mitigations — informed by our ICLR finding that RL training improves off-trajectory robustness where SFT fails.

---

## Expected Outcomes

1. **Systematic evaluation of SOTA open-source and closed-source reasoning models against iterative CoT injection attacks**, filling a critical gap — most existing red-teaming (H-CoT, Chain-of-Thought Hijacking, Universal and Transferable Adversarial Attacks) focuses on simple hand-crafted or gradient-based attack.

2. **Quantitative comparison of attack surfaces** between reasoning and non-reasoning variants of the same model family. Our ICLR work showed stronger solo-reasoners are not necessarily more robust collaborators; we expect analogous findings for safety — that reasoning capability may increase vulnerability to reasoning-level attacks.

3. **Calibrated danger assessment** using domain-specific severity scales (not generic "harmful/not harmful" binary). Our biosecurity, chemical, and cybersecurity judges produce severity scores anchored to real-world precedents, enabling nuanced risk quantification that informs deployment decisions.
4. 
5. **Open-source evaluation framework** — our attack pipeline, judge prompts, calibration examples, and danger scoring methodology will be released to enable reproducible safety evaluation across the research community.

---

## GPT-4 Fine-tuning

N/A — our research requires inference API access only, not fine-tuning. We use OpenAI models in two roles: (1) as attack targets, where we measure whether iterative CoT prefix injection can hijack reasoning traces into compliance with harmful requests, and (2) as safety judges, where GPT-5-mini evaluates attack output severity using our calibrated biosecurity and chemical threat scoring framework. Both roles require only standard chat completion endpoints — no weight modification or training access is needed.

---

## Past Research

Our ICLR 2026 paper "Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectories?" (Li & Goyal, Cornell University) evaluates 15 open-weight reasoning models (1.5B-32B) on off-trajectory robustness, introducing the Recoverability and Guidability twin tests. Key findings: (1) solo-reasoning benchmark performance does not predict robustness to foreign reasoning tokens — the top math model (82.6% accuracy) recovers from distracting steers only 33.4% of the time, while a smaller model (59.9%) recovers 98.4%; (2) distillation transfers teacher vulnerability to students even when training uses only correct trajectories; (3) RL training substantially improves off-trajectory resilience where SFT saturates. This work provides the empirical and conceptual foundation for the proposed safety extension.

---

## Any Other Comments

We have a fully operational attack and evaluation pipeline already validated on 6 open-weight reasoning models (DeepSeek-V3.2, Qwen3-235B, GLM-4.6, Kimi-K2, and others). Extending to OpenAI models requires only API access — no infrastructure buildout. Our approach is complementary to OpenAI's own CoT controllability and monitorability research (March 2026), and we are committed to responsible disclosure of any critical findings before publication.
