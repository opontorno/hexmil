# Paper Structure — ICCV / CVPR / ECCV (8 pages, two-column)

**Working title**: *Weakly-Supervised Deepfake Detection in Medical CT Volumes via Hierarchical Attention MIL*

> **Title alternatives** (sharper for top-tier):
> - *Hierarchical Attention MIL for Weakly-Supervised CT Manipulation Detection and Localization*
> - *HEXMIL: Ante-Hoc Explainable Detection of AI-Manipulated CT Volumes under Weak Supervision*
> - *From Volume Label to 3D Heatmap: Weakly-Supervised CT Forgery Detection via Hierarchical MIL*

**Model name**: **HEXMIL** (Hierarchical EXplainable Multiple Instance Learning)

---

## Page budget overview

| Section | ~Words | ~Lines (two-col) | Pages |
|---|---|---|---|
| Abstract | 150 | 20 | 0.15 |
| §1 Introduction | 700 | 90 | 0.7 |
| §2 Related Work | 550 | 70 | 0.55 |
| §3 Proposed Method | 1 400 | 180 | 1.4 |
| §4 Experiments | 1 400 | 180 | 1.4 |
| §5 Analysis | 500 | 65 | 0.5 |
| §6 Conclusion | 200 | 25 | 0.2 |
| Figures (2) | — | — | ~2.0 |
| Tables (2–3) | — | — | ~0.6 |
| References (~35) | — | — | ~0.7 |
| **Total** | **~4 900** | | **~8.2 pp** |

> **Space note**: if the budget is tight, §5.2 Robustness can be reduced to 2 inline sentences
> + one figure panel. §4.3 and §5.1 are the two sections that must survive any cut.

---

## Abstract (~150 words)

Four sentences, in order:

1. **Problem** — The proliferation of AI-generated CT manipulation poses a critical threat to medical
   imaging pipelines; existing detectors rely on patch-level supervision or 2D-only forensics that
   do not scale to volumetric data.
2. **Method** — We propose HEXMIL, a hierarchical Attention-Based MIL framework that decomposes a
   CT volume into patches → slices → volume, applying independent gated attention at each level and
   training exclusively on volume-level binary labels.
3. **Explainability** — The joint attention product β·α yields a 3D heatmap that is *ante-hoc* —
   it is the decision mechanism itself — and is optionally refined by an auxiliary KL loss that
   aligns patch attention with ground-truth mask coverage on a subset of training volumes.
4. **Results** — On M3DSynth (pix2pix, CycleGAN, diffusion), HEXMIL achieves strong AUC/F1 in
   both in-domain and OOD settings, and produces spatially coherent heatmaps without any
   localization supervision, outperforming post-hoc attribution methods in pointing-game score.

**Key claim to defend**: no spatial annotation is required for both classification *and* localization.

---

## §1 Introduction (~700 words)

### Narrative arc (4 blocks)

---

#### Block 1 — Hook: clinical threat (2–3 sentences, ~80 words)

Open on impact, not on technology. Reviewers must feel the problem is real in the first paragraph.

> *"The rapid advancement of generative AI has introduced a critical but underexplored vulnerability
> in medical imaging: the ability to synthesise or selectively alter findings in CT volumes in ways
> imperceptible to radiologists and standard diagnostic pipelines. A manipulated scan could
> suppress evidence of malignancies, inject false lesions, or alter measurements, with direct
> consequences for triage, treatment planning, and insurance fraud."*

**Key citations here**: M3DSynth [cite], Mirsky & Lee 2021 (deepfakes survey).

---

#### Block 2 — Existing approaches + their fundamental limit (3–4 sentences, ~150 words)

Do NOT write a survey — that is §2. The goal here is contrast: what exists, why it fails for your task.

> *"Existing deepfake detectors operate primarily in the 2D image domain, relying on CNN classifiers
> trained on patch crops [Wang et al. 2020, Rossler et al. 2019] or network-forensics signatures
> tied to specific GAN architectures. When applied to CT, these methods analyse individual slices
> independently, ignoring the spatial coherence of manipulation artefacts across the axial dimension.
> More critically, the few methods designed for medical image forensics [MVSS-Net] require
> pixel-level binary segmentation masks during training — supervision that is prohibitively expensive
> to obtain at scale in a clinical setting."*

**Key citations**: Wang et al. CVPR 2020, Rossler et al. ICCV 2019, MVSS-Net (Chen et al. ICCV 2021).

---

#### Block 3 — Gap statement + solution overview (~150 words)

The gap sentence is the single most important sentence in the paper — it sets up your entire
contribution. Write it as a sharp open question, then answer it.

**Gap**:
> *"We address the following open problem: can a single volume-level binary label — real or fake —
> be sufficient to both detect and spatially localise CT manipulation, without any patch-level or
> slice-level annotation?"*

**Solution** (narrative, no formulas):
> *"We propose HEXMIL, a hierarchical MIL framework that processes a CT volume in two stages.
> First, patches within each slice are aggregated via gated attention (α) into a slice representation
> and classified with a binary head; an optional auxiliary KL loss steers α towards manipulated
> regions using masks available for a subset of training volumes, without supervising the
> classification head. Second, slice representations are aggregated across the volume by an
> independent gated attention module (β). Their product β·α assembles an ante-hoc 3D heatmap —
> not a post-hoc saliency approximation, but the exact weighted sum driving the classification
> decision, faithful by construction."*

---

#### Block 4 — Contributions (numbered list, ~120 words)

Use active verbs. Avoid "we present" and "we propose" for every bullet — vary them.

1. **We formulate** CT manipulation detection as a hierarchical two-stage MIL problem over
   patches → slices → volumes, requiring only volume-level binary supervision.
2. **We introduce** an optional auxiliary KL loss that aligns patch-level attention with
   ground-truth mask coverage on a subset of training volumes, without supervising the
   classification head, preserving the weakly-supervised nature of the framework.
3. **We derive** an ante-hoc 3D attention heatmap faithful by construction: removing the attention
   mechanism collapses the model to uniform averaging and directly degrades classification.
4. **We demonstrate** on M3DSynth — across three GAN modalities (pix2pix, CycleGAN, diffusion)
   and an out-of-distribution setting — that HEXMIL yields spatially coherent heatmaps
   competitive with Grad-CAM and GradCAM++ under identical (zero localization) supervision.

---

#### Writing rules for §1

- **Do NOT** open with "Deep learning has revolutionized..." — every ICCV reviewer hates it.
- **Do NOT** claim to beat supervised localization SOTA — frame localization as emergent interpretability.
- **Do NOT** put equations in the introduction.
- **DO** use *we* throughout (not "the proposed method").
- **DO** make sure block 3 gap sentence would make a compelling oral-presentation one-liner.
- **DO** write contributions in order of novelty, not chronological order of the pipeline.

---

## §2 Related Work (~550 words)

No numbered subsections — use `\smallskip\noindent\textbf{...}` for each paragraph title.
Every paragraph **must close with a sentence positioning HEXMIL against that line of work**.

---

### ¶1 — Deepfake Detection in Natural Images (~120 words)

Survey arc: binary classifiers → GAN-fingerprint methods → face-specific → frequency domain.

Closing sentence:
> *"While these methods achieve strong performance on face forgeries, they operate on single 2D
> frames and exploit domain-specific priors — facial geometry, blending boundaries — that do not
> transfer to medical CT volumes, where manipulation spans a contiguous 3D cuboid region and
> leaves no face-specific artefact."*

**Citations**:
| Paper | Role |
|---|---|
| Rossler et al. ICCV 2019 — FaceForensics++ | Canonical 2D benchmark |
| Wang et al. CVPR 2020 — *CNN-Generated Images* | GAN fingerprint detection |
| Li & Lyu CVPR 2020 — Face X-Ray | Face-specific blending detector |
| Frank et al. ECCV 2020 — Frequency analysis | Frequency-domain artefacts |
| Tolosana et al. 2020 survey (optional) | Rapid coverage of the field |

---

### ¶2 — Medical Image Forensics (~120 words)

Describe M3DSynth in detail here (it is your dataset — reviewers need this context).
Cover any existing 2D or 3D medical forensics work.

Closing sentence:
> *"All prior medical forensics methods either operate slice-by-slice, discarding volumetric
> context, or require dense pixel annotations during training. HEXMIL is the first to address
> full-volume CT manipulation detection and localization under volume-level weak supervision."*

**Citations**:
| Paper | Role |
|---|---|
| **M3DSynth** [dataset paper] | Your dataset — describe the 3 modalities, removal+injection task |
| MVSS-Net (Chen et al. ICCV 2021) | Multi-view supervised manipulation detection — supervised contrast |
| Gragnaniello et al. / Marra et al. | Medical-domain forensics, if applicable |
| Mirsky & Lee 2021 survey | Deepfake threat in medical imaging |

---

### ¶3 — Multiple Instance Learning (~150 words)

Arc: Dietterich 1997 (origin) → deep MIL → attention MIL (ABMIL) → computational pathology
applications (CLAM, DSMIL, TransMIL) → your extension to 3D volumes.

Closing sentence:
> *"Existing MIL methods for medical imaging target 2D histology slides; HEXMIL extends the
> paradigm to 3D CT volumes by introducing a second aggregation level (patch → slice → volume),
> with an explicit two-stage training protocol that decouples low-level forensic feature learning
> from inter-slice evidence aggregation."*

**Citations (all important — do not drop)**:
| Paper | Role |
|---|---|
| Dietterich et al. 1997 | MIL origin |
| **Ilse et al. ICML 2018** — ABMIL | Direct foundation of your method |
| Lu et al. *Nature Biomed. Eng.* 2021 — CLAM | Dominant MIL in pathology, high citations |
| Li et al. CVPR 2021 — DSMIL | Dual-stream MIL |
| Shao et al. NeurIPS 2021 — TransMIL | Transformer-based MIL |
| Chen et al. CVPR 2022 — HIPT (optional) | Hierarchical ViT for pathology (closest hierarchical analogue) |

---

### ¶4 — Explainability in Medical Imaging (~120 words)

Key distinction: post-hoc methods produce *approximations* of saliency; ante-hoc attention
*is* the decision mechanism. This paragraph sets up §4.4 and the faithfulness argument.

Closing sentence:
> *"Unlike all post-hoc methods, the attention weights in HEXMIL are the exact aggregation
> function that produces the classification score — a constructive faithfulness guarantee that
> Grad-CAM and its variants cannot provide, as demonstrated empirically by the sanity checks
> of Adebayo et al. [2018]."*

**Citations**:
| Paper | Role |
|---|---|
| Zhou et al. CVPR 2016 — CAM | Origin of class activation mapping |
| **Selvaraju et al. ICCV 2017** — Grad-CAM | Baseline you compare against |
| **Chattopadhyay et al. WACV 2018** — GradCAM++ | Baseline you compare against |
| **Adebayo et al. NeurIPS 2018** — *Sanity Checks* | Fundamental critique of post-hoc saliency |
| Abnar & Zuidema ACL 2020 — Attention rollout (optional) | Faithfulness for Transformers |

---

#### Writing rules for §2

- Each paragraph: 2–4 sentences max — this is positioning, not a survey.
- The closing separator sentence for each paragraph is non-negotiable — it does the work.
- Do NOT criticize specific methods harshly — prefer "unsuitable for our setting" over "flawed".
- Cite as `[A, B, C]` inline, not as footnotes.

---

## §3 Proposed Method (~1 400 words)

→ **Already written**: `paper/proposed_method.tex`

Subsections:
- §3.1 Problem Formulation
- §3.2 Slice-Level Patch Aggregation (Gated ABMIL, α)
- §3.3 Volume-Level Slice Aggregation (Gated ABMIL, β)
- §3.4 Two-Stage Training (L_cls + L_aux KL, λ=0.3)
- §3.5 3D Attention Heatmap and Interpretability (ante-hoc framing — updated)

**Figure 1** (full-width, after §3.3 ~p.4): architecture pipeline
Caption: see `proposed_method.tex`

---

## §4 Experiments (~1 400 words)

### §4.1 Experimental Setup (~200 words)

- **Dataset**: M3DSynth — 3 GAN modalities (pix2pix, CycleGAN, diffusion), removal + injection
- **Splits**: standard train/val/test from M3DSynth
- **Backbone**: ResNet-50 pretrained on ImageNet; patch 128×128, stride 64
- **Training**: Adam, lr=1e-4, BCE + optional KL (λ=0.3), two-stage; AMP
- **Metrics**: AUC, ACC, AP, F1 — global + per-modality balanced; pd@1% (TPR at FPR=1%)
- **Baselines**: flat CNN classifier (ResNet-50 + global pool), ABMIL without aux loss

---

### §4.2 Classification Performance & OOD Generalisation (~600 words)

Unified narrative: the first half covers in-domain performance, the second half covers the OOD
matrix. A single table handles both — the OOD columns naturally extend the in-domain ones,
making the generalization gap visible at a glance.

**Table 1** — main results (in-domain diagonal + OOD off-diagonal):

| Train → | pix2pix AUC | CycleGAN AUC | Diffusion AUC | Global AUC | ACC | F1 |
|---|---|---|---|---|---|---|
| *Trained on pix2pix* | | | | | | |
| Flat CNN | | | | | | |
| ABMIL (no aux) | | | | | | |
| **HEXMIL** (ours) | | | | | | |
| *Trained on CycleGAN* | | | | | | |
| Flat CNN | | | | | | |
| ABMIL (no aux) | | | | | | |
| **HEXMIL** (ours) | | | | | | |
| *Trained on Diffusion* | | | | | | |
| Flat CNN | | | | | | |
| ABMIL (no aux) | | | | | | |
| **HEXMIL** (ours) | | | | | | |

- **In-domain** (diagonal): HEXMIL should match or exceed ABMIL (no aux) — show that the
  two-stage training does not hurt and that aux_attn_loss is neutral on AUC
- **OOD** (off-diagonal): hierarchical β learns manipulation *location* independent of GAN style;
  the flat CNN is brittle because it learns GAN-specific texture fingerprints
- Report Δ-AUC (in-domain vs best OOD) per model — the gap is smaller for HEXMIL → key claim
- pd@1% (TPR at FPR=1%) reported inline in text, not in table (saves a column)

**Key framing sentences**:
> *"The flat CNN achieves competitive in-domain AUC but collapses under distribution shift,
> confirming that it memorises GAN-specific texture fingerprints rather than learning
> manipulation-invariant features."*

> *"HEXMIL's β weights consistently peak on the axial range containing the manipulation,
> providing a structurally grounded inductive bias that transfers across GAN modalities."*

---

### §4.3 Interpretability Analysis (~600 words)

**Frame**: *"We evaluate the spatial coherence of the emergent attention heatmaps without any
localization supervision."* Never write "localization performance" — always "spatial coherence of
the emergent heatmaps".

**Figure 2** (single-column, ~p.6): XAI visual comparison grid

```
Rows:    CT slice | GT mask | ABMIL attn (ante-hoc) | Grad-CAM | GradCAM++
Columns: pix2pix  | CycleGAN | diffusion
         (central manipulation slice per volume)
```

Generated by: `experiments/ABMIL/compare_xai.py`

**Content order** (most to least convincing):

1. **Quantitative** — Pointing Game score (primary) + IoU_0.5 (secondary, inline or micro-table)
   - HEXMIL vs Grad-CAM / GradCAM++ under identical (zero localization) supervision
   - Note: IoU is an emergent metric, not a supervised localization benchmark
2. **Faithfulness argument** — ante-hoc vs post-hoc; cite Adebayo 2018
3. **Qualitative** — Figure 2: ABMIL attention consistently highlights the manipulation region;
   Grad-CAM/GradCAM++ are noisier and less spatially coherent

**Key framing sentences**:
> *"Unlike Grad-CAM, which approximates saliency via backpropagation through a frozen model,
> the ABMIL attention weights are the aggregation function — removing them collapses the model
> to uniform averaging, directly degrading classification."*

> *"Both Grad-CAM and GradCAM++ are applied to the same backbone with no retraining, providing
> a fair comparison under identical (zero localization) supervision."*

---

## §5 Analysis (~500 words)

This section stands independently from §4 to signal methodological rigor. Reviewers often
read this section before the main results table — it must be self-contained.

---

### §5.1 Ablation Study (~280 words)

One table, no prose repetition. Remove one component at a time and measure the impact on both
AUC (classification) and Pointing Game (spatial coherence) — this dual-metric view makes each
component's role unambiguous.

**Table 2** — ablation on the pix2pix split (representative; other splits in supplementary):

| Model variant | AUC ↑ | Pointing Game ↑ | IoU_0.5 ↑ |
|---|---|---|---|
| HEXMIL (full) | | | |
| w/o aux_attn_loss (λ=0) | | | |
| w/o Stage 2 (slice-only) | | | |
| w/o β-gating (τ_β = 0) | | | |
| α uniform (no patch attention) | | | |

**Reading the table** — what each row proves:

| Row removed | AUC impact | Pointing Game impact | Message |
|---|---|---|---|
| aux_attn_loss | ~stable | ↓ | KL steers spatial coherence without touching classification |
| Stage 2 | ↓ | ~stable | Volume-level aggregation is necessary for classification |
| β-gating (τ_β=0) | ~stable | ↓ | β filters uninformative slices, essential for clean heatmap |
| α uniform | ↓↓ | ↓↓ | Attention is the decision mechanism — ante-hoc claim quantified |

The last row ("α uniform") is the strongest result: it quantifies the ante-hoc claim directly.
If AUC degrades when attention is uniform, attention is not decorative — it drives the prediction.

---

### §5.2 Robustness (~220 words)

Two experiments only — keep prose tight, one figure or one small table each.

**Experiment A — Signal corruption** (simulates real DICOM transmission conditions):

Apply post-hoc degradation to test volumes, measure AUC and Pointing Game degradation:

| Degradation | Flat CNN ΔAUC | HEXMIL ΔAUC |
|---|---|---|
| JPEG q=90 | | |
| JPEG q=75 | | |
| JPEG q=50 | | |
| Gaussian noise σ=0.05 | | |
| Gaussian noise σ=0.10 | | |

Expected result: HEXMIL degrades more gracefully because aggregating over N patches averages out
per-patch noise. The flat CNN, relying on global texture statistics, is more brittle.

**Experiment B — Attention faithfulness perturbation test**:

Progressively mask the top-k% patches (by α weight) and measure AUC drop, compared to masking
k% *random* patches:

> If removing the most-attended patches degrades AUC more than removing random patches, the
> attention is genuinely informative — not a spatially arbitrary by-product of training.

Report for k ∈ {10%, 20%, 30%}. Plot as a curve (AUC vs % patches removed, two lines:
top-α vs random). This provides a quantitative faithfulness certificate that post-hoc methods
(Grad-CAM) cannot produce without retraining.

**Citation**: Samek et al. 2017 — *Evaluating the Visualization of What a Deep Neural Network
has Learned* — the canonical reference for perturbation-based faithfulness evaluation.

---

## §6 Conclusion (~200 words)

1. One paragraph summary of contributions — echo §1 contributions, do not repeat verbatim
2. **Limitations** (be explicit — reviewers respect honesty):
   - Localization quality depends on patch size; manipulations smaller than one patch are
     invisible to the attention mechanism
   - aux_attn_loss requires binary masks for a training subset; annotation cost not zero
   - Two-stage training increases wall-clock time vs a single-stage baseline
3. **Future work** (be concrete, not generic):
   - 3D backbone (SwinUNETR, ViT-3D) to capture cross-slice context at the encoder level
   - Semi-supervised mask generation (SAM, weakly-supervised segmentation) to extend
     aux_attn_loss without manual annotation
   - Cross-dataset evaluation (M3DSynth → clinical distributions, different scanners)

---

## References (~35 entries, ICCV style)

### Must cite

| Citation key | Paper | Why |
|---|---|---|
| ilse2018attention | Ilse et al. ICML 2018 — ABMIL | Foundation of the method |
| he2016deep | He et al. CVPR 2016 — ResNet | Backbone |
| selvaraju2017gradcam | Selvaraju et al. ICCV 2017 — Grad-CAM | XAI baseline §4.3 |
| chattopadhyay2018gradcam++ | Chattopadhyay et al. WACV 2018 — GradCAM++ | XAI baseline §4.3 |
| adebayo2018sanity | Adebayo et al. NeurIPS 2018 — Sanity checks | Faithfulness critique §4.3 |
| samek2017evaluating | Samek et al. IEEE TNNLS 2017 — *Evaluating DNN Visualization* | Perturbation faithfulness §5.2 |
| m3dsynth | M3DSynth dataset paper | Dataset |
| chen2021mvssnet | Chen et al. ICCV 2021 — MVSS-Net | Supervised localization contrast §2, §4.2 |
| rossler2019faceforensics | Rossler et al. ICCV 2019 — FaceForensics++ | 2D forensics context §2 |
| wang2020cnn | Wang et al. CVPR 2020 — CNN-Generated images | GAN detection, OOD brittleness §2 |
| lu2021clam | Lu et al. Nature Biomed. Eng. 2021 — CLAM | MIL in pathology, dominant §2 |
| li2021dsmil | Li et al. CVPR 2021 — DSMIL | MIL comparison §2 |
| shao2021transmil | Shao et al. NeurIPS 2021 — TransMIL | MIL comparison §2 |
| dietterich1997mil | Dietterich et al. 1997 — MIL origin | Historical context §2 |
| mirsky2021deepfakes | Mirsky & Lee 2021 — Deepfakes survey | Threat motivation §1 |

### Recommended additions

| Citation key | Paper | Why |
|---|---|---|
| li2020facexray | Li & Lyu CVPR 2020 — Face X-Ray | Face-specific forensics, domain gap §2 |
| zhou2016cam | Zhou et al. CVPR 2016 — CAM | XAI lineage §2 |
| chen2022hipt | Chen et al. CVPR 2022 — HIPT | Hierarchical ViT for pathology, structural analogue §2 |
| frank2020frequency | Frank et al. ECCV 2020 — Frequency analysis | GAN frequency forensics §2 |
| abnar2020rollout | Abnar & Zuidema ACL 2020 — Attention rollout | Faithfulness for Transformers §2 |

---

## Figure checklist

| Fig | Location | Description | Generated by |
|---|---|---|---|
| Fig. 1 | After §3.3 (~p.4) | Architecture pipeline — two-stage HEXMIL (full-width) | Manual / TikZ |
| Fig. 2 | §4.3 (~p.6) | XAI comparison grid: CT \| GT mask \| ABMIL \| Grad-CAM \| GradCAM++ | `experiments/ABMIL/compare_xai.py` |
| Fig. 3 (optional) | §5.2 (~p.7) | Faithfulness perturbation curve: AUC vs % patches masked (top-α vs random) | inline script |

---

## Table checklist

| Table | Location | Description |
|---|---|---|
| Table 1 | §4.2 (~p.5) | OOD matrix: train modality × test modality, AUC/ACC/F1 per model |
| Table 2 | §5.1 (~p.7) | Ablation: AUC + Pointing Game + IoU_0.5 per removed component |
| Table 3 (optional) | §5.2 | Signal corruption: ΔAUC for Flat CNN vs HEXMIL under JPEG/noise |
| Inline micro-table | §4.3 | Pointing Game + IoU_0.5: HEXMIL vs Grad-CAM vs GradCAM++ |

---

## Global writing rules (ICCV standard)

### Tone and verbs
- Use **we** throughout — never "the proposed method" or "our model"
- Active verbs: *formulate, introduce, derive, demonstrate, show, report* — not *present* or *utilize*
- Each claim in the abstract must be verifiable from a table or figure number

### Things to NEVER write
- "Deep learning has revolutionized..." (every reviewer's red flag)
- "To the best of our knowledge..." (only if 100% certain; often cut by reviewers)
- "In this paper, we propose..." as the opening sentence of §1
- "It is worth noting that..." (padding)
- "State-of-the-art" as a noun

### Framing discipline
- **Never** say "localization performance" in the context of §4.4 — always "spatial coherence of the emergent heatmaps" or "pointing-game score"
- **Never** compare IoU against fully supervised localization SOTA (different task, different supervision)
- **Always** frame localization as an emergent interpretability property, not a primary result
- **Always** describe Grad-CAM/GradCAM++ as operating "under identical (zero localization) supervision" — this is the fairness anchor

### Space discipline (ICCV 8-page limit)
- Equations count as ~4 lines each — §3 has 9 equations, estimate ~36 lines consumed
- Each 3-column full-width figure ≈ 35–40 lines
- Cut adjectives before cutting content — most adverbs can go
- If §4 runs long, collapse OOD into Table 1 extra rows instead of Table 2

---

## Submission checklist

**Content**
- [ ] Abstract ≤ 150 words; all 4 claims verifiable from a table or figure number
- [ ] All 4 contributions in §1 map 1-to-1 to results in §4 or §5
- [ ] §4.2 Table 1 shows both in-domain (diagonal) and OOD (off-diagonal) — bold HEXMIL rows
- [ ] §4.3 reports Pointing Game score (primary) before IoU_0.5 (secondary)
- [ ] §4.3 includes Figure 2 with explicit caption (rows and columns described)
- [ ] §5.1 Table 2 ablation includes "α uniform" row — quantifies ante-hoc claim
- [ ] §5.2 perturbation curve shows top-α vs random masking for k ∈ {10,20,30%}
- [ ] Adebayo 2018 cited in §4.3 faithfulness argument
- [ ] Samek 2017 cited in §5.2 perturbation test
- [ ] No comparison of IoU with fully supervised localization SOTA anywhere in the paper

**Framing**
- [ ] "Localization" never appears without "emergent" or "weakly-supervised" qualifier
- [ ] "Robustness" section explicitly states no adversarial training was used
- [ ] OOD framing uses "distribution shift" not "domain adaptation"

**Technical**
- [ ] All citation keys resolve in .bib file
- [ ] All \ref{} targets exist (no undefined references)
- [ ] Paper compiles with `pdflatex` without warnings on \ref / \cite
- [ ] Figure captions self-contained (readable without the section text)
