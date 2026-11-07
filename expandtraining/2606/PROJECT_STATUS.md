# 2606 project status: history, progress, resources, tools

Written 2026-11-07 from `git log` (170 commits), `run_notes.yaml` (75 entries), `FINDINGS.md`, `prereg/*.md`, `outputs/` and `data/` on disk.
Goal: a style predictor that suppresses line/scene topic and keeps speaker-habitual voice. The target is style, not author.

## 1. History

| Dates | Phase | What happened | Outcome |
|---|---|---|---|
| Feb 25-27 | Genshin base | Data scripts, naive style vectors, hidden-layer analysis, LoRA training, speaker sampler, layer fusion, attention pooling. | Foundation only. |
| May 2-6 | Paper replications | StyleDistance (roberta-base, LoRA, triplet loss); multilingual xlm-roberta with 5-language SynthSTEL; curriculum triplet types. | Merged to main Jun 6. |
| Jun 24-25 | 2606 lane starts | Training core, profiles, smoke tests, vector probes and courses, remote configs (1h, 3h, 5090x2, Colab T4). | Two 0.6B runs (1h, 3h). |
| Sep 13 | Failure analysis | `FINDINGS.md`: the 1h/3h models learn topic and provenance, not style. `origin_vs_para` violated 63% (1h) and 26% (3h); linear probe recovers generated-vs-real at ~88-89% and domain at ~91%. Validation covers ~6% of triplets and no unseen speakers. Throughput and checkpoint bugs found. | Root causes: adjacent-row positives (scene shortcut), provenance axis, blind validation. |
| Oct 11-13 | Data and curriculum | bookraw OCR pipeline, held-out speakers, staged curriculum, one-variable ablation harness. | |
| Oct 17-18 | Run infrastructure | Prebuilt triplets, AutoDL push/pull, Stage 0 preflight, resumable queue, SwanLab, auto power-off, per-run cost cap (CNY 6), curriculum_a/a2 and spread_a screens. | curriculum arms within noise; staged worse than mixed on 2 seeds; centring kept. |
| Oct 20-23 | InfoNCE | In-batch InfoNCE: top-1 .108 -> .235 but content leaks (origin_vs_para gap -.262). Hinge and content head: top-1 .194, leak halved. | Retrieval up, leak up. |
| Oct 23-26 | Leak fixes | Content twins (T .03 removed leak but cost ~40% of top-1), HSIC (dose-dependent), adversary (breaks at w2), subspace null (fails), chead v2 (best whitened top-1 .369), hard-share 0.30, clean/conv/nce/burst/cap/scene screens. | nce_clean_twin chosen as reference: top-1 .184, whitened .357, line leak .707. Scene batching ruled out. |
| Nov 1-2 | Frozen LMs and head ablations | frozen_a/b/c (StyleVector diffs, neurons, linear heads): nothing beats nce_clean_twin at matched leak. head_a nohead: leak .662 (head is a partial source, ~1/5). notwin run was a silent no-op bug (fixed). head_b twinfix and adv: no change. head_c episode InfoNCE: top-1 .225, leak .754. seed_b: noise floor leak .017, top-1 .016. | Leak gaps >= .05 are real; Fisher differences < .5 are not. |
| Nov 3-4 | Named traits and more voices | Jev's 17 traits (pinned to jev-1.14.0), tone twins, multi-author corpora (Weibo, Douban, CPED, CommentR, Reddit, AuthBench, AO3). head_d: dims track Jev at r .69 but hold 1.6% of norm. head_e: norm floor works, trait r falls, fails go rule. head_f: 579 speakers lift top-1 (+.025); Reddit episodes lift cross-topic voice .533 -> .571; push adds leak. | Direction changed to a separate trait predictor. |
| Nov 4 | Pre-registrations | External style test (CPED primary, char TF-IDF baseline), voice-change ratings. | |
| Nov 5 | Trait predictor | 17-trait predictor on Qwen3-0.6B (LoRA r8 mean/attention, frozen taps). Sealed confirmation on 67 tagged pairs: **no pass, inconclusive** (the +.373 gain on the 95 did not carry over). Sealed Argon-scan test on 300 small edits: **pass** (293 usable; LoRA attention .613, +.201 over edit size, 99% CI .069-.341). Blind audit of 25 largest disagreements: rater sided with Argon 13, model 5, neither 5, broken 2 (p .096). | The arms cannot be ranked from this test. |
| Nov 6 | Stage 4b, 红楼梦, landing audit | Stage 4b scan fine-tune on AutoDL (.613, level with unmodified LoRA attention; +.025 over its start, CI99 -.002 to .051): closed. 红楼梦 chapter-80 spike (AUC .98) but content alone splits at 80; matched test fails balance (content AUC .715 vs .65 limit); Dialogue subset holds all five traits (.688 vs content .669), Narration does not. Landing audit (local, $0.04): training is all Chinese; zero-shot en .73 / ja .69; label transfer no better than the model (r .72); voice-vs-topic not above TF-IDF on CPED/Reddit. | Check before any multilingual or real-world training. |

## 2. Progress (current state)

- **Best model:** LoRA r8 attention readout on Qwen3-0.6B. It scores .613 on the 293 Argon-scan pairs, but that set is now a development set (seen once), so it is not a fresh sealed number.
- **Settled negatives:** no frozen method, content-head variant or leak penalty beat nce_clean_twin at matched leak; Stage 4b adds nothing over LoRA attention; the sealed 67-pair confirmation did not pass.
- **Settled positives:** the model ranks small edits more like Argon than edit size does on unseen speakers (+.201); zero-shot reading of en/ja is fair (r .73/.69 with its own zh reading).
- **Open weaknesses:** training data is all zh Genshin; real-world voice is at chance on CPED (.492) and Reddit (.511); CPED traits are flat (12 of 17 below half the Genshin SD); chapter 80 cannot separate style from content.
- **Git:** all code and results committed on `style-traits` (last: eef07e5); nothing pushed.
- **Blocked:** AutoDL balance is CNY 2.40, so no box run until you top up and confirm.

## 3. Open items

1. Decision: Jev labels for fresh en/ja lines (~$4 per 20,000 each) plus a small real-world sample (CPED, Weibo, Douban, Reddit).
2. Decide the success rule for voice: author-labelled V may be the wrong pass rule for a style target.
3. Fix the AutoDL launch (must return at once; a separate watcher pulls results and stops the box).
4. Doc update, push approval, head_d/e scoring choice, Q1 redesign, Arena bonus (deferred), GCP key revocation, `.gitignore` and `/Volumes/EMC/tempdata` decisions, reviewer relays queued (session 5f6995, offline).

## 4. Resources

**Compute**
- Mac: Apple M2 Pro, 32 GB, MPS. Internal disk 926 GB, 97% used, 33 GB free.
- Volumes: `/Volumes/EMC` (canonical book data volume; `data/bookraw` here is empty), `Arion`, `Arion 1`.
- AutoDL box `pro-7922bda1e505`: stopped. CNY 6 per-run cost cap, auto power-off at queue end. GPU stock is intermittent (start can retry for hours); a no-GPU start works only for syncing.
- SwanLab for run logs (notes capped at 1024 characters). TensorBoard logs in `tf-logs/`. Colab T4 notebook in `colab/`.

**Paid and external services**
- Jev `jev-1.14.0` for 17 style traits (0-4), ~$1 per 10,000 lines. Labels on disk: 20k zh, ~2,000 en/ja translation lines, 400 ja recheck lines, 4,615 paired lines.
- Argon (Gemini) as scan reference; Stage 4b closed, no more tokens there. Gemini is also used as reviewer (cap ~100k tokens per consult).
- Human rater (you): not the gold standard; report against both you and Argon.

**Data on disk (`data/`, 17 GB datasets)**
- Genshin: `genshin.db` 51 MB, `genshin_v2.db` 144 MB (conversations), `genshin.json`, AGD/textmap sources; 269,565 lines aligned zh/en/ja; paraphrases zh 269,592, en 90,033, ja 44,754; unseen pool 77,985 lines / 5,510 speakers; `paratarget` 2.2 GB.
- Author corpora (`authorcorpora`, 593 MB): `unified.parquet` 272 MB, `unified_new_sources.parquet` 321 MB.
- Real text: Reddit authorship 161 MB, Douban short 387 MB, Douban side 349 MB, Weibo 36 MB, CPED 26 MB, CommentR 2.1 GB, AuthBench 549 MB (zh/en/ja), Aozora 230 MB, StreamProfileBench 75 MB, MAC 3.7 GB, Arena-140k 1.5 GB, `corpora` 7.3 GB, MSynthSTEL 356 MB.
- Episodes: `reddit_train` (153,638 comments, 3,864 authors), `reddit_probe`, `voice_edit_items`; external results in `data/episodes/external/results/`.
- Human ratings (3.2 MB): voice-change keys and ratings, Argon keys/answers/scores/audit, Jev todo lists, rerate notes.
- Book: 红楼梦 benchmark and probes (`HONGLOUMENG_probe.md`), bookraw OCR pipeline.

**Models and outputs (`outputs/`, 5.4 GB, 203 run dirs)**
- Backbone Qwen3-0.6B; frozen comparisons used Qwen3-1.7B and Qwen3-4B-Instruct-2507.
- `outputs/trait_predictor/`: LoRA r8 mean/attention, frozen taps (layers 6,12,18,24,28), `frozen_taps_scan_w1.0_s42` (Stage 4b).
- `outputs/runs/`: curriculum, spread, InfoNCE, leak, cleaning, conv, burst, cap, scene, frozen_a/b/c, head_a-f, seed_b, plus the 1h and 3h baselines.
- Others: `landing_audit/` (prediction cache), `hongloumeng/`, `trait_remote/` (box logs), `tonetwin/`, `bookstyle/`.

## 5. Tools

**Code layout:** `src/` (45 modules: `train.py`, `model.py`, `losses.py`, `leak.py`, `episodes.py`, `curriculum.py`, `traits.py`, `named_traits.py`, `trait_directions.py`, `report.py`, `probes.py`, importers for Genshin/bookraw/msynthstel/paratarget/authorcorpora), `scripts/` (49), `configs/` (profiles from dryrun and Colab T4 to 5090x2), `tests/`, `prereg/` (5 documents).

**Scripts by job**
- Train and queue: `run_queue.py`, `launch_queue.sh`, `train.py`, `trait_predictor.py`, `trait_pairs.py`, `trait_scan_finetune.py`, `frozen_llm_eval.py`, `blend_weights.py`.
- Box: `autodl_lib.sh` (start/stop with retry), `remote_setup.sh`, `sync_to_remote.sh`, `sync_from_remote.sh` (rsync, xxh128), `trait_remote.sh`, `frozen_remote.sh`.
- Labels and ratings: `jev_label.py` (`--texts`, `--budget`), `argon_scan.py`, `argon_audit_server.py`, `rate_voice_server.py` plus page, `genshin_tone_rewrite.py`, `book_style_rewrite.py`.
- Evaluation: `external_style_test.py`, `voice_topic_probe.py`, `voice_edit_probe.py`, `pairwise_voice_baseline.py`, `speaker_aggregate.py`, `landing_audit.py`.
- 红楼梦: `hongloumeng_probe.py`, `hongloumeng_trait_test.py`, `hongloumeng_matched_test.py`, `hongloumeng_style_vs_content.py`.
- Analysis: `dim_attribution.py`, `neuron_report.py`, `twin_style_delta.py`, `embed_cache.py`, `embed_samples_umap.py`, `swanlab_notes.py`.
- Guards: `preflight.py` (Stage 0a/0b: any change after a spec commit fails 0b), unit tests.

**Metrics defined:** top-1 and whitened top-1 on held-out speakers; effective rank; line leak and `origin_vs_para` centred gap; Fisher separability; pooled k-line accuracy; voice-over-topic (V and shuffled T); trait r (line and delta); gain over edit size on Argon scan; placebo AUC at a boundary with register/content residualisation.

**Session tooling:** shell and file tools, browser pane, artifact/doc publishers, subagents, linked sessions (reviewer session offline).

**Working rules:** commit trait code and results at once; pre-register tests and keep sealed sets sealed; no box start without your go-ahead; mind Argon cost and context; style, not author; content relation small and positive, never negative.
