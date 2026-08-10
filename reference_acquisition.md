# Reference acquisition list

Papers needed for the presentation and report, and where each is used.

**I cannot download PDFs.** Web fetching returns text, not binary files, and most of these are behind
publisher access. This is the list for you to collect. Drop each into `ref/` under the proposed
filename and I will read it before any slide cites it.

**Identifier status:** ✅ verified from a PDF already in the project · ⚠️ from general knowledge,
**confirm the DOI/arXiv ID before it goes in the bibliography**. I can verify the ⚠️ entries by web
search in a batch if you want — say the word.

**Naming convention.** `author_year_topic.pdf`, lowercase. Existing files mostly follow this;
`s42256-022-00568-3.pdf` does not and would be clearer as `van_de_ven_2022_three_types.pdf`. I have
not renamed it — `ref/` is under a deny rule and I would rather you did it or told me to.

---

## Already in `ref/`

| File | Reference | Used on |
|---|---|---|
| `song_bogacz_24.pdf` | Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024), *Nat Neurosci* 27:348–358 ✅ | Slides 8, 15, 18 — the interference claim, depth, "less erratic updates" |
| `s42256-022-00568-3.pdf` | van de Ven, Tuytelaars & Tolias (2022), *Nat Mach Intell* 4:1185 ✅ | Slide 4 — the scenario taxonomy |
| `dong_wu_rev_song_bogacz.pdf` | Dong, Peng & Wu (2025), *Intelligent Computing* 4:0244 ✅ | Slides 6, 8 — strong vs weak clamp, the plausibility counterweight |
| `kirkpatrick_17.pdf` | Kirkpatrick et al. (2017), *PNAS* 114:3521 ✅ | Slide 3 — dendritic-spine persistence, consolidation |
| `bogacz_2017_core_pcn.pdf` | Bogacz (2017), tutorial on the free-energy framework, *J Math Psychol* ⚠️ | Slide 7 — the PC equations we actually implement |
| `van_rossum_2020.pdf` | **Li & van Rossum (2020), *eLife* 9:e50804** ✅ DOI 10.7554/eLife.50804 | Slide 18 — inefficiency metric, synaptic caching, metabolic cost of plasticity |
| `cl_course/` (8 PDFs) | Continual learning lecture course | Background; check `02_forgetting.pdf` and `04_evaluation.pdf` for standard metrics |

---

## Tier 1 — needed for high-priority slides

| Proposed filename | Reference | Identifier | Used on |
|---|---|---|---|
| `mccloskey_1989_catastrophic.pdf` | McCloskey & Cohen (1989), *Psych. Learning & Motivation* 24:109–165 | ⚠️ book chapter | Slide 2 — the original demonstration |
| `rao_ballard_1999_predictive_coding.pdf` | Rao & Ballard (1999), *Nat Neurosci* 2:79–87 | ⚠️ doi:10.1038/4580 | **Slide 6** — predictive coding as a theory of cortex. Load-bearing for the degree framing |
| `bastos_2012_canonical_microcircuits.pdf` | Bastos et al. (2012), *Neuron* 76:695–711 | ⚠️ doi:10.1016/j.neuron.2012.10.038 | **Slide 6** — PC mapped onto cortical microcircuitry |
| `mcclelland_1995_cls.pdf` | McClelland, McNaughton & O'Reilly (1995), *Psych Review* 102:419–457 | ⚠️ | **Slide 3** — complementary learning systems. The core citation for why brains don't forget |
| `scellier_bengio_2017_eqprop.pdf` | Scellier & Bengio (2017), *Front Comput Neurosci* 11:24 | ⚠️ doi:10.3389/fncom.2017.00024 | Slides 6, 7 — the EqProp formulation we implement |
| `whittington_bogacz_2017_approximation.pdf` | Whittington & Bogacz (2017), *Neural Computation* 29:1229–1262 | ⚠️ doi:10.1162/NECO_a_00949 | Slide 7 — PC *approximates* backprop; the regime we depart from |
| `millidge_2022_infinitesimal_limit.pdf` | Millidge, Song, Salvatori, Lukasiewicz & Bogacz (2022) | ✅ arXiv:2206.02629 | Slide 7 — all three coincide in the infinitesimal limit. The honest note |

## Tier 2 — needed for medium-priority slides

| Proposed filename | Reference | Identifier | Used on |
|---|---|---|---|
| `van_de_ven_2020_brain_inspired_replay.pdf` | van de Ven, Siegelmann & Tolias (2020), *Nat Commun* 11:4069 | ⚠️ doi:10.1038/s41467-020-17866-2 | Slide 3 — replay as hippocampal function |
| `benna_fusi_2016_consolidation.pdf` | Benna & Fusi (2016), *Nat Neurosci* 19:1697 | ⚠️ doi:10.1038/nn.4401 | Slide 3 — multiple timescales within a synapse |
| `laborieux_2021_metaplasticity.pdf` | Laborieux, Ernoult, Hirtzlin & Querlioz (2021), *Nat Commun* 12:2549 | ⚠️ | Slide 3, and the gating next-step |
| `lopez_paz_2017_gem.pdf` | Lopez-Paz & Ranzato (2017), *NeurIPS* 30 | ⚠️ arXiv:1706.08840 | **Slide 11** — the standard ACC / BWT / FWT metric definitions. Worth having before we fix the metric grid |
| `pinchetti_2025_benchmarking_pcn.pdf` | Pinchetti et al. (2025), *ICLR* | ⚠️ arXiv:2407.01163 | Slide 19 — PC matches backprop at small scale, loses at large. The scaling authority |
| `zenke_2017_synaptic_intelligence.pdf` | Zenke, Poole & Ganguli (2017), *ICML* | ⚠️ arXiv:1703.04200 | Slide 3; nearest relative of importance-based consolidation |

## Tier 3 — useful, not blocking

| Proposed filename | Reference | Identifier | Used on |
|---|---|---|---|
| `li_2022_ebm_continual_learning.pdf` | Li, Du, van de Ven & Mordatch (2022), *CoLLAs* | ✅ arXiv:2011.12216 | Slide 22 — the acknowledged alternative EBM |
| `song_2020_can_brain_do_backprop.pdf` | Song, Lukasiewicz, Xu & Bogacz (2020), *NeurIPS* 33 | ⚠️ | Slide 7 — the engineered equivalence that is "not general" |
| `laborieux_2021_scaling_eqprop.pdf` | Laborieux et al. (2021), *Front Neurosci* 15:633674 | ⚠️ | Why EqProp does not scale — explains its behaviour if it underperforms |
| `mirzadeh_2022_wide_nets_forget_less.pdf` | Mirzadeh et al. (2022), *ICML* | ⚠️ | Slide 20 / width discussion |
| `ramasesh_2021_anatomy_of_forgetting.pdf` | Ramasesh, Dyer & Raghu (2021), *ICLR* | ⚠️ | Slide 19 — forgetting concentrates in later layers |
| `friston_2010_free_energy.pdf` | Friston (2010), *Nat Rev Neurosci* 11:127 | ⚠️ | Slide 6 — the wider free-energy framing, if wanted |

---

## Notes

**Highest value first.** If you collect only three, make them **Rao & Ballard 1999**,
**McClelland 1995** and **Bastos 2012** — slides 3 and 6 are the two neuroscience slides, they are
high priority, and they currently have no primary sources in the project at all. Everything else has
either a stand-in or a slide that can wait.

**`cl_course/` is worth a look before we finalise the metric grid.** If the course defines standard
evaluation metrics, matching its vocabulary costs nothing and makes the talk legible to anyone who
took it.

**Anything cited must be read first.** Nothing on a slide gets a citation I have not opened —
`knowledge_base.md` records at least one case where this project asserted something a source did not
support.
