# Data

Git does not track the files in this folder, except this README. Do not commit transcripts,
audio, extracted case files or indexes.

## 1. Synthetic interviews (default, no download)

```bash
symptom-sage synth --out data/synthetic --per-condition 8 --seed 21
```

| Path | Contents |
|---|---|
| `data/synthetic/transcripts/<CASE_ID>.txt` | Invented `D:` / `P:` interviews, 12 conditions, 96 cases |
| `data/synthetic/truth.jsonl` | Ground truth per case: `age`, `gender`, `symptoms`, `condition`, `plan` |

## 2. Public simulated interviews (optional)

| Item | Value |
|---|---|
| Name | A dataset of simulated patient-physician medical interviews with a focus on respiratory cases |
| Authors | Fareez F. et al., Scientific Data 9, 313 (2022) |
| Host | figshare (follow the link in the paper) |
| Licence | CC BY 4.0. Cite the paper |
| Contents | 272 acted OSCE-style interviews: MP3 audio and cleaned transcripts. No real patient data |
| Categories | `RES` respiratory, `MSK` musculoskeletal, `GAS` gastrointestinal, `CAR` cardiac, `DER` dermatology, `GEN` general |

Expected layout after the download:

| Path | Contents |
|---|---|
| `data/transcripts/<CASE_ID>.txt` | One transcript per case. Lines start with `D:` or `P:`. A turn can continue on the next line. UTF-8 or UTF-16 |
| `data/audio/<CASE_ID>.mp3` | Optional. Only for `symptom-sage asr-eval --audio-dir` |

About 78% of the cases are respiratory. A symptom checker built on this data does not cover most
conditions that people ask about.

## 3. Generated files

| Path | Command |
|---|---|
| `data/cases.jsonl` | `symptom-sage extract` (one validated `CaseRecord` per line) |
| `data/cases.failures.jsonl` | `symptom-sage extract --extractor llm` (cases that failed twice) |
| `index/vectors.npy`, `index/docs.json` | `symptom-sage index` |
