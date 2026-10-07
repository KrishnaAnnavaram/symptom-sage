# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **interview** | One simulated doctor-patient conversation | consultation, session |
| **transcript** | The text of an interview with `D:` and `P:` turns | dialogue, script |
| **turn** | One block of speech by the doctor or the patient | utterance, line |
| **case** | One interview with its case id (for example `RES0001`) | sample, row |
| **case record** | The validated `CaseRecord` of one case | extraction, row |
| **assessment** | The condition that the doctor names at the end of the interview | diagnosis (in prose), disease |
| **plan** | The next steps that the doctor names. Never shown to the user | prescription, treatment |
| **symptom** | A canonical name from the lexicon, for example `shortness of breath` | complaint, sign |
| **document** | The text of a case in the index: category, patient, symptoms, assessment | chunk, page |
| **index** | The vector store of case documents | database, vector DB |
| **hit** | One retrieved case with its scores | match, result |
| **red flag** | A rule that sends the user to emergency or urgent care | alarm, alert |
| **triage level** | `emergency`, `urgent` or `routine` | severity, priority |
| **answer kind** | `emergency`, `urgent`, `medication_refused`, `out_of_scope`, `no_match` or `answer` | response type |
| **disclaimer** | The fixed text at the end of each answer | warning text |
| **red-team set** | The fixed list of risky prompts with the expected answer kind | attack set |
| **WER** | Word error rate after normalization | accuracy |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **extract** | Make a case record from a transcript |
| **index** | Add or replace case documents in the index |
| **retrieve** | Find the most similar cases for a question |
| **triage** | Check a question against the red-flag rules |
| **cite** | Name a retrieved case as `[CASE_ID]` in an answer |
| **strip** | Remove sentences that name a medicine or a dose |
| **transcribe** | Turn audio into text |
