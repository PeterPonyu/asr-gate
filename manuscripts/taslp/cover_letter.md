# Cover letter — IEEE Transactions on Audio, Speech, and Language Processing (TASLP)

Date: TODO-USER

Dear Editor-in-Chief and Associate Editors,

We submit our manuscript, **"Certified Transcription Triage: a Distribution-Free Accept/Defer
Gate with an Audited Confidence Signal for Mandarin ASR,"** for consideration as a regular paper
in the IEEE Transactions on Audio, Speech, and Language Processing.

Automatic speech recognition is increasingly deployed with a human in the loop, auto-accepting
confident transcriptions and deferring the rest. The paper asks two questions this raises and
answers both empirically. First, can one certify, distribution-free, that the character error rate
(CER) of the auto-accepted set stays below a target? We give a Learn-then-Test selective-risk
certificate on the accepted-set macro-CER that, on Aishell-1 test with a Paraformer backbone,
holds at the target with 0/20 violations across 20 calibration reseeds while auto-accepting
85.7–96.2% of utterances at an accepted-set macro-CER of 1.00–1.53% (full-set 1.98%), and that
degrades gracefully under additive noise. Second, do the field-standard confidence scores that
would drive such a gate beat honest random deferral? An excess area-under-the-risk–coverage-curve
audit against an analytic random-deferral baseline, under a matched-abstention permutation null
with Holm control, finds positive skill in all twelve cells of a roster-derived family
(0.012–0.051, bootstrap CIs excluding zero) across architectures, noise levels, and corpora.

The work fits TASLP's scope directly: it is a speech-and-language-processing reliability method
evaluated on standard Mandarin ASR benchmarks, addressing the confidence-estimation and
selective-prediction problems that arise when ASR is deployed with human review. We position it
against set-valued conformal ASR and non-monotone risk control, and all reported numbers are
reproducible from frozen result files.

In keeping with the paper's emphasis on honest reliability claims, the abstract itself discloses
where the certificate fails and why: on a Whisper zero-shot Mandarin backbone (macro-CER 28.1%)
the certificate is vacuous at every target because no low-CER accepted region exists, and the
0/20-violation result is stated as coming from correlated reseeds of one fixed test set rather
than 20 independent trials — we regard reporting these boundaries as part of the contribution, not
a caveat to be minimized.

We confirm that this manuscript is original work, is not under consideration or review elsewhere,
and has not been submitted in whole or part to any other venue. The single author has approved the
submission. Data and code availability are as stated in the manuscript's availability statement
(all results reproducible from archived experimental artifacts and the released accept/defer-gate code).

Thank you for your consideration.

Sincerely,

Zeyu Fu
TODO-USER: affiliation line (department, institution, city, country)
e-mail: fuzeyu99@126.com

---
**Suggested reviewers** (TODO-USER: supply three; leave blank if you prefer the editors choose):
1. TODO-USER — expertise: ASR confidence estimation / selective prediction for speech.
2. TODO-USER — expertise: conformal prediction / distribution-free risk control (Learn-then-Test).
3. TODO-USER — expertise: Mandarin ASR / robustness under acoustic noise.
Pick reviewers without a recent co-authorship or shared-institution conflict with the author.

**Funding statement:** TODO-USER (state grant/support, or "The author received no specific funding
for this work.").
