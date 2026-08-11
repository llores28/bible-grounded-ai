# Risk Register

Version: 1.0  
Review cadence: before each training run, evaluation release, and policy change

| ID | Risk | Primary controls | Acceptance evidence | Current status |
| --- | --- | --- | --- | --- |
| R-001 | Fabricated verse, quotation, source-language form, or citation | Immutable approved corpus; exact verifier; retrieval IDs; correction gate | 100% quote exactness, >=99% citation accuracy, zero fabricated sources | Open: corpus pending |
| R-002 | Denominational preference masquerades as biblical evidence | Evidence classes; blinded review; organization weight 0; leakage scan | Zero organizational-source leakage; affiliation audit | Controlled in schema; review pending |
| R-003 | Historicist inference presented as explicit text | Prophetic registry; unit preservation; school labels; counter-readings | 100% arithmetic; class-label evals; no invented anchors | Controlled in registry |
| R-004 | Numerology or hidden-code overclaim | Pre-registration; controls; robustness tests; falsification requirement | Zero accepted unsupported hidden-code claims | Test corpus pending |
| R-005 | Abuse justified through honor, submission, forgiveness, or church authority | Hard-floor rule 5; abuse safety rule; lawful reporting and boundary tests | 100% pass on abuse/coercion sealed cases | Tests seeded; expert review pending |
| R-006 | Lethal or violent action authorized | Hard-floor rule 6; imminent-danger escalation; no lethal authorization | Zero accepted murder planning; force distinctions pass | Tests seeded; legal review pending |
| R-007 | Deception rewarded as kindness or confidentiality | Hard-floor rule 9; truthful refusal/silence paths; citation checks | Zero accepted lying/fabrication; confidentiality tests pass | Tests seeded |
| R-008 | Theft of data, credentials, labor, or intellectual property | Hard-floor rule 8; authorization-aware tests | Zero accepted credential or property theft | Tests seeded |
| R-009 | Sexual coercion, adultery, trafficking, or concealment enabled | Hard-floor rule 7; consent and exploitation checks | Zero accepted facilitation cases | Tests seeded |
| R-010 | Hidden motives asserted as fact | Rule 10; observable-conduct wording checks | Motive-uncertainty adversarial tests pass | Test corpus pending |
| R-011 | Religious paranoia, scrupulosity, or direct-message claims reinforced | No-revelation policy; uncertainty; mental-health/pastoral referral | Zero divine-certainty affirmations; referral coverage | Tests seeded |
| R-012 | Medical refusal or self-harm advice | Pastoral safety checks; emergency and professional referral | Zero unsafe medical/self-harm outputs | Tests seeded |
| R-013 | Sensitive pastoral or victim data leaks into training | Consent, minimization, redaction, provenance, access control | Privacy audit and zero unapproved PII | Process pending |
| R-014 | Train/eval contamination inflates metrics | Family-based splitting; hashes; sealed custody | Contamination report and signed manifest | Infrastructure pending |
| R-015 | Reviewer capture, affiliation bias, or sponsor pressure | Independent review, blinded preference order, conflicts, adjudication | Agreement statistics and conflict disclosures | Review board pending |
| R-016 | CUDA/QLoRA incompatibility or out-of-memory failure | Hardware preflight; smoke test; pinned run manifest; gradient checkpointing | Reproducible smoke run on target GPU | Not run |
| R-017 | Base-model or corpus license violation | Source registry; exact revisions/hashes; legal sign-off; attribution bundle | Every used source approved | Release blocker |
| R-018 | Safety checker produces false assurance | Defense-in-depth statement; semantic sealed eval; human review | Published false-positive/negative analysis | Evaluation pending |
| R-019 | Model output treated as divine or pastoral authority | Repeated scope disclosure; no first-person divine speech; referrals | Authority-claim adversarial suite passes | Tests seeded |
| R-020 | Public claims exceed evidence | Model/data cards; failed-case publication; release evaluator | Signed release report with all gates passed | Release blocker |

An open release-blocking risk cannot be waived by a model score. Any waiver requires a public rationale, two independent approvals, a sunset date, and an explicit statement of residual harm; hard-floor, fabricated-source, licensing, and sealed-test integrity gates are non-waivable.

