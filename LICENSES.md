# Source and License Policy

No biblical corpus, morphology, lexicon, commentary, organizational document, model, or derived dataset is approved merely because it is accessible online. Approval requires the exact artifact, revision, SHA-256 digest, license text, attribution, intended uses, and a named legal/repository decision in `configs/data/source_registry.json`.

Candidate findings checked on 2026-08-11:

- The official Apertus 8B Instruct model card reports Apache-2.0.
- Open Scriptures reports the WLC text as public domain and MorphHB lemma/morphology data as CC BY 4.0.
- The official Faithlife/SBLGNT repository reports SBLGNT as CC BY 4.0.
- MorphGNT licensing must be verified from the exact revision and bundled license rather than inferred from the SBLGNT text license.
- The eBible KJV 1769 source describes the text as public domain outside the United Kingdom and notes continuing UK printing restrictions. Deployment jurisdiction therefore remains part of approval.

These are research notes, not legal advice or approval. All candidates remain `pending_legal_review` until exact artifacts and downstream training/distribution obligations are accepted.

Proprietary translations, lexicons, commentaries, sermons, study notes, and manuscript databases are excluded by default. Fair-use assumptions are not a substitute for permission in a distributable training corpus.

