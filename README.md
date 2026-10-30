# Model based picture enlargement claim audit

An offline static auditor for documentation promises about learned upscaling.
No runtime dependencies; Python 3.9 or later is required. Source is parsed,
never executed. Install with pip from a distribution.

```sh
super-resolution-claim-audit ./project --json
python3 -m super_resolution_claim_audit ./project --verbose
super-resolution-claim-audit ./project --ignore 'vendor/*' --fail-on MEDIUM
```

Root README.md and Markdown files with a docs path component are checked.
Fenced examples are excluded; inline code counts. Matching uses exact normalized
phrases, paragraph joining, contractions and hyphen normalization. Negative
spans are removed before positive matching.

Evidence includes imported OpenCV DnnSuperResImpl_create, RealESRGANer,
RRDBNet, SwinIR, EDSR, RCAN, SRResNet, LapSRN and FSRCNN constructors,
and transformers pipeline construction with the corresponding literal task.
Construction alone counts. Imports alone, interpolation, cropping, filters,
and unrelated vision models do not count. Local definitions shadow imports.
Only a single module's import aliases are resolved; runtime behavior, model
weights, output quality and dynamically constructed tasks cannot be verified.

Production evidence excludes fixtures, docs, examples and tests; test evidence
can reduce a missing-evidence finding to MEDIUM. Symlinks are excluded.
Invalid syntax supplies no evidence. No claim yields PASS.

Rules:
```
documented-super-resolution-claim-false: HIGH when production evidence is absent;
MEDIUM when evidence exists only in tests.
documented-no-super-resolution-claim-false: HIGH when production evidence exists.
```

Exit 0 means the selected threshold is satisfied, 2 means a finding meets it,
and 1 means invalid arguments or an unreadable root. Default threshold is HIGH;
--fail-on MEDIUM includes both severities. JSON contains verdict, findings
(rule, severity, message), and files_scanned. Human output lists findings;
--verbose adds the scanned count. --version prints 0.1.0.
