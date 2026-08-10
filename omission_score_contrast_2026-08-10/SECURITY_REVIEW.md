# Security Review Report

**Scope:** Session work for frontier-directions / asr-gate contrast (2026-08-10)
- Primary: `ml-reliability-research/asr-gate/omission_score_contrast_2026-08-10/` (`run_contrast.py`, `results.json`, `run.log`, `SUMMARY.md`, `MAPPING.md`, `SELF_OVERLAP.md`)
- Secondary executable: `frontier-directions-research/.omc/scientist/score_directions.py`
- Breadcrumbs skimmed: paper-readiness `report.md`, ralplan plan, contrast markdowns
- Inherited dependency surface: vendored `experiments/007-omission-crc/vendor/jieba` (imported via `sys.path.insert`)

**Risk Level:** LOW (local offline research; no remote attack surface). Inherited vendor cache/deserialization issues rise to **MEDIUM** on shared multi-user hosts.

**Date:** 2026-08-10  
**Reviewer:** security-reviewer (OWASP Top 10 + secrets + deps)

## Summary
- Critical Issues: **0**
- High Issues: **0**
- Medium Issues: **2**
- Low Issues: **3**
- Secrets leaked in artifacts/docs: **none found**
- New dependency CVEs (scoped): **none found** (`pip-audit` on `numpy` → “No known vulnerabilities found”; no new `requirements` introduced by contrast)

**Posture:** Acceptable for single-user, offline CPU research. `run_contrast.py` itself has no pickle/eval/subprocess/network/SSRF/command-injection surface. Main residual risk is **trust of the jieba vendor + `/tmp/jieba.cache` marshal load**, plus **absolute home-path leakage** if artifacts are published.

---

## Critical Issues (Fix Immediately)

_None._

## High Issues

_None._

---

## Medium Issues

### 1. Jieba prefix-dict cache uses `marshal.load` from world-writable `/tmp`
**Severity:** MEDIUM  
**Category:** A08 Integrity Failures / A03 Injection (unsafe deserialization)  
**Location:** `experiments/007-omission-crc/vendor/jieba/__init__.py:124-136` (triggered at import/use from `run_contrast.py:31-32`; observed in `run.log:4` → `Loading model from cache /tmp/jieba.cache`)  
**Exploitability:** Local, unauthenticated (another local user / shared `/tmp` race before cache is created or via symlink tricks on multi-user hosts)  
**Blast Radius:** Arbitrary code execution as the researcher user when jieba initializes  
**Issue:** Session script imports vendored jieba POS. Jieba writes/loads a prefix dictionary cache under `tempfile.gettempdir()` (`/tmp/jieba.cache`) via `marshal.load`. Marshal is not a safe untrusted-deserialization format. Current cache mode was `0600` for the owner (good once created), but creation/replacement races on shared hosts remain a known jieba class of issue.  
**Remediation:**
```python
# BAD (inherited jieba default): cache in shared /tmp + marshal.load
cache_file = os.path.join(tempfile.gettempdir(), "jieba.cache")
with open(cache_file, "rb") as cf:
    self.FREQ, self.total = marshal.load(cf)

# GOOD: pin cache to a private directory before importing jieba
import os
from pathlib import Path
cache_dir = Path.home() / ".cache" / "asr-gate-jieba"
cache_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
os.environ.setdefault("TMPDIR", str(cache_dir))  # or set jieba.tmp_dir if available
# Prefer: hash-pin / vendor-lock jieba; avoid re-reading mutable shared caches.
# Strongest: replace jieba POS with a pure-data tagger that does not marshal/pickle.
```

### 2. Cross-repo `sys.path.insert` executes unpinned vendored code
**Severity:** MEDIUM  
**Category:** A08 Integrity Failures / A06 Vulnerable Components  
**Location:** `run_contrast.py:27-32`  
**Exploitability:** Local (requires write access to vendor tree or ability to replace path contents/symlinks)  
**Blast Radius:** Full code execution at import time under researcher privileges; supply-chain compromise of POS tagging + any side effects in vendor `__init__`  
**Issue:** Absolute path to `frontier-directions-research/.../vendor` is prepended to `sys.path`, then `import jieba.posseg` runs that tree. No integrity hash, signature, or content pin. Vendor also contains dormant dangerous helpers (e.g. `os.system("pip install ...")` in `_compat.enable_paddle`) and pickle loaders for Jython paths (not used on CPython posseg hot path, but present).  
**Remediation:**
```python
# BAD
JIEBA_VENDOR = Path("/home/.../vendor")
sys.path.insert(0, str(JIEBA_VENDOR))
import jieba.posseg as pseg

# GOOD — integrity-gated vendor load
import hashlib
EXPECTED = "sha256:<pin-of-vendor-tree-or-wheel>"
def _assert_vendor_hash(root: Path, expected: str) -> None:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    digest = "sha256:" + h.hexdigest()
    if digest != expected:
        raise RuntimeError(f"jieba vendor integrity mismatch: {digest}")

_assert_vendor_hash(JIEBA_VENDOR, EXPECTED)
# Better: install a pinned jieba wheel from PyPI/lockfile instead of sys.path hacks.
```

---

## Low Issues

### 3. Absolute home-directory paths embedded in results / logs
**Severity:** LOW  
**Category:** A01 Broken Access Control (info disclosure) / A09 Logging Failures  
**Location:** `results.json:5,27` (and per-cell `path`); `SUMMARY.md` landscape line; `run.log:1,4,64`; `run_contrast.py:256,302,381`  
**Exploitability:** Anyone with read access to published artifacts  
**Blast Radius:** Leaks username (`zeyufu`), machine directory layout, and research corpus paths — useful for targeted local attacks or doxxing in public releases  
**Issue:** Outputs store absolute paths such as `/home/zeyufu/Desktop/...`. No API keys/transcripts found in `results.json` (good: no `ref_text`/`hyp_text`/`token_logps` persisted).  
**Remediation:**
```python
# BAD
out["landscape"] = str(LANDSCAPE)
cell_meta["path"] = d["_path"]

# GOOD
out["landscape"] = str(LANDSCAPE.relative_to(HERE.parent))  # or basename only
cell_meta["path"] = Path(d["_path"]).name
# Also redact paths from run.log when archiving for public release.
```

### 4. Landscape JSONL trusted without schema / size bounds
**Severity:** LOW  
**Category:** A04 Insecure Design / A03 Injection (data integrity)  
**Location:** `run_contrast.py:129-140`  
**Exploitability:** Local (attacker must replace frozen landscape files)  
**Blast Radius:** Process crash / memory exhaustion DoS; corrupted scientific conclusions; not remote RCE via `json.loads` alone  
**Issue:** `load_cell` builds `LANDSCAPE / f"decode_{corpus}_{backbone}_test.jsonl"` from hardcoded corpora/backbones (no CLI path traversal), then `json.loads` each line with no max line size, type checks, or allowlist on nested structures. Hostile JSONL could allocate huge `token_logps` lists.  
**Remediation:**
```python
# BAD
for line in fh:
    r = json.loads(line)

# GOOD
MAX_LINE = 2_000_000
for line in fh:
    if len(line) > MAX_LINE:
        raise ValueError("jsonl line too large")
    r = json.loads(line)
    if not isinstance(r, dict):
        continue
    tl = (r.get("nbest") or [{}])[0].get("token_logps") or []
    if not isinstance(tl, list) or len(tl) > 50_000:
        continue
```

### 5. Vendored jieba contains `os.system` install helper (not on hot path)
**Severity:** LOW  
**Category:** A03 Command Injection / A05 Security Misconfiguration  
**Location:** `vendor/jieba/_compat.py:27-32` (`enable_paddle`)  
**Exploitability:** Local, only if `enable_paddle()` is called  
**Blast Radius:** Arbitrary package install via shell to `pip` (network + integrity risk)  
**Issue:** `run_contrast.py` does **not** call `enable_paddle()`, so current session execution path is clean. Dead/latent dangerous API remains in the imported package tree.  
**Remediation:**
```python
# BAD
os.system("pip install paddlepaddle-tiny")

# GOOD — delete enable_paddle from vendor, or:
raise RuntimeError("paddle path disabled in research vendor")
# subprocess with argv list + pinned version + no shell, if ever required
```

---

## OWASP Top 10 Evaluation

| ID | Category | Result for session scope |
|----|----------|--------------------------|
| A01 | Broken Access Control | N/A (no auth routes). Path disclosure in artifacts → Low #3 |
| A02 | Cryptographic Failures | No secrets/crypto; N/A |
| A03 | Injection | No SQL/cmd/XSS in `run_contrast.py`. Inherited marshal/os.system in jieba → Med #1, Low #5 |
| A04 | Insecure Design | Offline batch OK; unbounded JSONL trust → Low #4 |
| A05 | Security Misconfiguration | Hardcoded absolute paths; debug-ish vendor logger noise in `run.log` |
| A06 | Vulnerable Components | No new deps; numpy audit clean in scoped check; unpinned vendor → Med #2 |
| A07 | Auth Failures | N/A (no AuthN) |
| A08 | Integrity Failures | Vendor path + `/tmp` marshal cache → Med #1/#2 |
| A09 | Logging Failures | Logs/results avoid transcripts/secrets; leak absolute paths → Low #3 |
| A10 | SSRF | **None** — no URL fetch / HTTP client |

### Targeted checks (requested)
| Check | Verdict |
|-------|---------|
| Path traversal | **Not present** via argv (no CLI). Paths from constants + allowlisted corpus/backbone tuples |
| pickle / eval | **Not in `run_contrast.py`**. CPython `posseg` uses `.py` probability tables; jieba still `marshal.load`s `/tmp` cache |
| Command injection | **Not in session script**. Latent `os.system` only in unused jieba paddle helper |
| Unsafe deserialization | **Inherited** jieba marshal cache (Med #1) |
| Secret leakage in logs/results | **None found** (no keys/tokens); absolute paths only |
| SSRF | **None** |

### `score_directions.py` (frontier session script)
- Stdlib only (`os`/`re`/`json`); reads `directions/*.md`, writes maturity JSON.
- No network, secrets, subprocess, or deserialization.
- Finding: hardcoded absolute `root` path only (portability / mild path disclosure if output published) — informational / covered by Low #3 pattern.

---

## Secrets Scan
- Patterns scanned: `api_key`, `password`, `secret`, `Bearer`, `sk-`, `hf_`, `AKIA`, private key PEM, etc.
- Targets: contrast dir (code + results + logs + md), paper-readiness report, ralplan breadcrumbs.
- **Result:** no hardcoded credentials or tokens found.

## Dependency Audit
- Contrast introduces **no** new declared dependencies (uses existing `numpy` + vendored jieba).
- `asr-gate/pyproject.toml` deps unchanged by this session (`relmetrics`, `numpy`, `pandas`, `scipy`).
- Scoped `pip-audit` on `numpy`: **No known vulnerabilities found**.
- Installed numpy observed: `2.2.6` (environment-dependent).
- Recommendation: if publishing/reproducing, add a lockfile / pin jieba instead of `sys.path` vendor.

## AuthN / AuthZ
- No authentication, authorization, sessions, JWT, or network API endpoints in scope. **N/A.**

---

## Security Checklist
- [x] No hardcoded secrets
- [x] Injection prevention verified for session-owned code (no SQL/cmd/eval/pickle)
- [x] SSRF not applicable / absent
- [x] Authentication/authorization N/A
- [x] Dependencies audited (scoped; no new CRITICAL/HIGH)
- [ ] Inputs fully validated (landscape JSONL lacks bounds/schema) — Low #4 open
- [ ] Vendor integrity pinned — Med #2 open
- [ ] Jieba cache isolated from shared `/tmp` — Med #1 open
- [ ] Absolute paths redacted before public release — Low #3 open

---

## Priority Remediations
1. **Before shared-host runs:** set private `TMPDIR` / jieba tmp dir (Med #1).
2. **Before trusting vendor long-term:** hash-pin or wheel-pin jieba; avoid bare `sys.path.insert` (Med #2).
3. **Before publishing artifacts:** strip `/home/...` paths from `results.json` / logs (Low #3).

## Top Findings (quick)
1. **MEDIUM** — jieba `/tmp/jieba.cache` + `marshal.load` (inherited)
2. **MEDIUM** — unpinned cross-repo vendor on `sys.path`
3. **LOW** — absolute home paths in results/logs
4. **LOW** — unbounded trusted JSONL parse
5. **LOW** — latent `os.system` in unused jieba paddle helper

**Overall posture:** **LOW risk** for the intended offline, single-user research use case; no CRITICAL/HIGH in session-authored code. Harden vendor/cache integrity before multi-user or public distribution.
