# FINDING `s3c` — encoder slot padding is numerically broken on the opaque-attention path. Main is fine. The fusion was masking it.

Pod `tpa-spyre-dev-2`. Branch `s3c-b3-fusion` (fork `s3-dev2-b3-attn-inline-fusion` @ `8191de7` + harness).
Continues `HANDOFF-s3-dev2-FINAL`. All arms and scripts: `b3/` in that worktree.

---

## 1. The result

`HANDOFF-s3-dev2-FINAL` §2 asked one question — does the degenerate fusion-OFF arm reproduce on the
shipped default buckets, or was it an artefact of forcing `compile_sizes=[2048]`? **It reproduces, and
the arm that localises it says slot padding owns it.**

Every arm below is the same 4 texts (352–398 tokens, 4 languages) against the same CPU HuggingFace
golden (`golden_cpu.py`, fp32, CLS + L2, matching the served `PoolerConfig`).

| arm | `INLINE` | `SLOT_PADDING` | buckets | vs golden | own spread | verdict |
|---|---|---|---|---|---|---|
| `ref2-noinline` | 0 | 1 | [2048] | 0.468795 | 0.99986..0.99995 | FAIL |
| `ref3-noinline-default` | 0 | 1 | default | 0.468889 | 0.99986..0.99996 | FAIL |
| `s3c-noinline-solo` | 0 | 1 | default, **1 seq/step** | 0.468757 | 0.99986..0.99996 | FAIL |
| `s3c-noinline-grouped` | 0 | 1 | default, 4 seq | 0.468795 | 0.99986..0.99995 | FAIL |
| **`s3c-noinline-noslot`** | **0** | **0** | default | **0.999974** | 0.61632..0.79373 | **PASS** |
| `test2-inline` | 1 | 1 | [2048] | 0.999975 | 0.61649..0.79369 | PASS |
| golden (reference) | — | — | — | 1.0 | 0.61556..0.79368 | — |

**`SPYRE_ENCODER_SLOT_PADDING` defaults to `1` on this branch**, so the broken configuration is the
patch's own default. It does not exist on `main` or on the PR961 base (`git grep SLOT_PADDING main` is
empty), so **`main` is unaffected** — `s3c-noinline-noslot` is main's packed var-len path and it passes.

**The fusion is not a fix.** It is correct, but only because inlining bypasses whatever the opaque path
does wrong. Shipping the fusion on top of slot padding would bury a live correctness bug under a
perf win.

### Two things the numbers say that the single worst-cosine number does not

- **Own within-run spread is the better degeneracy detector.** The golden's four vectors span
  0.6156..0.7937. Both passing arms reproduce that to three decimals. The failing arms sit at
  0.9999 — four different languages collapsing onto one vector. A `FAIL` verdict against a golden
  tells you something is wrong; the spread tells you *how*, and it needs no golden at all, so it is
  the cheap gate to put in front of any future encoder arm.
- **The defect is batch-size-independent.** `s3c-noinline-solo` embeds each text in its own call:
  one request, `1 slots x 512 rows`, no second slot in the step. Still degenerate, to four decimals.
  Any explanation that needs two slots in one step is therefore wrong.

## 2. What is ruled out, with the arm that rules it out

I proposed two mechanisms and **both are dead**; recording them so nobody re-runs them.

1. **Cross-slot contamination** (every query attending every slot's keys, which would explain the
   collapse because CLS queries are near-identical across texts). Refuted by `s3c-noinline-solo`:
   the bug is present at `group == 1`.
2. **The additive mask being dropped** (torch-spyre#4526's failure mode, and group-independent, so it
   survived (1)). Refuted by `b3/slot_kernel_probe.py`.

`slot_kernel_probe.py` calls `_encoder_slot_compiled` directly against a CPU reference of the same
function, at group 1/2/4, extent 512, 12 heads, head_size 64, with per-slot `kv_len` mirroring the
probe texts, and with `FallbackWarning` promoted to an error so a silent CPU fallback cannot fake a
verdict. Three references separate the cases: `isolated` (correct), `contaminated` (batch dim
collapsed), `nomask` (mask dropped).

```
group=1  vs isolated rel=0.0074   vs nomask rel=0.5216   -> MATCHES ISOLATED (correct)
group=2  vs isolated rel=0.0120   vs contaminated rel=0.9013  vs nomask rel=0.9734  -> correct
group=4  vs isolated rel=0.0058   vs contaminated rel=1.3662  vs nomask rel=0.5885  -> correct
```

**The compiled slot kernel's math is right and its mask is applied.** So the fault is not in that
kernel given correct operands.

⚠️ **This probe is necessary, not sufficient, and the reason is written in the code it tests.**
`_warm_kernels`' docstring: *"a Spyre tensor's device layout is part of the cache key, and a stand-in
does not reproduce it … a fused-QKV `query` is a strided view, and `output` arrives as
`torch.empty(rows, H*D).view(-1, H, D)` whose layout differs from a same-shaped `torch.zeros`."* The
probe converts fresh host tensors, i.e. exactly the stand-in that docstring warns about. **A probe
that reproduces the real operands' layouts is the next step**, and it is cheap (no engine, seconds
per arm) compared with a 5–7 minute end-to-end arm.

## 3. Correction to my own earlier localisation — do not cite it

I claimed from source that the two arms "differ in exactly one thing", `_slot_fn` being
`_encoder_slot_compiled` vs the raw `_encoder_slot_kernel`. **That is too narrow.** `_inline_attn`
also decides whether attention is registered as an opaque custom op at all, so with the fusion off
the *whole transformer block* is a different compilation unit. The defect need not be in attention.
Combined with §2, the live suspects are the layout/aliasing of the real buffers across the opaque-op
boundary, not the attention math.

## 4. Where to look next, cheapest first

1. **Layout-faithful kernel probe** (§2's ⚠️): build `output` as `torch.empty(rows, H*D).view(-1, H, D)`
   and `query` as a strided fused-QKV view, then re-run `slot_kernel_probe.py`. Seconds per arm.
   `tests/attention/test_spyre_encoder_attn.py` already has `_vllm_style_output` for the output half.
2. **The `layout_ok` probe is load-bearing and its failure is silent.** With the fusion off,
   `fused_store_ok` gates on `output.storage_offset() == 0 and output.is_contiguous()`; when it fails,
   `store_mode` becomes `"none"` *and* `self._slot_ok` goes false, which silently drops the whole step
   to the var-len path. Same hazard class as the handoff's §4 note. **Log a warning when it flips** —
   an arm that thinks it is testing slot padding may not be.
3. **The coverage gap that let this land.** `test_grouped_attention_matches_the_per_sequence_reference`
   builds a *packed* query (`total_tokens = sum(query_lens)`), so `query.shape[0] != slots.total_rows`
   and the slot fast path is never taken. **`_encoder_slot_kernel` has no end-to-end numerical test
   against a reference anywhere in the suite.** `slot_kernel_probe.py` is the missing test's skeleton;
   the `own-spread` check in `b3/table.py` is the cheap end-to-end form.

## 5. FIXED, then MEASURED: the fused path is ~11% faster per forward

### The per-call recompile, and what it actually was

`perf_steps.py` embeds the same 4 texts repeatedly. Before the fix, three
consecutive settle calls with `SPYRE_ATTN_INLINE=1` cost **286 / 311 / 321 s** --
flat, so the fused body graph was recompiling every call, not once per shape.
Steady-state hostprobe put it beyond doubt: `_model_forward` 189,173 ms/call while
everything outside it was microseconds.

`TORCH_LOGS=recompiles` named the guards. Four causes, and **three of them are
diagnostic instrumentation that was only ever safe while attention was opaque** --
the same root cause as the handoff's five blockers, in new places:

1. `if attn_metadata.encoder_seq_plans is None` was traced, and dynamo guarded on
   the None-ness: None on the first trace, set by layer 0 after. Fixed by hoisting
   the plan build out of the graph (`prime_slot_plan`).
2. `_call_kernel` read dynamo's own `counters["stats"]`, which changes constantly,
   so the graph guarded on that dict. 276 guard failures.
3. `note_unattributed_compiles` called `logger.warning` inside the traced region --
   a hard crash (`logging.Logger method not supported`) as soon as it had anything
   to report.
4. `forward_context.slot_mapping[...]` size mismatch, the dominant trigger. Fixed
   by `SPYRE_ENCODER_FASTPATH=1`, an existing flag whose whole job is to reuse one
   slot-mapping buffer per shape; it is **off by default**.

Also fixed, latent rather than firing: the slot mask was keyed on `kv_lens`, which
fixes its contents but never its shape, and that key was read inside the traced
region -- so every distinct set of request lengths was a distinct tensor. Keyed on
`(extent, num_slots)` now, one buffer per shape.

⚠️ **`SPYRE_COMPILE_GRANULARITY=model` is NOT the fix and should not be tried.**
The `layer_name` guard re-specialises one block code object per layer, but that is
**bounded at 12 and paid once** -- it is not per-call and it is not a problem. The
107 cache entries seen before the fix were the *product* of 12 layers and ~9 calls,
because the per-call guards kept failing. A whole-model graph addresses neither
per-call guard and costs far more to compile.

### After the fix

```
settle (s): ['382.8697', '0.0585', '0.0566']     <- converges, was flat at ~300
B3_STEPS inline=1 bucket=2048 n=6  median=56.62  mean=56.65  min=56.37  p90=57.22
```

One 383 s compile on the first call, then 58 ms. Recompiles bounded at 50, one-time.

### The number, matched arms

Same 4 texts, `bucket=2048`, `granularity=block`, `n=6`. Both arms verified against
the CPU golden in the same process that timed them (0.999975 and 0.999974), and the
fused arm is bit-identical to the pre-fix fused arm, so the fixes moved no numerics.

| arm | median ms/call | `_model_forward` ms/call |
|---|---|---|
| fused: `INLINE=1` + slot padding + `FASTPATH=1` | **56.62** | **23.27** |
| baseline: `INLINE=0`, packed var-len (what ships) | 63.17 | 26.15 |

**~10% per call, ~11% per forward, i.e. 2.9 ms/step.** That is in the same ballpark
as the handoff's ~4 ms/step estimate from 43 -> 23 launches, so its launch-cost
argument was roughly right and `FINDING-s3-opus5-L-2`'s "prize is small" was too
pessimistic. It is also nowhere near the 2x that GOAL.md's 200 req/s needs.

⚠️ **Caveats on the number.** `n=6`. The baseline's mean/p90 are void -- a pooler
compile fired inside its timed window (`_pool` 134 ms/call in steady state); its
median and min agree at 62-63 ms so the median stands, but that arm deserves a
clean re-run. And **only ~23 of the 56.62 ms per call is the forward**: the rest is
in-process API overhead both arms pay, so per-call wall understates the difference
and `_model_forward` is the honest term to compare.

### Still open: warmup coverage

The fused path pays ~383 s on the first call for a layout warmup did not build,
because `_warm_kernels` is skipped when inlining (`not (self._inline_attn and
torch.compiler.is_compiling())`) and **nothing replaces its enumeration** of every
reachable `(buffer_rows, slots)` pair. Warmup produced only `4 slots x
{64,128,256,512}` while a real step wanted `2 slots x 512`. This is now a
first-call cost per layout rather than a per-call cost, but a server must still
cover every reachable layout or pay it mid-request.

### ⚠️ Retraction — I got the 185 s attribution wrong; the handoff was right

I claimed the handoff's unexplained 185 s/warmup-request was a **pooler** compile, "not specific to
the fusion", on the strength of this line in its server log:

```
14:24:39 WARNING [spyre_attn.py:218] 1 graph(s) compiled outside warmup and outside an
                 attention kernel (observed at pooler; 1 total there)
```

**That reading is wrong.** The warning comes from `note_unattributed_compiles("model body / pooler")`,
which lumps body and pooler into one label: "observed at pooler" says where the compile was *noticed*,
not what compiled. The cost separates the two — in the `SLOT_PADDING=0` baseline these compiles are
~0.5 s each (pooler-sized), while the fused arm's equivalent is ~300 s (body-sized). So the
handoff's 185 s was the fused body graph, and its "surviving hypothesis … a per-shape recompile of
the fused kernel in the serving path" was **correct**. Do not cite my reattribution.

What survives from that analysis, and is still worth knowing: the client's warmup phase is
**sequential** (`1%| | 1/100` in its tqdm), so every warmup request is a 1-sequence batch — a shape
warmup never compiled — and the 5-hour ETA tqdm printed is `185 s × 99`, which assumes every request
recompiles. Given §5's finding that recompiles *do* repeat, that extrapolation may not have been as
wrong as it looked.

### Why warmup cannot currently cover this

With `SPYRE_ATTN_INLINE=1`, `_warm_kernels` is skipped — its guard is
`store_mode == "index" and not (self._inline_attn and torch.compiler.is_compiling())`, and warmup
traces, so `is_compiling()` is true. That loop is the only thing that enumerated every reachable
`(buffer_rows, slots)` pair, and **nothing replaces it on the inline path** (no inline-specific
enumeration exists in `spyre_model_runner.py`). Meanwhile `group` and `extent` are Python ints traced
*into* the body graph, so the body graph is keyed on them. Warmup in the fusion-OFF arm only ever
produced `4 slots x {64,128,256,512}` while the real step was `2 slots x 512` — a layout warmup never
built. But note that fixing warmup coverage alone would **not** explain a cost that repeats across
three identical calls, so coverage is necessary and not sufficient.

### ⚠️ The inline path cannot currently be verified from its log

`Encoder slot padding active: …` never appears in a fusion-ON log: blocker #2 put that
`logger.info_once` behind `not torch.compiler.is_compiling()`. So neither the slot layout nor even
*whether the slot fast path engaged* is observable, which is the handoff's §4 "silently falls
through" hazard in practice. `perf_steps.py` dumps the last call's embeddings so correctness can be
checked another way, but **the layout itself should be logged from outside the traced region** (the
runner's wrapper `__call__` already hoists the mask there — the same place would do).

## 6. Reusable harness added (`b3/`)

- `table.py` — every dump vs the golden **and its own within-run spread**, one table. Free, no device.
- `slot_kernel_probe.py` / `slot_probe.sh` — compiled slot kernel vs CPU reference, group 1/2/4,
  three references, fallbacks fatal. No engine, so it is the fast arm.
- `embed_solo.py` / `solo_arm.sh` — solo vs grouped in one process, so batch-size-dependence is one arm.
- `perf_steps.py` / `perf_steps_arm.sh` — fixed-shape per-step latency in-process with
  `SPYRE_HOSTPROBE=1`. Deliberately not `vllm bench serve`: see §5 for what that path measures instead.
- `embed_arm.sh` takes the bucket as `$3` (`default` leaves `compile_sizes` alone).
