# Lifting PowerPC AltiVec / VSX → AMD AIE2 (XDNA NPU) vector intrinsics

A practical translation table for porting SIMD kernels (e.g. the `pse-vcipher-collapse`
non-bijunctive AltiVec code) to the AIE2 vector unit on the XDNA1 NPU. The AIE2 ISA is a
different mnemonic set over the **same primitive classes** — load/store, elementwise
arithmetic, fused multiply-accumulate, compare→mask, select, broadcast, permute, reduce.

Header: `#include <aie_api/aie.hpp>`. Vectors: `aie::vector<T, N>` (e.g. `aie::vector<bfloat16,32>`,
`aie::vector<int16,32>`). Accumulators: `aie::accum<accXX, N>`. Masks: `aie::mask<N>`.

✅ = used/verified in this repo's kernels (`mul.cc`, `scale.cc`, `relu.cc`, `collapse.cc`).

| AltiVec / VSX | AIE2 (`aie::`) | Notes |
|---------------|----------------|-------|
| `vec_ld` / `*(vector*)p` | ✅ `aie::load_v<N>(p)` | aligned vector load |
| `vec_st` / `*(vector*)p = v` | ✅ `aie::store_v(p, v)` | vector store |
| `vec_splat` / `vec_splats(x)` | ✅ `aie::broadcast<T,N>(x)` | scalar → vector |
| (zero vector) | ✅ `aie::zeros<T,N>()` | `broadcast_zero_*` also exists |
| `vec_add` | `aie::add(a, b)` | elementwise add |
| `vec_sub` | ✅ `aie::sub(a, b)` | elementwise sub (the τ-shift: `aie::sub(x, tau_v)`) |
| `vec_mul` / `vec_madd` | ✅ `aie::mul(a, b)` → `aie::accum`; `aie::mac(acc, a, b)` | mul returns accum; `.to_vector<T>(shift)` to narrow |
| `vec_max` | ✅ `aie::max(a, b)` | elementwise max (ReLU = `max(x, zeros)`) |
| `vec_min` | `aie::min(a, b)` | elementwise min |
| `vec_cmpgt` | `aie::gt(a, b)` → `aie::mask<N>` | compare → mask |
| `vec_cmpge` | ✅ `aie::ge(a, b)` → `aie::mask<N>` | the collapse threshold test |
| `vec_cmpeq` | `aie::eq(a, b)` → `aie::mask<N>` | |
| `vec_sel(a, b, mask)` | ✅ `aie::select(a, b, mask)` | `mask ? b : a` — the prune/keep |
| `vec_and` / `vec_or` / `vec_xor` | `aie::bit_and / bit_or / bit_xor` | bitwise |
| `vec_perm` | `aie::shuffle(v, pattern)` / `aie::shuffle(a, b, mode)` | **the non-bijunctive collapse core** — gather/permute lanes |
| `vec_sld` / shifts | `aie::shuffle_up / shuffle_down` | lane shift |
| (horizontal sum) | `aie::reduce_add(v)` | tree reduction → scalar |
| (horizontal max/min) | `aie::reduce_max(v)` / `aie::reduce_min(v)` | scalar reduction (see `reduce_max.cc`) |
| `vec_cts` / `vec_ctf` (convert) | `v.cast_to<U>()` / `aie::to_float` / accum `.to_vector<T>(shift)` | dtype/fixed-point convert |

## Patterns

**ReLU (keep ≥ 0):**
```cpp
auto x = aie::load_v<32>(a+i);
aie::store_v(c+i, aie::max(x, aie::zeros<bfloat16,32>()));
```

**Non-bijunctive collapse (keep ≥ τ, runtime τ, on-NPU shift):**
```cpp
auto tau_v = aie::broadcast<bfloat16,32>(tau);   // vec_splat
auto x     = aie::load_v<32>(a+i);               // vec_ld
auto diff  = aie::sub(x, tau_v);                 // vec_sub  (on-NPU τ-shift)
auto keep  = aie::ge(diff, aie::zeros<bfloat16,32>()); // vec_cmpge → mask
auto out   = aie::select(aie::zeros<bfloat16,32>(), x, keep); // vec_sel: keep?x:0
aie::store_v(c+i, out);                          // vec_st
```

**Top-k cutoff (the vec_perm part):** `aie::reduce_max` for the peak, then `aie::shuffle`
to gather survivors into a dense prefix — the AIE analogue of the AltiVec `vec_perm`
prune+compact that drives `pse-vcipher-collapse`.

## Gotchas vs AltiVec

- **Mul returns an accumulator**, not a vector: `aie::accum<acc32,N> c = aie::mul(a,b);`
  then `c.to_vector<T>(shift)`. Mirrors VSX MACs but explicit about the accumulator.
- **Compare returns `aie::mask<N>`**, consumed by `aie::select` — no "all-ones vector" idiom.
- **Vector width is type-dependent** (bf16: 32-lane; int32: 16-lane). Size loops by the lane count.
- **bf16 compares** widen internally; works for `ge`/`gt` as shown in `collapse.cc`.
- AIE is a **dataflow array**: per-element data-dependent branching isn't free — express
  selection as `mask + select`, not scalar `if` (exactly like writing branchless AltiVec).

## See also
- `examples/kernels/collapse.cc` — hand-authored compare+select collapse (this repo).
- `mlir_aie/include/aie_kernels/aie2/*.cc` — AMD's kernels = reference AIE-intrinsic examples.
- AMD AIE API docs: `aie_api/aie.hpp` (the `aie::` namespace).
