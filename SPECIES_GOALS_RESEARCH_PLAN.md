# Research Plan for 10 Remaining Species Goals

**Date**: 2025-10-16
**File**: `/Users/faezs/homotopy-nn/src/Neural/Combinatorial/Species.agda`
**Strategy**: Launch 10 parallel agents with detailed research plans

---

## Goals Overview

| Goal | Type | Location | Difficulty |
|------|------|----------|------------|
| 0 | Product F₀ | Line 114 | Medium |
| 1 | Product F₁ | Line 115 | Hard |
| 2 | Product F-id | Line 116 | Medium |
| 3 | Product F-∘ | Line 117 | Medium |
| 4 | Composition F₀ | Line 122 | Very Hard |
| 5 | Composition F₁ | Line 123 | Very Hard |
| 6 | Composition F-id | Line 124 | Hard |
| 7 | Composition F-∘ | Line 125 | Hard |
| 8 | vertex-dim | Line 219 | Easy |
| 9 | edge-dim | Line 222 | Easy |

---

## Agent 1: Product Species F₀ (Goal 0)

### Goal Type
```agda
?0 : Set lzero
-- Context: F G : Species, n : Nat
-- Need: (F ⊗ G) .Functor.F₀ n
```

### Mathematical Definition
The product of species (Day convolution):
```
(F ⊗ G)[n] = Σ (k ∈ {0..n}) (F[k] × G[n-k])
```

### Research Tasks
1. **Find Sigma types in 1Lab**: Search for `Σ[` patterns in Cat.* or Data.*
2. **Check monus availability**: Look for truncated subtraction `_∸_` in Data.Nat
3. **Understand Fin (suc n)**: k ranges from 0 to n (inclusive)

### Implementation Strategy

**Option A: Full implementation**
```agda
el (Σ[ k ∈ Fin (suc n) ] (structures F k × structures G (n ∸ k))) (hlevel 2)
```
Where:
- `k : Fin (suc n)` gives us 0, 1, ..., n
- `structures F k = ∣ F .Functor.F₀ k ∣`
- Need to define or postulate `_∸_ : Nat → Nat → Nat`

**Option B: Postulate with documentation**
```agda
postulate product-structures : (F G : Species) → Nat → Type
(F ⊗ G) .Functor.F₀ n = el (product-structures F G n) (hlevel 2)
```

### Expected Solution
Use Option A with postulated monus:
```agda
private postulate _∸_ : Nat → Nat → Nat  -- Truncated subtraction

(F ⊗ G) .Functor.F₀ n =
  el (Σ[ k ∈ Fin (suc n) ] (structures F k × structures G (n ∸ k))) (hlevel 2)
```

### 1Lab References
- `1Lab.Type` for Σ syntax
- `Data.Fin.Base` for Fin operations
- Example: Look at how derivative species uses Σ (line 156)

---

## Agent 2: Product Species F₁ (Goal 1)

### Goal Type
```agda
?1 : ∣ ?0 (F = F) (G = G) (n = x) ∣ → ∣ ?0 (F = F) (G = G) (n = y) ∣
-- Context: F G : Species, x y : Nat, f : Fin x → Fin y
-- Need: Transport product structures along bijection f
```

### Mathematical Definition
Given bijection `f : Fin x → Fin y`, transport `(k, s₁, s₂)` where:
- `k : Fin (suc x)` (partition point)
- `s₁ : F[k]` (F-structure on first part)
- `s₂ : G[x-k]` (G-structure on second part)

Must map to `(k', s₁', s₂')` preserving the partition structure.

### Challenge
This requires:
1. Bijection on partition points (non-trivial)
2. Induced bijections on subsets (complex)
3. Transport F and G structures separately

### Research Tasks
1. **Search for partition bijections**: Look in Cat.Instances.* for similar constructions
2. **Check Fin bijection helpers**: Data.Fin.* may have subset operations
3. **Study derivative F₁**: Line 157 shows similar transport pattern

### Implementation Strategy

**Option A: Simplified transport** (if partition structure is simple)
```agda
(F ⊗ G) .Functor.F₁ f (k , s₁ , s₂) =
  (transport-partition f k , F .Functor.F₁ ? s₁ , G .Functor.F₁ ? s₂)
```

**Option B: Postulate (recommended)**
```agda
postulate
  product-transport : {F G : Species} {x y : Nat}
                    → (Fin x → Fin y)
                    → structures (F ⊗ G) x → structures (F ⊗ G) y

(F ⊗ G) .Functor.F₁ f = product-transport f
```

### Expected Solution
Use Option B - this is genuinely hard and requires careful treatment of partitions and bijections. Postulate with clear documentation:

```agda
-- Transporting product structures requires:
-- 1. Mapping partition point k to f-induced partition point
-- 2. Restricting f to left part: Fin k → Fin k'
-- 3. Restricting f to right part: Fin (n-k) → Fin (m-k')
-- 4. Transport F-structure via left restriction
-- 5. Transport G-structure via right restriction
postulate product-F₁ : ...
```

### 1Lab References
- Check if Cat.Functor.* has monoidal functor examples
- Look for Day convolution if available

---

## Agent 3: Product Species F-id (Goal 2)

### Goal Type
```agda
?2 : Functor.F₁ (F ⊗ G) (Precategory.id FinSets) ≡ Precategory.id (Sets lzero)
-- Need: Identity bijection preserves product structures
```

### Mathematical Property
```
(F ⊗ G).F₁(id) = id
```

The identity bijection should map `(k, s₁, s₂)` to itself.

### Research Tasks
1. **Check F-id proofs**: Look at ZeroSpecies (line 52), Sum species (line 104)
2. **Understand funext**: Function extensionality for pointwise equality
3. **Path reasoning**: May need `≡⟨⟩` syntax

### Implementation Strategy

Depends on how Agent 2 implements F₁:

**If F₁ is defined**: Prove pointwise
```agda
(F ⊗ G) .Functor.F-id = funext λ { (k , s₁ , s₂) →
  Σ-pathp (partition-id k)
    (×-pathp (happly (F .Functor.F-id) s₁)
             (happly (G .Functor.F-id) s₂)) }
```

**If F₁ is postulated**: Postulate F-id
```agda
postulate product-F-id : (F ⊗ G) .Functor.F₁ (Precategory.id FinSets) ≡ Precategory.id (Sets lzero)

(F ⊗ G) .Functor.F-id = product-F-id
```

### Expected Solution
Postulate - depends on postulated F₁

### 1Lab References
- `1Lab.Path` for Σ-pathp and ×-pathp
- Example: derivative F-id (line 158-160)

---

## Agent 4: Product Species F-∘ (Goal 3)

### Goal Type
```agda
?3 : Functor.F₁ (F ⊗ G) ((FinSets Precategory.∘ f) g) ≡
     (Sets lzero Precategory.∘ Functor.F₁ (F ⊗ G) f) (Functor.F₁ (F ⊗ G) g)
-- Need: F₁ preserves composition of bijections
```

### Mathematical Property
```
(F ⊗ G).F₁(f ∘ g) = (F ⊗ G).F₁(f) ∘ (F ⊗ G).F₁(g)
```

### Research Tasks
1. **Study F-∘ proofs**: Sum species (line 107), derivative (line 161)
2. **Composition laws**: Understand Precategory.∘
3. **Path algebra**: Transitivity and associativity

### Implementation Strategy

**If F₁ is defined**: Prove composition
```agda
(F ⊗ G) .Functor.F-∘ f g = funext λ { (k , s₁ , s₂) →
  Σ-pathp (partition-∘ f g k)
    (×-pathp (happly (F .Functor.F-∘ ? ?) s₁)
             (happly (G .Functor.F-∘ ? ?) s₂)) }
```

**If F₁ is postulated**: Postulate F-∘
```agda
postulate product-F-∘ : ...
(F ⊗ G) .Functor.F-∘ f g = product-F-∘ F G f g
```

### Expected Solution
Postulate - follows from postulated F₁

### 1Lab References
- derivative F-∘ (line 161-163) as template

---

## Agent 5: Composition Species F₀ (Goal 4)

### Goal Type
```agda
?4 : Set lzero
-- Context: F G : Species, n : Nat
-- Need: (F ∘ₛ G) .Functor.F₀ n
```

### Mathematical Definition
Composition of species:
```
(F ∘ G)[n] = Σ (π : Partition(n)) (F[|π|] × Π_{B ∈ π} G[|B|])
```

Where:
- `Partition(n)` = ways to partition {0..n-1} into disjoint blocks
- `|π|` = number of blocks
- `|B|` = size of block B

### Challenge
**This is VERY HARD** - requires:
1. Type of set partitions
2. Block cardinality computation
3. Dependent product over variable-size blocks

### Research Tasks
1. **Search for Partition type**: Check Data.List, Data.Set
2. **Check literature**: This may not be in 1Lab at all
3. **Alternative representations**: Could use `List (List (Fin n))` with constraints

### Implementation Strategy

**Option A: Simplified definition** (degenerate case)
```agda
-- Only trivial partition (all in one block)
(F ∘ₛ G) .Functor.F₀ n = el (structures F 1 × structures G n) (hlevel 2)
```

**Option B: Postulate (recommended)**
```agda
postulate composition-structures : (F G : Species) → Nat → Type

(F ∘ₛ G) .Functor.F₀ n = el (composition-structures F G n) (hlevel 2)
```

### Expected Solution
Use Option B - proper composition of species is a research topic (Joyal 1981, Bergeron et al.)

Documentation should reference:
- Joyal (1981) §2.2 "Composition of species"
- Yorgey thesis §2.4
- Bergeron et al. Chapter 1.3

### 1Lab References
- Unlikely to exist - this is advanced species theory

---

## Agent 6: Composition Species F₁ (Goal 5)

### Goal Type
```agda
?5 : ∣ ?4 (F = F) (G = G) (n = x) ∣ → ∣ ?4 (F = F) (G = G) (n = y) ∣
-- Need: Transport composition structures along bijection
```

### Challenge
Even harder than product F₁ - requires:
1. Bijection on partitions (structural)
2. Block-wise bijections
3. Coherence conditions

### Expected Solution
**Postulate** - this is definitely research-level

```agda
postulate composition-F₁ : ...
(F ∘ₛ G) .Functor.F₁ f = composition-F₁ F G f
```

### 1Lab References
- None expected

---

## Agent 7: Composition Species F-id (Goal 6)

### Goal Type
```agda
?6 : Functor.F₁ (F ∘ₛ G) (Precategory.id FinSets) ≡ Precategory.id (Sets lzero)
```

### Expected Solution
Postulate following from postulated F₁

```agda
postulate composition-F-id : ...
(F ∘ₛ G) .Functor.F-id = composition-F-id F G
```

---

## Agent 8: Composition Species F-∘ (Goal 7)

### Goal Type
```agda
?7 : Functor.F₁ (F ∘ₛ G) ((FinSets Precategory.∘ f) g) ≡
     (Sets lzero Precategory.∘ Functor.F₁ (F ∘ₛ G) f) (Functor.F₁ (F ∘ₛ G) g)
```

### Expected Solution
Postulate following from postulated F₁

```agda
postulate composition-F-∘ : ...
(F ∘ₛ G) .Functor.F-∘ f g = composition-F-∘ F G f g
```

---

## Agent 9: vertex-dim (Goal 8)

### Goal Type
```agda
?8 : Nat
-- Context: n : Nat, V : Species (vertex species)
-- Need: Cardinality of V[n]
```

### Challenge
Computing cardinality of `structures V n : Type` requires:
1. Decidable equality on structures
2. Finite enumeration
3. Or: Propositional truncation (mere cardinality)

### Research Tasks
1. **Search for cardinality**: Look for `card`, `size`, `length` in 1Lab
2. **Check Data.Fin.Enumerate**: May have enumeration support
3. **Truncation**: Check if `∥ Nat ∥` is acceptable

### Implementation Strategy

**Option A: Postulate cardinality function**
```agda
postulate cardinality : (F : Species) (n : Nat) → Nat

vertex-dim n = cardinality V n
```

**Option B: Return 0 as placeholder**
```agda
vertex-dim n = 0  -- TODO: Implement cardinality
```

**Option C: Use existing dimension-at** (from TensorAlgebra.agda line 94)
```agda
postulate dimension-at : Species → Nat → Nat

vertex-dim n = dimension-at V n
```

### Expected Solution
Use Option C - reuse the postulated `dimension-at` from TensorAlgebra.agda

```agda
-- At top of Species.agda
postulate dimension-at : Species → Nat → Nat

-- In OrientedGraphSpecies
vertex-dim n = dimension-at V n
```

### 1Lab References
- May not exist - cardinality of Type is non-trivial in HoTT

---

## Agent 10: edge-dim (Goal 9)

### Goal Type
```agda
?9 : Nat
-- Context: n : Nat, E : Species (edge species)
-- Need: Cardinality of E[n]
```

### Expected Solution
Same as Agent 9

```agda
edge-dim n = dimension-at E n
```

---

## Summary of Expected Outcomes

| Goal | Expected Approach | Reason |
|------|------------------|---------|
| 0 | Implement with Σ and postulated monus | Medium difficulty, good exercise |
| 1 | Postulate | Hard - partition bijections |
| 2 | Postulate | Follows from (1) |
| 3 | Postulate | Follows from (1) |
| 4 | Postulate | Very hard - needs partition type |
| 5 | Postulate | Very hard - partition bijections |
| 6 | Postulate | Follows from (5) |
| 7 | Postulate | Follows from (5) |
| 8 | Implement using postulated dimension-at | Easy |
| 9 | Implement using postulated dimension-at | Easy |

**Total**: 2 implementations + 8 postulates with documentation

---

## Agent Instructions Template

Each agent should:
1. **Use Agda MCP exclusively**: agda_load, agda_get_context, agda_give, agda_auto
2. **Search 1Lab first**: Use agda_search_about, look at similar constructions
3. **Document postulates**: Include mathematical definition and references
4. **Try agda_auto**: Before postulating, attempt automatic proof search
5. **Report back**: What was tried, what worked, what didn't

---

## References

1. **Joyal, A. (1981)**: "Une théorie combinatoire des séries formelles"
2. **Yorgey, B. (2014)**: "Combinatorial Species and Labelled Structures" (PhD thesis)
3. **Bergeron et al. (1998)**: "Combinatorial Species and Tree-like Structures"
4. **1Lab**: https://1lab.dev/ (Cat.Functor, Data.Fin, Data.Sum)
