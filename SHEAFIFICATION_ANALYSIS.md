# Sheafification and Left-Exactness for Fork Topology

**Date**: 2025-10-15
**Context**: Proving `is-lex Sheafification` for fork-coverage topology

## Summary

The paper (Topos of DNNs, lines 572-577) provides an **explicit construction** of sheafification for the fork topology, which is much simpler than the general case. This gives us a path to constructively prove that sheafification preserves finite limits.

## Key Insight from the Paper

> "The sheafification process, associating a sheaf X★ over (C,J) to any presheaf X over C is **easy to describe**: no value is changed except at a place A★, where X_A★ is replaced by the product X★_A★ of the X_a', and the map from X★_A = X_A to X★_A★ is replaced by the product of the maps from X_A to the X_a' given by the functor X."

**Translation**: Sheafification for fork-coverage is defined pointwise:
- **At original vertices**: `X_sheaf(v) = X(v)` (unchanged)
- **At fork-tang A**: `X_sheaf(A) = X(A)` (unchanged)
- **At fork-star A★**: `X_sheaf(A★) = ∏_{a'→A★} X(a')` (product of incoming)

## Why This Makes Left-Exactness Easier

### Standard Grothendieck Sheafification (Hard)
- Uses colimit-of-limits construction
- Involves covering sieves and matching families
- Requires complex HIT (Higher Inductive Type) proofs
- Generic for any Grothendieck topology

### Fork-Specific Sheafification (Easier!)
- **Explicit pointwise construction**
- Only modifies fork-star vertices
- Product construction at A★
- **Products preserve limits!**

## Proof Strategy for `pres-⊤` (Terminal Preservation)

**Goal**: Show that if T is terminal in presheaves, then Sheafification(T) is terminal in sheaves.

**Proof Outline**:
1. **Terminal in presheaves**: T(x) ≅ {★} (singleton) for all x
2. **At fork-star after sheafification**:
   ```
   T_sheaf(A★) = ∏_{a'→A★} T(a')
               = ∏_{a'→A★} {★}
               ≅ {★}  (product of singletons is singleton)
   ```
3. **At other vertices**: unchanged, so still singleton
4. **Therefore**: T_sheaf is singleton everywhere → terminal in sheaves ✓

**Key Lemma Needed**: Product of terminal objects is terminal
- This is standard category theory
- Should be in 1Lab or easy to prove

## Proof Strategy for `pres-pullback` (Pullback Preservation)

**Goal**: Show that if P is pullback of X ← Y → Z in presheaves, then Sheafification preserves this pullback square.

**Proof Outline**:
1. **Pullback in presheaves computed pointwise**:
   ```
   P(x) = X(x) ×_{Y(x)} Z(x)  for all x
   ```

2. **At fork-star after sheafification**:
   ```
   P_sheaf(A★) = ∏_{a'→A★} P(a')
               = ∏_{a'→A★} (X(a') ×_{Y(a')} Z(a'))
               = (∏ X(a')) ×_{∏ Y(a')} (∏ Z(a'))   [products preserve limits!]
               = X_sheaf(A★) ×_{Y_sheaf(A★)} Z_sheaf(A★)
   ```

3. **At other vertices**: unchanged, so pullback squares preserved

4. **Therefore**: Sheafified diagram is still a pullback ✓

**Key Lemma Needed**: Products preserve pullbacks
- Equivalently: Limits commute with limits
- This is standard category theory
- Cartesian closure of Set guarantees this

## Implementation Status

**Current Code** (Architecture.agda lines 603-622):
```agda
fork-sheafification-lex : is-lex (Sheafification {C = Fork-Category} {J = fork-coverage})
fork-sheafification-lex .is-lex.pres-⊤ {T} term-psh = sheaf-term
  where
    sheaf-term : is-terminal Sh[...] (Functor.₀ Sheafification T)
    sheaf-term = {!!}  -- TODO

fork-sheafification-lex .is-lex.pres-pullback pb-psh = {!!}  -- TODO
```

## What's Needed to Complete the Proof

### For Terminal Preservation:
1. **Use adjunction**: Sheafification ⊣ ι (inclusion)
   - We have `Sheafification⊣ι` imported
   - Left adjoints preserve colimits
   - But we need terminals (limits)...

2. **Alternative approach**: Direct construction
   - Show that for any sheaf F, ∃! morphism F → Sheafification(T)
   - Use the fact that T is terminal in presheaves
   - Compose with sheafification unit: F → ι(Sheafification(F)) → ι(Sheafification(T))
   - Use adjunction to get F → Sheafification(T)

3. **Simplest approach** (if fork sheafification is computable):
   - Compute Sheafification(T) explicitly at each vertex
   - Show it's singleton everywhere
   - Invoke terminal-is-singleton lemma

### For Pullback Preservation:
1. **Product-limit interchange**:
   - Prove: ∏_i (X_i ×_Y_i Z_i) ≅ (∏_i X_i) ×_{∏_i Y_i} (∏_i Z_i)
   - This should be in 1Lab or provable from Set being cartesian closed

2. **Pointwise argument**:
   - Pullbacks in presheaf categories are computed pointwise
   - Sheafification at non-fork-star: identity
   - Sheafification at fork-star: product
   - Products preserve pullbacks
   - Therefore sheafified diagram is pullback

3. **Universal property verification**:
   - Given a cone to sheafified diagram
   - Need to show unique factorization through sheafified pullback
   - Use pointwise uniqueness + product universal property

## Dependencies on 1Lab

**What we need from 1Lab**:
1. ✓ `Sheafification` functor (have it)
2. ✓ `Sheafification⊣ι` adjunction (have it)
3. ✓ `fork-coverage` topology (have it)
4. ❓ Product of terminals is terminal
5. ❓ Products preserve pullbacks in Set
6. ❓ Limit-limit interchange

**Search strategy**:
```bash
# Terminal lemmas
find 1lab -name "*.agda" -exec grep -l "product.*terminal\|terminal.*product" {} \;

# Pullback lemmas
find 1lab -name "*.agda" -exec grep -l "product.*pullback\|pullback.*product" {} \;

# Limit interchange
find 1lab -name "*.agda" -exec grep -l "limit.*commute\|limit.*interchange" {} \;
```

## Comparison to General Case

### General Grothendieck Topology
- Sheafification: `L(X)(U) = colim_{S ∈ Cov(U)} lim_{V→U ∈ S} X(V)`
- Complex colimit-limit construction
- Proving lex requires showing this preserves all finite limits
- Research-level difficulty

### Fork Topology (Our Case)
- Sheafification: Pointwise, only changing fork-stars to products
- Simple product construction
- Proving lex reduces to showing products preserve limits
- Undergraduate-level category theory!

## Conclusion

**The fork topology's simplicity is our advantage!**

The paper explicitly tells us that sheafification is "easy to describe" for fork-coverage. We should leverage this explicit construction rather than trying to prove the general sheafification-is-lex theorem.

**Recommendation**:
1. Search 1Lab for product-limit lemmas
2. If found: Use them to fill the holes directly
3. If not found: Prove the product-limit lemmas (they're standard)
4. Build up to the full pres-⊤ and pres-pullback proofs

**Status**: Two holes remaining, but with clear path forward based on paper's explicit construction.
