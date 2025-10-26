{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix E.8: Examples of Linear Semantic Information

This module provides **concrete examples** of the theoretical constructs from
Appendix E, demonstrating:
- Lambek calculus for natural language syntax
- Montague grammar for compositional semantics
- Information-theoretic semantics
- Neural network connections

## Key Examples

1. **Simple Sentences**: Intransitive verbs
   - "John sleeps" : S
   - NP ⊗ (NP\\S) → S

2. **Transitive Sentences**: Transitive verbs
   - "Mary sees John" : S
   - NP ⊗ ((NP\\S)/NP) ⊗ NP → S

3. **Noun Phrases**: Determiners and nouns
   - "the cat" : NP
   - (NP/N) ⊗ N → NP

4. **Montague Semantics**: Lambda calculus interpretation
   - [[John]] = j : e
   - [[sleeps]] = λx. sleep(x) : e → t
   - [[John sleeps]] = sleep(j) : t

5. **Information Measures**:
   - Entropy of linguistic distributions
   - Compression via grammar
   - Neural language models as theories

## References

- [Lam58] Lambek (1958): The mathematics of sentence structure
- [Mon70] Montague (1970): Universal grammar
- [BB22] Belfiore & Bennequin (2022), Appendix E

-}

module Neural.Semantics.Examples where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Path
open import 1Lab.HLevel
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Instances.Sets

open import Neural.Semantics.ClosedMonoidal
open import Neural.Semantics.BiClosed
open import Neural.Semantics.LinearExponential
open import Neural.Semantics.TensorialNegation
open import Neural.Semantics.StrongMonad
open import Neural.Semantics.NegationExponential
open import Neural.Semantics.LinearInformation

private variable
  o ℓ : Level

--------------------------------------------------------------------------------
-- Lambek Calculus Examples

{-|
## Example 1: Simple Intransitive Sentence

Parse and derive "John sleeps":
- John : NP
- sleeps : NP\\S (intransitive verb)
- Result: NP ⊗ (NP\\S) → S via eval-left
-}

module SimpleSentenceExample {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C

  -- Postulate basic categories
  postulate
    NP : Ob
    S : Ob

  -- Intransitive verb category
  IV : Ob
  IV = NP \\ S

  -- Lexical items (morphisms representing words)
  postulate
    john : Hom Unit NP
    sleeps : Hom Unit IV

  -- Sentence composition: John sleeps
  -- Simplified: postulate combined morphism
  postulate
    john-sleeps : Hom Unit S

  -- In practice: would use monoidal coherence to build
  -- john ⊗ sleeps : Unit → NP ⊗ IV
  -- eval-left ∘ (john ⊗ sleeps) : Unit → S

--------------------------------------------------------------------------------
-- Transitive Verb Example

{-|
## Example 2: Transitive Sentence

Parse and derive "Mary sees John":
- Mary, John : NP
- sees : (NP\\S)/NP (transitive verb)
- Result: NP ⊗ TV ⊗ NP → S

**Derivation**:
1. sees ⊗ john : TV ⊗ NP → (NP\\S) via eval-right
2. mary ⊗ (sees@john) : NP ⊗ (NP\\S) → S via eval-left
-}

module TransitiveSentenceExample {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C

  postulate
    NP : Ob
    S : Ob

  IV : Ob
  IV = NP \\ S

  TV : Ob
  TV = IV / NP

  postulate
    mary : Hom Unit NP
    john : Hom Unit NP
    sees : Hom Unit TV

  -- Simplified: postulate result
  postulate
    mary-sees-john : Hom Unit S

  -- In practice: would construct via monoidal coherence
  -- sees ⊗ john : Unit → TV ⊗ NP
  -- eval-right : TV ⊗ NP → IV
  -- mary ⊗ (sees@john) : Unit → NP ⊗ IV
  -- eval-left : NP ⊗ IV → S

--------------------------------------------------------------------------------
-- Noun Phrase Example

{-|
## Example 3: Determiner + Noun

Parse and derive "the cat":
- the : NP/N (determiner)
- cat : N (noun)
- Result: (NP/N) ⊗ N → NP via eval-right
-}

module NounPhraseExample {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C

  postulate
    NP : Ob
    N : Ob

  Det : Ob
  Det = NP / N

  postulate
    the : Hom Unit Det
    cat : Hom Unit N

  postulate
    the-cat : Hom Unit NP

  -- In practice: eval-right ∘ (the ⊗ cat)

--------------------------------------------------------------------------------
-- Montague Semantics

{-|
## Montague Grammar Interpretation

Montague semantics interprets syntactic categories as semantic types:
- NP → e (entities)
- S → t (truth values)
- N → e → t (predicates)
- (NP\\S) → e → t (properties)
- (S\\NP)/NP → e → e → t (binary relations)

**Compositionality**: [[α ⊗ β]] = [[α]]([[β]])

**Example**:
- [[John]] = j : e
- [[sleeps]] = sleep : e → t
- [[John sleeps]] = sleep(j) : t
-}

module MontagueSemantics where
  -- Semantic types
  postulate
    e : Type  -- Entities
    t : Type  -- Truth values

  -- Predicates
  Pred : Type
  Pred = e → t

  -- Binary relations
  Rel : Type
  Rel = e → e → t

  -- Lexical semantics (postulated)
  postulate
    j : e  -- John
    m : e  -- Mary
    sleep : Pred  -- sleeps
    see : Rel  -- sees

  -- Semantic interpretation of "John sleeps"
  john-sleeps-sem : t
  john-sleeps-sem = sleep j

  -- Semantic interpretation of "Mary sees John"
  mary-sees-john-sem : t
  mary-sees-john-sem = see m j

  -- Quantifiers: "every cat sleeps"
  postulate
    cat : Pred
    every : (e → t) → (e → t) → t

  every-cat-sleeps : t
  every-cat-sleeps = every cat sleep

--------------------------------------------------------------------------------
-- Information-Theoretic Semantics

{-|
## Example 4: Information Content

Calculate information content of linguistic expressions:
- H(word) = -log P(word)
- H(sentence) = sum of word entropies (independent approximation)
- Compression via grammar: H(parse) < H(word sequence)

**Example**: "the cat sleeps"
- H(the) = 2 bits
- H(cat) = 8 bits
- H(sleeps) = 6 bits
- H(sentence) = 16 bits (uncompressed)
- H(parse) = 12 bits (with grammar compression)
-}

module InformationTheoreticSemantics {o ℓ} {C : Precategory o ℓ}
                                      {M : Monoidal-category C}
                                      (Th : Theory C M) where
  open Precategory C
  open Monoidal-category M
  open Theory Th
  open KolmogorovComplexity Th
  open ShannonEntropy

  -- Word as object
  postulate
    Word : Ob

  -- Sentence as composition
  postulate
    Sentence : Ob
    sentence-composition : Hom (Word ⊗ Word ⊗ Word) Sentence

  -- Information measures
  postulate
    word-entropy : ℝ
    sentence-entropy : ℝ
    grammar-compression : sentence-entropy ≤ℝ ((word-entropy +ℝ word-entropy) +ℝ word-entropy)

  -- Theory T represents grammar knowledge
  -- T(Word) = compressed representation using grammatical structure
  compressed-word : Ob
  compressed-word = T Word

  -- Compression quality
  postulate
    grammar-efficiency : ℝ₊

--------------------------------------------------------------------------------
-- Neural Network Interpretation

{-|
## Example 5: Neural Language Models

Neural language models as comonadic theories:
- T(Text) = Hidden representations (embeddings)
- ε: T(Text) → Text (decoder/generation)
- δ: T(Text) → T(T(Text)) (meta-representations)

**GPT/BERT example**:
- Input: Token sequence
- T(Input) = Contextualized embeddings
- ε: Generate next token (sampling)
- Kleisli morphisms: Text → T(Text) (encoding)
-}

module NeuralLanguageModel {o ℓ} {C : Precategory o ℓ}
                           {M : Monoidal-category C}
                           (Th : Theory C M) where
  open Precategory C
  open Monoidal-category M
  open Theory Th

  -- Token sequence
  postulate
    Tokens : Ob

  -- Embeddings (compressed representation)
  Embedding : Ob
  Embedding = T Tokens

  -- Encoder: Tokens → Embeddings
  encoder : Hom Tokens Embedding
  encoder = BarComplex.unit-map Th  -- Would be constructed from neural network weights

  -- Decoder: Embeddings → Tokens (via counit)
  decoder : Hom Embedding Tokens
  decoder = ε

  -- Meta-representation (contextualized embeddings)
  MetaEmbedding : Ob
  MetaEmbedding = T (T Tokens)

  -- Generate meta-representations
  contextualize : Hom Embedding MetaEmbedding
  contextualize = δ

  -- Next-token prediction: T(Tokens) → Tokens
  next-token : Hom Embedding Tokens
  next-token = ε  -- Decode to next token

--------------------------------------------------------------------------------
-- Dialogue Category Example

{-|
## Example 6: Question-Answer Dialogue

Use dialogue categories for question-answer semantics:
- Question: Q (proposition)
- Answer: A (proposition)
- Q ℘ A: Par (parallel composition)
- ¬'Q ⊗ A: Sequent (Q entails A)

**Example**: "Who sleeps?" / "John sleeps"
- Question: ¬'NP ⊗ S (missing NP)
- Answer: NP (John)
- Composition: (¬'NP ⊗ S) ⊗ NP → S
-}

module DialogueExample {o ℓ} (C : DialogueCategory o ℓ) where
  open DialogueCategory C

  postulate
    NP : Ob
    S : Ob

  -- Question: "Who sleeps?" = ¬'NP ⊗ S
  Question : Ob
  Question = (¬' NP) ⊗ S

  -- Answer: NP (entity)
  Answer : Ob
  Answer = NP

  -- Composition: Question ⊗ Answer → S
  postulate
    resolve : Hom (Question ⊗ Answer) S

  -- Using par: Q ℘ A = ¬'(¬'Q ⊗ ¬'A)
  question-par-answer : Ob
  question-par-answer = Question ℘ Answer

--------------------------------------------------------------------------------
-- Exponential Modality Example

{-|
## Example 7: Resource-Aware Semantics

Use linear exponential ! for resource-aware composition:
- !Word: Duplicable word (closed class)
- Word: Resource-sensitive word (open class)
- Grammar rules: !(A⊸B) allows multiple applications

**Example**: "very very big"
- very : !(Adj ⊸ Adj)
- big : Adj
- Result: very(very(big)) via δ (duplication)
-}

module ResourceAwareExample {o ℓ} {C : Precategory o ℓ}
                            {M : Monoidal-category C}
                            (E : has-exponential-comonad C) where
  open Precategory C
  open Monoidal-category M
  open has-exponential-comonad E

  postulate
    Adj : Ob  -- Adjectives

  -- Intensifier: Adj ⊸ Adj (consumes adjective, produces intensified)
  postulate
    Intensifier : Ob

  -- Duplicable intensifier
  DuplicableIntensifier : Ob
  DuplicableIntensifier = ! Intensifier

  postulate
    very : Hom Unit DuplicableIntensifier
    big : Hom Unit Adj

  -- Apply twice via duplication
  -- very-very-big : δ(very) : ! (! Intensifier)
  postulate
    double-application : Hom ((! (! Intensifier)) ⊗ Adj) Adj

--------------------------------------------------------------------------------
-- Complete Example: Complex Sentence

{-|
## Example 8: Complex Sentence with Modifiers

"The big cat sleeps quietly"

**Syntax**:
- the : NP/N
- big : N/N (adjective)
- cat : N
- sleeps : NP\\S
- quietly : (NP\\S)\\(NP\\S) (adverb)

**Derivation**:
1. big ⊗ cat : (N/N) ⊗ N → N via eval-right
2. the ⊗ (big@cat) : (NP/N) ⊗ N → NP via eval-right
3. sleeps ⊗ quietly : IV ⊗ (IV\\IV) → IV via eval-right
4. the-big-cat ⊗ sleeps-quietly : NP ⊗ IV → S via eval-left
-}

module ComplexSentenceExample {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C

  postulate
    NP : Ob
    N : Ob
    S : Ob

  IV : Ob
  IV = NP \\ S

  Det : Ob
  Det = NP / N

  Adj : Ob
  Adj = N / N

  Adv : Ob
  Adv = IV \\ IV

  postulate
    the : Hom Unit Det
    big : Hom Unit Adj
    cat : Hom Unit N
    sleeps : Hom Unit IV
    quietly : Hom Unit Adv

  -- Simplified: postulate result
  postulate
    sentence : Hom Unit S

  -- In practice: multi-step derivation via monoidal coherence
  -- big ⊗ cat : Unit → N
  -- the ⊗ (big@cat) : Unit → NP
  -- sleeps ⊗ quietly : Unit → IV
  -- (the@big@cat) ⊗ (sleeps@quietly) : Unit → S

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary of Examples

This module demonstrated:

1. **Simple sentences**: NP ⊗ (NP\\S) → S
2. **Transitive verbs**: NP ⊗ ((NP\\S)/NP) ⊗ NP → S
3. **Noun phrases**: (NP/N) ⊗ N → NP
4. **Montague semantics**: Compositional λ-calculus interpretation
5. **Information theory**: Entropy and compression via grammar
6. **Neural models**: Language models as comonadic theories
7. **Dialogue**: Question-answer via negation and par
8. **Resources**: Linear exponential ! for duplication
9. **Complex sentences**: Multi-step derivations with modifiers

**Key insights**:
- Bi-closed categories naturally model word order sensitivity
- Exponential ! captures grammatical regularities (closed vs open class)
- Comonads model compression and contextualization in neural models
- Information measures quantify semantic content
- Dialogue categories handle interactive semantics

**Future work**:
- Dependent types for semantic precision
- Quantum models for semantic ambiguity
- Neural-symbolic integration
- Cross-lingual transfer
-}
