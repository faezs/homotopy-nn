{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Interactive REPL for testing Einsum Python bridge

Usage:
  $ einsum-repl
  > matmul
  Result: Tensor [2,2] [89.0,98.0,116.0,128.0]
  > dot
  Result: Tensor [] [15.0]
  > quit
-}

module Main where

import Einsum.PythonBridge
import Attention.Bridge
import qualified Data.Text as T
import System.IO (hFlush, stdout)
import Control.Exception (catch)
import Control.Monad (forever, unless, when)
import Control.Monad.IO.Class (liftIO)
import Data.List (isInfixOf, isPrefixOf)
import System.Console.Haskeline

-- Helper function to split strings by delimiter
wordsBy :: (Char -> Bool) -> String -> [String]
wordsBy p s = case dropWhile p s of
  "" -> []
  s' -> w : wordsBy p s''
    where (w, s'') = break p s'

-- Example tensors with compatible shapes for common operations
exampleMatrixA :: Tensor
exampleMatrixA = Tensor [3, 2] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

exampleMatrixB :: Tensor
exampleMatrixB = Tensor [2, 4] [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
-- Changed from [3,2] to [2,4] so A[3,2] @ B[2,4] -> [3,4] works

exampleVectorV :: Tensor
exampleVectorV = Tensor [5] [1.0, 2.0, 3.0, 4.0, 5.0]

exampleVectorW :: Tensor
exampleVectorW = Tensor [5] [1.0, 1.0, 1.0, 1.0, 1.0]

-- Attention example tensors (batch=1 omitted, seq_len=2, dim=3)
-- NOTE: Contract puts contracted indices FIRST!
-- So (contract [k] [[i] [j]]) expects shapes [k,i] and [k,j]
exampleQuery :: Tensor
exampleQuery = Tensor [3, 2] [1.0, 0.0,  -- dim 0 for both tokens
                               0.0, 1.0,  -- dim 1 for both tokens
                               1.0, 0.0]  -- dim 2 for both tokens
                               -- [k, i] layout

exampleKey :: Tensor
exampleKey = Tensor [3, 2] [1.0, 0.0,    -- dim 0 for both tokens
                            1.0, 1.0,    -- dim 1 for both tokens
                            0.0, 1.0]    -- dim 2 for both tokens
                            -- [k, j] layout

exampleValue :: Tensor
exampleValue = Tensor [2, 3] [1.0, 2.0, 3.0,   -- token 1 value
                               4.0, 5.0, 6.0]   -- token 2 value
                               -- [j, h] layout

main :: IO ()
main = do
  putStrLn "Einsum Python Bridge REPL"
  putStrLn "========================="
  putStrLn ""
  putStrLn "Starting Python session..."

  withPythonSession $ \session -> do
    putStrLn "Python session ready!"
    putStrLn ""
    putStrLn "DIRECT EINSUM NOTATION (PyTorch/JAX style):"
    putStrLn "  Just type einsum directly: ij->j, ij,jk->ik, i,i->, etc."
    putStrLn ""
    putStrLn "COMMANDS:"
    putStrLn "  einsum    - Interactive einsum with custom formula"
    putStrLn "  eval      - Evaluate with custom tensor input"
    putStrLn ""
    putStrLn "EINSUM EXAMPLES:"
    putStrLn "  matmul    - Matrix multiplication"
    putStrLn "  dot       - Dot product"
    putStrLn "  transpose - Transpose a matrix"
    putStrLn "  reduce    - Sum over dimension"
    putStrLn "  broadcast - Expand vector to matrix"
    putStrLn "  reshape   - Flatten tensor"
    putStrLn "  attention - Self-attention via einsum"
    putStrLn "  parallel  - Parallel operations"
    putStrLn ""
    putStrLn "ATTENTION 3-CATEGORY (JAX/Flax via Agda formalization):"
    putStrLn "  attention-head - Single attention head (smooth map)"
    putStrLn "  multi-head     - Multi-head attention (parallel composition ⊗)"
    putStrLn "  transformer    - Full transformer block (with FFN)"
    putStrLn ""
    putStrLn "  quit      - Exit REPL (or Ctrl-D)"
    putStrLn ""
    putStrLn "Use arrow keys to navigate history"
    putStrLn ""

    -- Run REPL with history enabled
    let settings = defaultSettings
          { historyFile = Just ".einsum_history"
          , autoAddHistory = True
          }
    runInputT settings (repl session)

repl :: PythonSession -> InputT IO ()
repl session = do
  minput <- getInputLine "> "
  case minput of
    Nothing -> do  -- EOF/Ctrl-D
      liftIO $ putStrLn "Goodbye!"
      return ()
    Just cmd -> do
      processCommand session cmd
      repl session  -- Continue REPL loop

processCommand :: PythonSession -> String -> InputT IO ()
processCommand session cmd = case () of
    _ | cmd == "quit" -> do
      liftIO $ putStrLn "Goodbye!"
      return ()  -- Will exit after this

    -- Handle "einsum FORMULA" on same line
    _ | "einsum " `isPrefixOf` cmd -> do
      let formula = drop 7 cmd  -- Remove "einsum " prefix
      liftIO $ putStrLn $ "Evaluating formula: " ++ formula
      liftIO $ putStrLn "Sampling random tensors with compatible shapes..."

      result <- liftIO $ sampleEinsum session (T.pack formula)
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []

      liftIO $ unless (null $ tensorShape result) $ do
        putStrLn $ "Result shape: " ++ show (tensorShape result)
        if length (tensorData result) <= 20
          then putStrLn $ "Result data: " ++ show (tensorData result)
          else putStrLn $ "Result data: [" ++ show (take 5 (tensorData result)) ++
                        " ... " ++ show (drop (length (tensorData result) - 5) (tensorData result)) ++ "]"
      liftIO $ putStrLn ""

    _ | cmd == "eval" -> do
      liftIO $ putStrLn "Enter einsum formula (e.g., '(contract [i] [[] []])'):"
      mformula <- getInputLine "> "
      case mformula of
        Nothing -> return ()
        Just formula -> do
          liftIO $ putStrLn "How many input tensors? (e.g., 2):"
          mnumTensorsStr <- getInputLine "> "
          case mnumTensorsStr of
            Nothing -> return ()
            Just numTensorsStr -> do
              let numTensors = read numTensorsStr :: Int

              tensors <- sequence $ replicate numTensors $ do
                liftIO $ putStrLn "Enter tensor shape as comma-separated dimensions (e.g., '3,2'):"
                mshapeStr <- getInputLine "> "
                case mshapeStr of
                  Nothing -> return $ Tensor [] []
                  Just shapeStr -> do
                    let shape = map read (wordsBy (==',') shapeStr) :: [Int]

                    liftIO $ putStrLn "Enter tensor data as comma-separated values (e.g., '1.0,2.0,3.0,4.0,5.0,6.0'):"
                    mdataStr <- getInputLine "> "
                    case mdataStr of
                      Nothing -> return $ Tensor shape []
                      Just dataStr -> do
                        let tensorData = map read (wordsBy (==',') dataStr) :: [Double]
                        return $ Tensor shape tensorData

              liftIO $ putStrLn $ "Executing: " ++ formula
              liftIO $ putStrLn $ "With tensors: " ++ show tensors

              result <- liftIO $ evalEinsum session (T.pack formula) tensors
                `catch` \(e :: EinsumError) -> do
                  liftIO $ putStrLn $ "Error: " ++ show e
                  return $ Tensor [] []

              liftIO $ putStrLn $ "Result: " ++ show result
              liftIO $ putStrLn ""

    -- Interactive einsum with prompt
    _ | cmd == "einsum" -> do
      liftIO $ putStrLn "Enter einsum formula (PyTorch/JAX notation or S-expression):"
      liftIO $ putStrLn "  Examples: 'ij->j', 'ij,jk->ik', 'i,i->', or '(contract [j] [[i] [k]])'"
      mformula <- getInputLine "> "
      case mformula of
        Nothing -> return ()
        Just formula -> do
          -- Provide some example tensors
          let v = Tensor [3] [1.0, 2.0, 3.0]
          let m = Tensor [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
          let m2 = Tensor [3, 2] [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

          liftIO $ putStrLn "Using example tensors:"
          liftIO $ putStrLn $ "  v[3] = " ++ show v
          liftIO $ putStrLn $ "  m[2,3] = " ++ show m
          liftIO $ putStrLn $ "  m2[3,2] = " ++ show m2
          liftIO $ putStrLn ""

          -- Try to infer number of tensors from formula
          let numCommas = length $ filter (== ',') formula
          let numTensors = if "->" `isInfixOf` formula
                            then numCommas + 1
                            else 2  -- Default for S-expressions

          liftIO $ putStrLn $ "Formula appears to need " ++ show numTensors ++ " tensor(s)"

          -- Try appropriate tensor combinations based on inferred count
          case numTensors of
            1 -> do
              liftIO $ putStrLn "Trying single tensor operations..."
              tryEval [m] formula
              tryEval [v] formula
            2 -> do
              liftIO $ putStrLn "Trying two-tensor operations..."
              tryEval [v, v] formula
              tryEval [m, m2] formula
              tryEval [m2, m] formula
            _ -> do
              liftIO $ putStrLn "Complex formula - trying various combinations..."
              tryEval [v] formula
              tryEval [m] formula
              tryEval [v, v] formula
              tryEval [m, m2] formula
          liftIO $ putStrLn ""
          where
            tryEval tensors formula = do
              liftIO $ putStrLn $ "  Evaluating with " ++ show (map tensorShape tensors)
              result <- liftIO $ evalEinsum session (T.pack formula) tensors
                `catch` \(e :: EinsumError) -> do
                  liftIO $ putStrLn $ "    Failed: " ++ show e
                  return $ Tensor [] []
              liftIO $ unless (null $ tensorShape result) $
                putStrLn $ "    Result: " ++ show result

    _ | cmd == "matmul" -> do
      let formula = "(contract [j] [[i] [k]])"
      liftIO $ putStrLn "Matrix multiplication: A[j,i] × B[j,k] → C[i,k]"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "PyTorch equivalent: torch.einsum('ji,jk->ik', A, B)"
      liftIO $ putStrLn $ "A = " ++ show exampleMatrixA
      liftIO $ putStrLn $ "B = " ++ show exampleMatrixB
      result <- liftIO $ evalEinsum session formula [exampleMatrixA, exampleMatrixB]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "Result: " ++ show result
      liftIO $ putStrLn ""

    _ | cmd == "dot" -> do
      let formula = "(contract [i] [[] []])"
      liftIO $ putStrLn "Dot product: v[i] · w[i] → scalar"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "PyTorch equivalent: torch.einsum('i,i->', v, w)"
      liftIO $ putStrLn $ "v = " ++ show exampleVectorV
      liftIO $ putStrLn $ "w = " ++ show exampleVectorW
      result <- liftIO $ evalEinsum session formula [exampleVectorV, exampleVectorW]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "Result: " ++ show result
      liftIO $ putStrLn ""

    _ | cmd == "attention" -> do
      let formula = "(seq (contract [k] [[i] [j]]) (contract [j] [[i] [h]]))"
      liftIO $ putStrLn "Self-Attention: Q[i,k] @ K[j,k]^T → S[i,j] @ V[j,h] → O[i,h]"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "PyTorch equivalent: torch.einsum('ik,jk->ij', Q, K) then torch.einsum('ij,jh->ih', scores, V)"
      liftIO $ putStrLn $ "Q (Query) = " ++ show exampleQuery
      liftIO $ putStrLn $ "K (Key) = " ++ show exampleKey
      liftIO $ putStrLn $ "V (Value) = " ++ show exampleValue
      result <- liftIO $ evalEinsum session formula [exampleQuery, exampleKey, exampleValue]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "Result (Output): " ++ show result
      liftIO $ putStrLn ""

    _ | cmd == "transpose" -> do
      let formula = "(transpose [i j] [j i])"
      liftIO $ putStrLn "Matrix Transpose: M[i,j] → M^T[j,i]"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "PyTorch equivalent: torch.einsum('ij->ji', M)"
      let m = Tensor [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      liftIO $ putStrLn $ "M = " ++ show m
      result <- liftIO $ evalEinsum session formula [m]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "M^T = " ++ show result
      liftIO $ putStrLn ""

    _ | cmd == "reduce" -> do
      let formula = "(reduce [i j] j)"
      liftIO $ putStrLn "Column Sum: M[i,j] → v[i] (sum over j)"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "PyTorch equivalent: M.sum(dim=1)"
      let m = Tensor [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      liftIO $ putStrLn $ "M = " ++ show m
      result <- liftIO $ evalEinsum session formula [m]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "Column sums = " ++ show result
      liftIO $ putStrLn ""

    _ | cmd == "broadcast" -> do
      let formula = "(broadcast [i] [j])"
      liftIO $ putStrLn "Broadcast: v[i] → M[i,j] (repeat along j axis)"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "PyTorch equivalent: v.unsqueeze(1).expand(-1, j_size)"
      let v = Tensor [3] [1.0, 2.0, 3.0]
      liftIO $ putStrLn $ "v = " ++ show v
      result <- liftIO $ evalEinsum session formula [v]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "Broadcasted = " ++ show result
      liftIO $ putStrLn ""

    _ | cmd == "reshape" -> do
      let formula = "(reshape [i j] [ij])"
      liftIO $ putStrLn "Flatten: M[i,j] → v[i*j]"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "PyTorch equivalent: M.flatten()"
      let m = Tensor [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      liftIO $ putStrLn $ "M = " ++ show m
      result <- liftIO $ evalEinsum session formula [m]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "Flattened = " ++ show result
      liftIO $ putStrLn ""

    _ | cmd == "parallel" -> do
      let formula = "(par (contract [i] [[] []]) (contract [j] [[k] []]))"
      liftIO $ putStrLn "Parallel: dot(v1,v2) || sum(M)"
      liftIO $ putStrLn $ "Einsum formula: " ++ show formula
      liftIO $ putStrLn $ "Meaning: Run two operations on disjoint inputs"
      let v1 = Tensor [3] [1.0, 2.0, 3.0]
      let v2 = Tensor [3] [1.0, 1.0, 1.0]
      let m = Tensor [2, 2] [1.0, 2.0, 3.0, 4.0]
      liftIO $ putStrLn $ "v1 = " ++ show v1
      liftIO $ putStrLn $ "v2 = " ++ show v2
      liftIO $ putStrLn $ "M = " ++ show m
      result <- liftIO $ evalEinsum session formula [v1, v2, m]
        `catch` \(e :: EinsumError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      liftIO $ putStrLn $ "Results = " ++ show result
      liftIO $ putStrLn ""

    -- 3-Category Attention Operations (from Agda formalization)
    _ | cmd == "attention-head" -> do
      liftIO $ putStrLn "=== Single Attention Head (1-morphism: SmoothMap) ==="
      liftIO $ putStrLn "From Agda 3-category formalization: AttentionHead ℝ⁺-semiring"
      liftIO $ putStrLn "Input: [batch, seq_len, d_model] → Output: [batch, seq_len, d_v]"
      liftIO $ putStrLn ""

      let config = defaultAttentionConfig { cfgNHeads = 1, cfgDModel = 64, cfgDK = 32, cfgDV = 32 }
      let batch = 2
      let seqLen = 5
      let dModel = 64
      let input = Tensor [batch, seqLen, dModel] (replicate (batch * seqLen * dModel) 0.1)

      liftIO $ putStrLn $ "Config: " ++ show config
      liftIO $ putStrLn $ "Input shape: " ++ show (tensorShape input)
      liftIO $ putStrLn "Starting attention session..."

      result <- liftIO $ withAttentionSession $ \attSession -> do
        evalAttentionHead attSession config input
        `catch` \(e :: AttentionError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ AttentionResult (Tensor [] []) Nothing

      liftIO $ unless (null $ tensorShape $ resOutput result) $ do
        putStrLn $ "Output shape: " ++ show (tensorShape $ resOutput result)
        putStrLn $ "Output preview: " ++ show (take 10 $ tensorData $ resOutput result) ++ " ..."
        case resAttentionWeights result of
          Just weights -> do
            putStrLn $ "Attention weights shape: " ++ show (tensorShape weights)
            putStrLn $ "Attention weights preview: " ++ show (take 10 $ tensorData weights) ++ " ..."
          Nothing -> putStrLn "No attention weights returned"
      liftIO $ putStrLn ""

    _ | cmd == "multi-head" -> do
      liftIO $ putStrLn "=== Multi-Head Attention (Horizontal Composition ⊗) ==="
      liftIO $ putStrLn "From Agda 3-category: MultiHeadAttention ℝ⁺-semiring 8 512"
      liftIO $ putStrLn "Parallel composition of 8 attention heads"
      liftIO $ putStrLn "Input: [batch, seq_len, d_model] → Output: [batch, seq_len, d_model]"
      liftIO $ putStrLn ""

      let config = defaultAttentionConfig  -- 8 heads, d_model=512
      let batch = 1
      let seqLen = 10
      let dModel = 512
      let input = Tensor [batch, seqLen, dModel] (replicate (batch * seqLen * dModel) 0.1)

      liftIO $ putStrLn $ "Config: n_heads=" ++ show (cfgNHeads config) ++
                         ", d_model=" ++ show (cfgDModel config)
      liftIO $ putStrLn $ "Input shape: " ++ show (tensorShape input)
      liftIO $ putStrLn "Starting attention session..."

      result <- liftIO $ withAttentionSession $ \attSession -> do
        evalMultiHeadAttention attSession config input
        `catch` \(e :: AttentionError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ AttentionResult (Tensor [] []) Nothing

      liftIO $ unless (null $ tensorShape $ resOutput result) $ do
        putStrLn $ "Output shape: " ++ show (tensorShape $ resOutput result)
        putStrLn $ "Output preview: " ++ show (take 10 $ tensorData $ resOutput result) ++ " ..."
      liftIO $ putStrLn ""

    _ | cmd == "transformer" -> do
      liftIO $ putStrLn "=== Transformer Block (Attention + FFN) ==="
      liftIO $ putStrLn "Complete transformer block with multi-head attention and feed-forward"
      liftIO $ putStrLn "Input: [batch, seq_len, d_model] → Output: [batch, seq_len, d_model]"
      liftIO $ putStrLn ""

      let config = defaultAttentionConfig  -- 8 heads, d_model=512
      let batch = 1
      let seqLen = 8
      let dModel = 512
      let input = Tensor [batch, seqLen, dModel] (replicate (batch * seqLen * dModel) 0.1)

      liftIO $ putStrLn $ "Config: " ++ show config
      liftIO $ putStrLn $ "Input shape: " ++ show (tensorShape input)
      liftIO $ putStrLn "Starting attention session..."

      result <- liftIO $ withAttentionSession $ \attSession -> do
        evalTransformerBlock attSession config input
        `catch` \(e :: AttentionError) -> do
          liftIO $ putStrLn $ "Error: " ++ show e
          return $ AttentionResult (Tensor [] []) Nothing

      liftIO $ unless (null $ tensorShape $ resOutput result) $ do
        putStrLn $ "Output shape: " ++ show (tensorShape $ resOutput result)
        putStrLn $ "Output preview: " ++ show (take 10 $ tensorData $ resOutput result) ++ " ..."
      liftIO $ putStrLn ""

    _ | cmd == "" -> return ()  -- Empty line, do nothing

    _ -> do
      -- Check if it looks like einsum notation
      if "->" `isInfixOf` cmd || (not (null cmd) && all (`elem` ("ijklmnabcdefghopqrstuvwxyz,->" :: String)) cmd)
        then do
          -- Direct einsum evaluation with SAMPLED tensors
          liftIO $ putStrLn $ "Evaluating: " ++ cmd
          liftIO $ putStrLn "Sampling random tensors with compatible shapes..."

          -- Use the sampleEinsum function to generate compatible tensors automatically
          result <- liftIO $ sampleEinsum session (T.pack cmd)
            `catch` \(e :: EinsumError) -> do
              liftIO $ putStrLn $ "Error: " ++ show e
              return $ Tensor [] []

          liftIO $ unless (null $ tensorShape result) $ do
            putStrLn $ "Result shape: " ++ show (tensorShape result)
            -- Only show data if result is small
            if length (tensorData result) <= 20
              then putStrLn $ "Result data: " ++ show (tensorData result)
              else putStrLn $ "Result data: [" ++ show (take 5 (tensorData result)) ++
                            " ... " ++ show (drop (length (tensorData result) - 5) (tensorData result)) ++ "]"
          liftIO $ putStrLn ""
        else do
          liftIO $ putStrLn $ "Unknown command: " ++ cmd
          liftIO $ putStrLn ""
          liftIO $ putStrLn "EINSUM OPERATION REFERENCE:"
      liftIO $ putStrLn ""
      liftIO $ putStrLn "Basic Operations:"
      liftIO $ putStrLn "  (contract [contracted] [remaining])  - Sum over indices"
      liftIO $ putStrLn "  (transpose [old] [new])              - Reorder dimensions"
      liftIO $ putStrLn "  (reduce [context] index)             - Sum over one dimension"
      liftIO $ putStrLn "  (broadcast [old] [new])              - Add dimensions"
      liftIO $ putStrLn "  (reshape [old] [new])                - Change shape"
      liftIO $ putStrLn ""
      liftIO $ putStrLn "Compositions:"
      liftIO $ putStrLn "  (seq op1 op2)                        - Sequential (op2 ∘ op1)"
      liftIO $ putStrLn "  (par op1 op2)                        - Parallel on disjoint inputs"
      liftIO $ putStrLn ""
