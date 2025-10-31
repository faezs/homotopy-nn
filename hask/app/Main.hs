{-# LANGUAGE OverloadedStrings #-}

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
import qualified Data.Text as T
import System.IO (hFlush, stdout)
import Control.Exception (catch)
import Control.Monad (forever)

-- Example tensors
exampleMatrixA :: Tensor
exampleMatrixA = Tensor [3, 2] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

exampleMatrixB :: Tensor
exampleMatrixB = Tensor [3, 2] [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

exampleVectorV :: Tensor
exampleVectorV = Tensor [5] [1.0, 2.0, 3.0, 4.0, 5.0]

exampleVectorW :: Tensor
exampleVectorW = Tensor [5] [1.0, 1.0, 1.0, 1.0, 1.0]

main :: IO ()
main = do
  putStrLn "Einsum Python Bridge REPL"
  putStrLn "========================="
  putStrLn ""
  putStrLn "Starting Python session..."

  withPythonSession $ \session -> do
    putStrLn "Python session ready!"
    putStrLn ""
    putStrLn "Available commands:"
    putStrLn "  matmul  - Matrix multiplication example"
    putStrLn "  dot     - Dot product example"
    putStrLn "  quit    - Exit REPL"
    putStrLn ""

    repl session

repl :: PythonSession -> IO ()
repl session = forever $ do
  putStr "> "
  hFlush stdout
  cmd <- getLine

  case cmd of
    "quit" -> do
      putStrLn "Goodbye!"
      return ()

    "matmul" -> do
      putStrLn "Matrix multiplication: A[j,i] × B[j,k] → C[i,k]"
      putStrLn $ "A = " ++ show exampleMatrixA
      putStrLn $ "B = " ++ show exampleMatrixB
      result <- evalEinsum session "(contract [j] [[i] [k]])" [exampleMatrixA, exampleMatrixB]
        `catch` \(e :: EinsumError) -> do
          putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      putStrLn $ "Result: " ++ show result
      putStrLn ""

    "dot" -> do
      putStrLn "Dot product: v[i] · w[i] → scalar"
      putStrLn $ "v = " ++ show exampleVectorV
      putStrLn $ "w = " ++ show exampleVectorW
      result <- evalEinsum session "(contract [i] [[] []])" [exampleVectorV, exampleVectorW]
        `catch` \(e :: EinsumError) -> do
          putStrLn $ "Error: " ++ show e
          return $ Tensor [] []
      putStrLn $ "Result: " ++ show result
      putStrLn ""

    "" -> return ()  -- Empty line, do nothing

    _ -> putStrLn $ "Unknown command: " ++ cmd
