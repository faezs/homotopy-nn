{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module: Attention.Bridge
Description: FFI bridge to Python JAX attention session

Manages persistent Python subprocess for executing attention operations
compiled from Agda 3-category formalization.

## Architecture

```
Agda (Attention/Tricategory.agda)
    ↓ 3-category formalization
Haskell (Attention/Bridge.hs)
    ↓ JSON over stdin/stdout
Python Session (attention_session.py)
    ↓ JAX/Flax execution
Attention Results + Weights
```

## Usage

@
session <- startAttentionSession
output <- evalAttentionHead session config input
stopAttentionSession session
@

## Protocol

Request (JSON):
{
  "operation": "attention_head",
  "config": {
    "n_heads": 1,
    "d_model": 512,
    "d_k": 64,
    "d_v": 64,
    "dropout": 0.1
  },
  "inputs": [
    {"shape": [1, 10, 512], "data": [...]},
    {"shape": [1, 10, 10], "data": [...]}  // optional mask
  ]
}

Response (JSON):
{
  "success": true,
  "output": {"shape": [1, 10, 64], "data": [...]},
  "attention_weights": {"shape": [1, 10, 10], "data": [...]}
}
-}

module Attention.Bridge
  ( -- * Session Management
    AttentionSession
  , startAttentionSession
  , stopAttentionSession
  , withAttentionSession

    -- * Configuration
  , AttentionConfig(..)
  , defaultAttentionConfig

    -- * Operations
  , AttentionOp(..)
  , evalAttention
  , evalAttentionHead
  , evalMultiHeadAttention
  , evalTransformerBlock

    -- * Response Types
  , AttentionResult(..)
  , AttentionError(..)

    -- * Re-export tensor types
  , Tensor(..)
  , Shape
  , TensorData
  ) where

import Control.Exception (Exception, throwIO, bracket, catch)
import Control.Monad (unless, when)
import Data.Aeson (FromJSON, ToJSON, (.=), (.:), (.:?), object, toJSON)
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as AKey
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BLC
import Data.Text (Text)
import qualified Data.Text as T
import System.IO (Handle, hFlush, hSetBuffering, BufferMode(..), hGetLine, hGetContents)
import System.IO.Error (IOError)
import System.Process
import System.Directory (doesFileExist)
import System.Environment (lookupEnv)

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

-- | Python subprocess session for attention operations
data AttentionSession = AttentionSession
  { attStdin  :: Handle
  , attStdout :: Handle
  , attStderr :: Handle
  , attProc   :: ProcessHandle
  }

-- | Tensor shape (list of dimensions)
type Shape = [Int]

-- | Tensor data (flattened in row-major order)
type TensorData = [Double]

-- | Tensor with shape and data
data Tensor = Tensor
  { tensorShape :: Shape
  , tensorData  :: TensorData
  } deriving (Show, Eq)

instance ToJSON Tensor where
  toJSON Tensor{..} = A.object
    [ "shape" .= tensorShape
    , "data"  .= tensorData
    ]

instance FromJSON Tensor where
  parseJSON = A.withObject "Tensor" $ \o -> do
    tensorShape <- o .: "shape"
    tensorData  <- o .: "data"
    return Tensor{..}

-- | Attention configuration (from 3-category formalization)
data AttentionConfig = AttentionConfig
  { cfgNHeads    :: Int     -- ^ Number of attention heads
  , cfgDModel    :: Int     -- ^ Model dimension (d_model)
  , cfgDK        :: Int     -- ^ Key/Query dimension (d_k)
  , cfgDV        :: Int     -- ^ Value dimension (d_v)
  , cfgDropout   :: Double  -- ^ Dropout rate (0.0 to 1.0)
  , cfgSeqLen    :: Maybe Int  -- ^ Optional max sequence length
  } deriving (Show, Eq)

instance ToJSON AttentionConfig where
  toJSON AttentionConfig{..} = A.object $
    [ "n_heads"  .= cfgNHeads
    , "d_model"  .= cfgDModel
    , "d_k"      .= cfgDK
    , "d_v"      .= cfgDV
    , "dropout"  .= cfgDropout
    ] ++ maybe [] (\sl -> ["seq_len" .= sl]) cfgSeqLen

instance FromJSON AttentionConfig where
  parseJSON = A.withObject "AttentionConfig" $ \o -> do
    cfgNHeads  <- o .: "n_heads"
    cfgDModel  <- o .: "d_model"
    cfgDK      <- o .: "d_k"
    cfgDV      <- o .: "d_v"
    cfgDropout <- o .: "dropout"
    cfgSeqLen  <- o .:? "seq_len"
    return AttentionConfig{..}

-- | Default attention configuration (matches paper)
defaultAttentionConfig :: AttentionConfig
defaultAttentionConfig = AttentionConfig
  { cfgNHeads  = 8
  , cfgDModel  = 512
  , cfgDK      = 64
  , cfgDV      = 64
  , cfgDropout = 0.1
  , cfgSeqLen  = Nothing
  }

-- | Attention operation types
data AttentionOp
  = OpAttentionHead      -- ^ Single attention head (Q,K,V → output)
  | OpMultiHeadAttention -- ^ Multi-head attention (parallel composition)
  | OpTransformerBlock   -- ^ Full transformer block (attention + FFN)
  deriving (Show, Eq)

instance ToJSON AttentionOp where
  toJSON OpAttentionHead      = A.String "attention_head"
  toJSON OpMultiHeadAttention = A.String "multi_head_attention"
  toJSON OpTransformerBlock   = A.String "transformer_block"

instance FromJSON AttentionOp where
  parseJSON = A.withText "AttentionOp" $ \t ->
    case t of
      "attention_head"       -> return OpAttentionHead
      "multi_head_attention" -> return OpMultiHeadAttention
      "transformer_block"    -> return OpTransformerBlock
      _ -> fail $ "Unknown attention operation: " ++ T.unpack t

-- | Attention request to Python
data AttentionRequest = AttentionRequest
  { reqOperation :: AttentionOp
  , reqConfig    :: AttentionConfig
  , reqInputs    :: [Tensor]  -- ^ Input tensor + optional mask
  } deriving (Show, Eq)

instance ToJSON AttentionRequest where
  toJSON AttentionRequest{..} = A.object
    [ "operation" .= reqOperation
    , "config"    .= reqConfig
    , "inputs"    .= reqInputs
    ]

-- | Attention result from Python
data AttentionResult = AttentionResult
  { resOutput          :: Tensor       -- ^ Output tensor
  , resAttentionWeights :: Maybe Tensor -- ^ Attention weights (optional)
  } deriving (Show, Eq)

instance FromJSON AttentionResult where
  parseJSON = A.withObject "AttentionResult" $ \o -> do
    success <- o .: "success"
    if success
      then do
        resOutput          <- o .: "output"
        resAttentionWeights <- o .:? "attention_weights"
        return AttentionResult{..}
      else do
        error_msg <- o .: "error"
        fail $ "Attention operation failed: " ++ T.unpack error_msg

-- | Attention execution errors
data AttentionError
  = AttentionProcessError String
  | AttentionParseError String
  | AttentionExecutionError Text
  deriving (Show, Eq)

instance Exception AttentionError

--------------------------------------------------------------------------------
-- Session Management
--------------------------------------------------------------------------------

-- | Start persistent Python attention session
--
-- Spawns @python3 attention_session.py@ and waits for ready signal.
--
-- Throws 'AttentionProcessError' if subprocess fails to start.
startAttentionSession :: IO AttentionSession
startAttentionSession = do
  -- Get Python executable (from Nix environment or fallback)
  pythonExe <- lookupEnv "ATTENTION_PYTHON"
  let pythonCmd = maybe "python3" id pythonExe

  -- Find attention_session.py script
  let scriptPaths = [ "python-runtime/attention_session.py"
                    , "../python-runtime/attention_session.py"
                    , "../../python-runtime/attention_session.py"
                    ]

  scriptPath <- findScript scriptPaths
  case scriptPath of
    Nothing -> throwIO $ AttentionProcessError
                "Cannot find python-runtime/attention_session.py"
    Just path -> do
      -- Spawn Python subprocess
      (Just attStdin, Just attStdout, Just attStderr, attProc) <-
        createProcess (proc pythonCmd [path])
          { std_in  = CreatePipe
          , std_out = CreatePipe
          , std_err = CreatePipe
          }

      -- Set line buffering
      hSetBuffering attStdin  LineBuffering
      hSetBuffering attStdout LineBuffering

      -- Wait for ready signal: {"status": "ready"}
      readyLine <- hGetLine attStdout `catch` \(e :: IOError) -> do
        stderrOutput <- hGetContents attStderr
        throwIO $ AttentionProcessError $
          "Python attention session failed to start. Stderr: " ++ stderrOutput

      case A.decode (BLC.pack readyLine) of
        Just (A.Object o) | Just (A.String "ready") <- KM.lookup "status" o ->
          return AttentionSession{..}
        _ ->
          throwIO $ AttentionProcessError $
            "Python session didn't send ready signal. Got: " ++ readyLine
  where
    findScript :: [FilePath] -> IO (Maybe FilePath)
    findScript [] = return Nothing
    findScript (p:ps) = do
      exists <- doesFileExist p
      if exists then return (Just p) else findScript ps

-- | Stop attention session gracefully
stopAttentionSession :: AttentionSession -> IO ()
stopAttentionSession AttentionSession{..} = do
  terminateProcess attProc
  _ <- waitForProcess attProc
  return ()

-- | Run action with managed attention session
--
-- Ensures session is properly cleaned up even if action throws exception.
withAttentionSession :: (AttentionSession -> IO a) -> IO a
withAttentionSession = bracket startAttentionSession stopAttentionSession

--------------------------------------------------------------------------------
-- Request/Response
--------------------------------------------------------------------------------

-- | Send JSON request to attention session
sendAttentionRequest :: AttentionSession -> AttentionRequest -> IO ()
sendAttentionRequest AttentionSession{..} req = do
  let jsonLine = A.encode req
  BL.hPutStr attStdin jsonLine
  BLC.hPutStrLn attStdin ""  -- Newline
  hFlush attStdin

-- | Receive JSON response from attention session
receiveAttentionResponse :: AttentionSession -> IO AttentionResult
receiveAttentionResponse AttentionSession{..} = do
  responseLine <- hGetLine attStdout
  case A.decode (BLC.pack responseLine) of
    Just resp -> return resp
    Nothing -> throwIO $ AttentionParseError $
      "Failed to parse attention response: " ++ responseLine

--------------------------------------------------------------------------------
-- High-level API
--------------------------------------------------------------------------------

-- | Execute attention operation via Python/JAX
--
-- Sends operation configuration and inputs to Python session, waits for result.
--
-- Throws:
-- - 'AttentionParseError' if response JSON is malformed
-- - 'AttentionExecutionError' if Python reports error
evalAttention :: AttentionSession
              -> AttentionOp
              -> AttentionConfig
              -> [Tensor]
              -> IO AttentionResult
evalAttention session op config inputs = do
  let req = AttentionRequest op config inputs
  sendAttentionRequest session req
  receiveAttentionResponse session

-- | Execute single attention head
--
-- Input: [batch_size, seq_len, d_model]
-- Output: ([batch_size, seq_len, d_v], [batch_size, seq_len, seq_len])
--
-- Returns output tensor and attention weights.
evalAttentionHead :: AttentionSession
                  -> AttentionConfig
                  -> Tensor
                  -> IO AttentionResult
evalAttentionHead session config input =
  evalAttention session OpAttentionHead config [input]

-- | Execute multi-head attention (parallel composition in 3-category)
--
-- Input: [batch_size, seq_len, d_model]
-- Output: [batch_size, seq_len, d_model]
--
-- Composes n_heads attention heads in parallel (horizontal composition ⊗).
evalMultiHeadAttention :: AttentionSession
                       -> AttentionConfig
                       -> Tensor
                       -> IO AttentionResult
evalMultiHeadAttention session config input =
  evalAttention session OpMultiHeadAttention config [input]

-- | Execute full transformer block (attention + FFN)
--
-- Input: [batch_size, seq_len, d_model]
-- Output: [batch_size, seq_len, d_model]
evalTransformerBlock :: AttentionSession
                     -> AttentionConfig
                     -> Tensor
                     -> IO AttentionResult
evalTransformerBlock session config input =
  evalAttention session OpTransformerBlock config [input]

--------------------------------------------------------------------------------
-- Utilities
--------------------------------------------------------------------------------

-- | Check if attention session is still running
isAttentionSessionAlive :: AttentionSession -> IO Bool
isAttentionSessionAlive AttentionSession{..} = do
  result <- getProcessExitCode attProc
  return $ case result of
    Nothing -> True   -- Still running
    Just _  -> False  -- Exited

--------------------------------------------------------------------------------
-- Example Usage
--------------------------------------------------------------------------------

{-|
Example: Single attention head

>>> session <- startAttentionSession
>>> let config = defaultAttentionConfig { cfgNHeads = 1 }
>>> let input = Tensor [1, 10, 512] (replicate (1*10*512) 0.1)
>>> result <- evalAttentionHead session config input
>>> tensorShape (resOutput result)
[1,10,64]
>>> stopAttentionSession session

Example: Multi-head attention

>>> withAttentionSession $ \session -> do
...   let config = defaultAttentionConfig  -- 8 heads
...   let input = Tensor [2, 5, 512] (replicate (2*5*512) 0.1)
...   result <- evalMultiHeadAttention session config input
...   return (tensorShape (resOutput result))
[2,5,512]

Example: Transformer block

>>> withAttentionSession $ \session -> do
...   let input = Tensor [1, 10, 512] (replicate (1*10*512) 0.1)
...   result <- evalTransformerBlock session defaultAttentionConfig input
...   return (resOutput result)
Tensor {tensorShape = [1,10,512], tensorData = [...]}
-}
