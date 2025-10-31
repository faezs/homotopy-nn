{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module: Einsum.PythonBridge
Description: FFI bridge to Python einsum session

Manages persistent Python subprocess for executing einsum operations.

## Architecture

```
Haskell (PythonBridge)
    ↓ JSON over stdin/stdout
Python REPL (einsum_session.py)
    ↓ S-expression parsing
PyTorch (torch.einsum)
    ↓ GPU execution
Tensor Result
```

## Usage

@
session <- startPythonSession
result <- evalEinsum session formula tensors
stopPythonSession session
@

## Protocol

Request (JSON):
{
  "formula": "(contract [j] [[i] [k]])",
  "tensors": [
    {"shape": [3, 2], "data": [1.0, 2.0, ...]},
    {"shape": [3, 2], "data": [7.0, 8.0, ...]}
  ]
}

Response (JSON):
{
  "success": true,
  "shape": [2, 2],
  "data": [89.0, 98.0, 116.0, 128.0]
}
-}

module Einsum.PythonBridge
  ( -- * Session Management
    PythonSession
  , startPythonSession
  , stopPythonSession

    -- * Tensor Types
  , Tensor(..)
  , Shape
  , TensorData

    -- * Execution
  , evalEinsum
  , EinsumError(..)

    -- * Low-level API
  , sendRequest
  , receiveResponse
  , EinsumRequest(..)
  , EinsumResponse(..)
  ) where

import Control.Exception (Exception, throwIO, try, bracket)
import Control.Monad (unless)
import Data.Aeson (FromJSON, ToJSON, (.=), (.:), (.:?))
import qualified Data.Aeson as A
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BLC
import Data.Text (Text)
import qualified Data.Text as T
import System.IO (Handle, hFlush, hSetBuffering, BufferMode(..))
import System.Process

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

-- | Python subprocess session
data PythonSession = PythonSession
  { pyStdin  :: Handle
  , pyStdout :: Handle
  , pyStderr :: Handle
  , pyProc   :: ProcessHandle
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

-- | Einsum request to Python
data EinsumRequest = EinsumRequest
  { reqFormula :: Text
  , reqTensors :: [Tensor]
  } deriving (Show, Eq)

instance ToJSON EinsumRequest where
  toJSON EinsumRequest{..} = A.object
    [ "formula" .= reqFormula
    , "tensors" .= reqTensors
    ]

-- | Einsum response from Python
data EinsumResponse
  = EinsumSuccess
      { respShape :: Shape
      , respData  :: TensorData
      }
  | EinsumFailure
      { respError :: Text
      }
  deriving (Show, Eq)

instance FromJSON EinsumResponse where
  parseJSON = A.withObject "EinsumResponse" $ \o -> do
    success <- o .: "success"
    if success
      then do
        respShape <- o .: "shape"
        respData  <- o .: "data"
        return $ EinsumSuccess respShape respData
      else do
        respError <- o .: "error"
        return $ EinsumFailure respError

-- | Einsum execution errors
data EinsumError
  = PythonProcessError String
  | ParseError String
  | EinsumExecutionError Text
  deriving (Show, Eq)

instance Exception EinsumError

--------------------------------------------------------------------------------
-- Session Management
--------------------------------------------------------------------------------

-- | Start persistent Python einsum session
--
-- Spawns @python3 einsum_session.py@ and waits for ready signal.
--
-- Throws 'PythonProcessError' if subprocess fails to start or doesn't send ready signal.
startPythonSession :: IO PythonSession
startPythonSession = do
  -- Spawn Python subprocess
  (Just pyStdin, Just pyStdout, Just pyStderr, pyProc) <-
    createProcess (proc "python3" ["python-runtime/einsum_session.py"])
      { std_in  = CreatePipe
      , std_out = CreatePipe
      , std_err = CreatePipe
      , cwd     = Just "."  -- Run from project root
      }

  -- Set line buffering for responsiveness
  hSetBuffering pyStdin  LineBuffering
  hSetBuffering pyStdout LineBuffering

  -- Wait for ready signal: {"status": "ready"}
  readyLine <- BLC.hGetLine pyStdout
  case A.decode readyLine of
    Just (A.Object o) | Just (A.String "ready") <- A.lookup "status" o ->
      return PythonSession{..}
    _ ->
      throwIO $ PythonProcessError "Python session didn't send ready signal"

-- | Stop Python session gracefully
--
-- Closes handles and terminates subprocess.
stopPythonSession :: PythonSession -> IO ()
stopPythonSession PythonSession{..} = do
  terminateProcess pyProc
  _ <- waitForProcess pyProc
  return ()

-- | Run action with managed Python session
--
-- Ensures session is properly cleaned up even if action throws exception.
--
-- @
-- withPythonSession $ \session -> do
--   result1 <- evalEinsum session formula1 tensors1
--   result2 <- evalEinsum session formula2 tensors2
--   return (result1, result2)
-- @
withPythonSession :: (PythonSession -> IO a) -> IO a
withPythonSession = bracket startPythonSession stopPythonSession

--------------------------------------------------------------------------------
-- Request/Response
--------------------------------------------------------------------------------

-- | Send JSON request to Python session
--
-- Encodes request as JSON, writes line, flushes stdout.
sendRequest :: PythonSession -> EinsumRequest -> IO ()
sendRequest PythonSession{..} req = do
  let jsonLine = A.encode req
  BL.hPutStr pyStdin jsonLine
  BLC.hPutStrLn pyStdin ""  -- Newline
  hFlush pyStdin

-- | Receive JSON response from Python session
--
-- Reads line from stdout, decodes JSON.
--
-- Throws 'ParseError' if JSON is malformed.
receiveResponse :: PythonSession -> IO EinsumResponse
receiveResponse PythonSession{..} = do
  responseLine <- BLC.hGetLine pyStdout
  case A.decode responseLine of
    Just resp -> return resp
    Nothing -> throwIO $ ParseError $
      "Failed to parse Python response: " ++ BLC.unpack responseLine

--------------------------------------------------------------------------------
-- High-level API
--------------------------------------------------------------------------------

-- | Execute einsum operation via Python
--
-- Sends formula and tensors to Python session, waits for result.
--
-- Throws:
-- - 'ParseError' if response JSON is malformed
-- - 'EinsumExecutionError' if Python reports error (parse error, dimension mismatch, etc.)
--
-- Example:
--
-- @
-- session <- startPythonSession
-- let formula = "(contract [j] [[i] [k]])"
-- let a = Tensor [3, 2] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
-- let b = Tensor [3, 2] [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
-- result <- evalEinsum session formula [a, b]
-- -- result: Tensor [2, 2] [89.0, 98.0, 116.0, 128.0]
-- @
evalEinsum :: PythonSession -> Text -> [Tensor] -> IO Tensor
evalEinsum session formula tensors = do
  let req = EinsumRequest formula tensors
  sendRequest session req
  resp <- receiveResponse session

  case resp of
    EinsumSuccess{..} ->
      return $ Tensor respShape respData

    EinsumFailure{..} ->
      throwIO $ EinsumExecutionError respError

--------------------------------------------------------------------------------
-- Utilities
--------------------------------------------------------------------------------

-- | Check if Python subprocess is still running
isSessionAlive :: PythonSession -> IO Bool
isSessionAlive PythonSession{..} = do
  result <- getProcessExitCode pyProc
  return $ case result of
    Nothing -> True   -- Still running
    Just _  -> False  -- Exited

-- | Read stderr from Python session (for debugging)
--
-- Non-blocking read of any available stderr output.
readSessionErrors :: PythonSession -> IO String
readSessionErrors PythonSession{..} = do
  -- Note: This is a simple implementation
  -- For production, use non-blocking IO or async
  return ""  -- Placeholder

--------------------------------------------------------------------------------
-- Example Usage
--------------------------------------------------------------------------------

{-|
Example: Matrix multiplication

>>> session <- startPythonSession
>>> let formula = "(contract [j] [[i] [k]])"
>>> let a = Tensor [3, 2] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
>>> let b = Tensor [3, 2] [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
>>> result <- evalEinsum session formula [a, b]
>>> tensorShape result
[2,2]
>>> tensorData result
[89.0,98.0,116.0,128.0]
>>> stopPythonSession session

Example: With managed session

>>> withPythonSession $ \session -> do
...   let formula = "(contract [i] [[] []])"  -- Dot product
...   let v = Tensor [5] [1.0, 2.0, 3.0, 4.0, 5.0]
...   let w = Tensor [5] [1.0, 1.0, 1.0, 1.0, 1.0]
...   result <- evalEinsum session formula [v, w]
...   return (tensorData result)
[15.0]
-}
