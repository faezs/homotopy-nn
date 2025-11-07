{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module MLP.Compiler
  ( -- * MLP Compilation Types
    LayerSpec(..)
  , Connection(..)
  , JAXNetworkSpec(..)

  -- * Compilation Functions
  , mlpCompileFFI
  , networkToJsonFFI

  -- * Session Management
  , MLPSession
  , withMLPSession
  , evalMLP

  -- * JSON Conversion
  , specToJSON
  , specToPython
  ) where

import qualified Data.Aeson as A
import Data.Aeson ((.=), (.:))
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Lazy.Char8 as BSL8
import System.Process
import System.IO
import GHC.Generics (Generic)
import Control.Exception (bracket)

--------------------------------------------------------------------------------
-- § 1: MLP Layer Specification (matching Agda types)
--------------------------------------------------------------------------------

-- | Layer specification matching Agda LayerSpec
data LayerSpec
  = InputLayer { layerDim :: Int }
  | HiddenLayer { layerWidth :: Int
                , layerInitStd :: Double
                , layerLR :: Double
                }
  | OutputLayer { layerDim :: Int
                , layerInitStd :: Double
                , layerLR :: Double
                }
  deriving (Show, Eq, Generic)

instance A.ToJSON LayerSpec where
  toJSON (InputLayer dim) = A.object
    [ "type" .= ("input" :: Text)
    , "dim" .= dim
    ]
  toJSON (HiddenLayer width initStd lr) = A.object
    [ "type" .= ("hidden" :: Text)
    , "width" .= width
    , "init_std" .= initStd
    , "lr" .= lr
    ]
  toJSON (OutputLayer dim initStd lr) = A.object
    [ "type" .= ("output" :: Text)
    , "dim" .= dim
    , "init_std" .= initStd
    , "lr" .= lr
    ]

instance A.FromJSON LayerSpec where
  parseJSON = A.withObject "LayerSpec" $ \v -> do
    layerType <- v .: "type" :: A.Parser Text
    case layerType of
      "input" -> InputLayer <$> v .: "dim"
      "hidden" -> HiddenLayer <$> v .: "width" <*> v .: "init_std" <*> v .: "lr"
      "output" -> OutputLayer <$> v .: "dim" <*> v .: "init_std" <*> v .: "lr"
      _ -> fail $ "Unknown layer type: " ++ T.unpack layerType

-- | Connection between layers (edge in chain)
data Connection = Connection
  { connFrom :: Int  -- ^ Source layer index
  , connTo   :: Int  -- ^ Target layer index
  } deriving (Show, Eq, Generic)

instance A.ToJSON Connection where
  toJSON Connection{..} = A.object
    [ "from" .= connFrom
    , "to" .= connTo
    ]

instance A.FromJSON Connection where
  parseJSON = A.withObject "Connection" $ \v ->
    Connection <$> v .: "from" <*> v .: "to"

-- | Complete network specification for JAX
data JAXNetworkSpec = JAXNetworkSpec
  { netLayers :: [LayerSpec]
  , netConnections :: [Connection]
  , netBatchSize :: Int
  , netNumEpochs :: Int
  , netOptimizer :: Text
  } deriving (Show, Eq, Generic)

instance A.ToJSON JAXNetworkSpec where
  toJSON JAXNetworkSpec{..} = A.object
    [ "layers" .= netLayers
    , "connections" .= netConnections
    , "batch_size" .= netBatchSize
    , "num_epochs" .= netNumEpochs
    , "optimizer" .= netOptimizer
    ]

instance A.FromJSON JAXNetworkSpec where
  parseJSON = A.withObject "JAXNetworkSpec" $ \v ->
    JAXNetworkSpec
      <$> v .: "layers"
      <*> v .: "connections"
      <*> v .: "batch_size"
      <*> v .: "num_epochs"
      <*> v .: "optimizer"

--------------------------------------------------------------------------------
-- § 2: FFI Functions (called from Agda)
--------------------------------------------------------------------------------

-- | Stub for Agda FFI: compile MLP at specific width
-- In real implementation, this would receive Agda MUPBase and width
mlpCompileFFI :: Int -> JAXNetworkSpec
mlpCompileFFI width = JAXNetworkSpec
  { netLayers =
      [ InputLayer 784  -- MNIST input
      , HiddenLayer width (0.02 / sqrt (fromIntegral width)) 0.1
      , HiddenLayer width (0.02 / sqrt (fromIntegral width)) 0.1
      , OutputLayer 10 (0.02 / fromIntegral width) (0.1 / fromIntegral width)
      ]
  , netConnections =
      [ Connection 0 1  -- input → h₁
      , Connection 1 2  -- h₁ → h₂
      , Connection 2 3  -- h₂ → output
      ]
  , netBatchSize = 32
  , netNumEpochs = 100
  , netOptimizer = "adam"
  }

-- | Convert network spec to JSON (for Agda FFI)
networkToJsonFFI :: JAXNetworkSpec -> Text
networkToJsonFFI = T.decodeUtf8 . BSL.toStrict . A.encode

-- | Convert JSON to network spec
specFromJSON :: BSL.ByteString -> Either String JAXNetworkSpec
specFromJSON = A.eitherDecode

-- | Convert network spec to JSON ByteString
specToJSON :: JAXNetworkSpec -> BSL.ByteString
specToJSON = A.encode

--------------------------------------------------------------------------------
-- § 3: Python Session Management (JSON Protocol)
--------------------------------------------------------------------------------

-- | MLP execution session (Python subprocess)
data MLPSession = MLPSession
  { mlpStdin  :: Handle
  , mlpStdout :: Handle
  , mlpStderr :: Handle
  , mlpProc   :: ProcessHandle
  }

-- | Start Python MLP session
startMLPSession :: IO MLPSession
startMLPSession = do
  let pythonRuntime = "python-runtime/mlp_mup_runner.py"
  (Just stdin, Just stdout, Just stderr, proc) <- createProcess $
    (proc "python3" [pythonRuntime])
      { std_in  = CreatePipe
      , std_out = CreatePipe
      , std_err = CreatePipe
      }

  -- Configure handles
  hSetBuffering stdin LineBuffering
  hSetBuffering stdout LineBuffering
  hSetBinaryMode stdin False
  hSetBinaryMode stdout False

  return $ MLPSession stdin stdout stderr proc

-- | Stop Python MLP session
stopMLPSession :: MLPSession -> IO ()
stopMLPSession MLPSession{..} = do
  hClose mlpStdin
  hClose mlpStdout
  hClose mlpStderr
  _ <- waitForProcess mlpProc
  return ()

-- | Resource-safe session wrapper
withMLPSession :: (MLPSession -> IO a) -> IO a
withMLPSession = bracket startMLPSession stopMLPSession

--------------------------------------------------------------------------------
-- § 4: MLP Execution via JSON Protocol
--------------------------------------------------------------------------------

-- | Request to Python runtime
data MLPRequest = MLPRequest
  { reqCommand :: Text
  , reqNetwork :: JAXNetworkSpec
  } deriving (Show, Eq, Generic)

instance A.ToJSON MLPRequest where
  toJSON MLPRequest{..} = A.object
    [ "command" .= reqCommand
    , "network" .= reqNetwork
    ]

-- | Response from Python runtime
data MLPResponse = MLPResponse
  { respStatus :: Text
  , respLoss :: Maybe Double
  , respAccuracy :: Maybe Double
  , respFeatureNorms :: Maybe A.Object
  , respError :: Maybe Text
  } deriving (Show, Eq, Generic)

instance A.FromJSON MLPResponse where
  parseJSON = A.withObject "MLPResponse" $ \v ->
    MLPResponse
      <$> v .: "status"
      <*> v .: "loss"
      <*> v .: "accuracy"
      <*> v .: "feature_norms"
      <*> v .: "error"

-- | Train MLP and return results
evalMLP :: MLPSession -> JAXNetworkSpec -> IO (Either String MLPResponse)
evalMLP MLPSession{..} spec = do
  -- Send request
  let request = MLPRequest "train" spec
  BSL8.hPutStrLn mlpStdin (A.encode request)

  -- Read response
  responseLine <- TIO.hGetLine mlpStdout
  case A.eitherDecodeStrict (T.encodeUtf8 responseLine) of
    Left err -> return $ Left $ "JSON parse error: " ++ err
    Right resp -> return $ Right resp

--------------------------------------------------------------------------------
-- § 5: Python Code Generation (for testing)
--------------------------------------------------------------------------------

-- | Generate Python code for network spec
specToPython :: JAXNetworkSpec -> Text
specToPython JAXNetworkSpec{..} = T.unlines
  [ "# Generated MLP network with MUP scaling"
  , "import jax"
  , "import jax.numpy as jnp"
  , "import flax.linen as nn"
  , ""
  , "class MUPMLP(nn.Module):"
  , "    @nn.compact"
  , "    def __call__(self, x):"
  , T.unlines (map layerToPython netLayers)
  , "        return x"
  , ""
  , "# Training config:"
  , "batch_size = " <> T.pack (show netBatchSize)
  , "num_epochs = " <> T.pack (show netNumEpochs)
  , "optimizer = '" <> netOptimizer <> "'"
  ]
  where
    layerToPython :: LayerSpec -> Text
    layerToPython (InputLayer _) = "        # Input layer (flatten)"
    layerToPython (HiddenLayer width initStd lr) = T.unlines
      [ "        x = nn.Dense(" <> T.pack (show width)
          <> ", kernel_init=nn.initializers.normal(" <> T.pack (show initStd) <> "))(x)"
      , "        x = nn.relu(x)"
      ]
    layerToPython (OutputLayer dim initStd _) = T.unlines
      [ "        x = nn.Dense(" <> T.pack (show dim)
          <> ", kernel_init=nn.initializers.normal(" <> T.pack (show initStd) <> "))(x)"
      ]

--------------------------------------------------------------------------------
-- § 6: Example Usage
--------------------------------------------------------------------------------

-- | Example: Compile networks at different widths for MUP transfer test
exampleMUPTransfer :: IO ()
exampleMUPTransfer = do
  let widths = [64, 256, 1024]
  let networks = map mlpCompileFFI widths

  putStrLn "=== MUP Transfer Test Configuration ==="
  mapM_ (\(w, net) -> do
    putStrLn $ "\nWidth " ++ show w ++ ":"
    BSL8.putStrLn $ A.encode net
    ) (zip widths networks)

  putStrLn "\n=== Training networks ==="
  withMLPSession $ \session -> do
    results <- mapM (evalMLP session) networks
    mapM_ (\(w, result) -> do
      putStrLn $ "\nWidth " ++ show w ++ ":"
      case result of
        Left err -> putStrLn $ "Error: " ++ err
        Right resp -> do
          putStrLn $ "Status: " ++ T.unpack (respStatus resp)
          putStrLn $ "Loss: " ++ show (respLoss resp)
          putStrLn $ "Accuracy: " ++ show (respAccuracy resp)
      ) (zip widths results)
