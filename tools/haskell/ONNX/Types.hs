{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ONNX.Types where

import Data.Aeson
import Data.Text (Text)
import GHC.Generics

-- | Mirror Agda ONNX types for JSON deserialization

data TensorElementType
  = UNDEFINED
  | FLOAT
  | UINT8
  | INT8
  | UINT16
  | INT16
  | INT32
  | INT64
  | STRING
  | BOOL
  | FLOAT16
  | DOUBLE
  | UINT32
  | UINT64
  | COMPLEX64
  | COMPLEX128
  | BFLOAT16
  deriving (Show, Eq, Generic)

instance FromJSON TensorElementType
instance ToJSON TensorElementType

data Dimension
  = DimValue Int
  | DimParam Text
  deriving (Show, Eq, Generic)

instance FromJSON Dimension where
  parseJSON = withObject "Dimension" $ \v ->
    (DimValue <$> v .: "dim-value") <|>
    (DimParam <$> v .: "dim-param")

instance ToJSON Dimension where
  toJSON (DimValue n) = object ["dim-value" .= n]
  toJSON (DimParam s) = object ["dim-param" .= s]

type TensorShape = [Dimension]

data TensorTypeProto = TensorTypeProto
  { elemType :: TensorElementType
  , shape :: TensorShape
  } deriving (Show, Eq, Generic)

instance FromJSON TensorTypeProto where
  parseJSON = withObject "TensorTypeProto" $ \v ->
    TensorTypeProto
      <$> v .: "elem-type"
      <*> v .: "shape"

instance ToJSON TensorTypeProto where
  toJSON (TensorTypeProto et s) = object
    [ "elem-type" .= et
    , "shape" .= s
    ]

data TypeProto = TensorType TensorTypeProto
  deriving (Show, Eq, Generic)

instance FromJSON TypeProto where
  parseJSON = withObject "TypeProto" $ \v ->
    TensorType <$> v .: "tensor-type"

instance ToJSON TypeProto where
  toJSON (TensorType tt) = object ["tensor-type" .= tt]

data ValueInfoProto = ValueInfoProto
  { viName :: Text
  , viType :: TypeProto
  , viDoc :: Text
  } deriving (Show, Eq, Generic)

instance FromJSON ValueInfoProto where
  parseJSON = withObject "ValueInfoProto" $ \v ->
    ValueInfoProto
      <$> v .: "name"
      <*> v .: "type"
      <*> v .: "doc"

instance ToJSON ValueInfoProto where
  toJSON (ValueInfoProto n t d) = object
    [ "name" .= n
    , "type" .= t
    , "doc" .= d
    ]

data AttributeValue
  = AttrFloat Double
  | AttrInt Int
  | AttrString Text
  | AttrFloats [Double]
  | AttrInts [Int]
  | AttrStrings [Text]
  deriving (Show, Eq, Generic)

instance FromJSON AttributeValue where
  parseJSON = withObject "AttributeValue" $ \v ->
    (AttrFloat <$> v .: "attr-float") <|>
    (AttrInt <$> v .: "attr-int") <|>
    (AttrString <$> v .: "attr-string") <|>
    (AttrFloats <$> v .: "attr-floats") <|>
    (AttrInts <$> v .: "attr-ints") <|>
    (AttrStrings <$> v .: "attr-strings")

instance ToJSON AttributeValue where
  toJSON (AttrFloat f) = object ["attr-float" .= f]
  toJSON (AttrInt i) = object ["attr-int" .= i]
  toJSON (AttrString s) = object ["attr-string" .= s]
  toJSON (AttrFloats fs) = object ["attr-floats" .= fs]
  toJSON (AttrInts is) = object ["attr-ints" .= is]
  toJSON (AttrStrings ss) = object ["attr-strings" .= ss]

data AttributeProto = AttributeProto
  { attrName :: Text
  , attrValue :: AttributeValue
  } deriving (Show, Eq, Generic)

instance FromJSON AttributeProto where
  parseJSON = withObject "AttributeProto" $ \v ->
    AttributeProto
      <$> v .: "name"
      <*> v .: "value"

instance ToJSON AttributeProto where
  toJSON (AttributeProto n v) = object
    [ "name" .= n
    , "value" .= v
    ]

data NodeProto = NodeProto
  { nodeOpType :: Text
  , nodeInputs :: [Text]
  , nodeOutputs :: [Text]
  , nodeAttributes :: [AttributeProto]
  , nodeName :: Text
  , nodeDomain :: Text
  } deriving (Show, Eq, Generic)

instance FromJSON NodeProto where
  parseJSON = withObject "NodeProto" $ \v ->
    NodeProto
      <$> v .: "op-type"
      <*> v .: "inputs"
      <*> v .: "outputs"
      <*> v .: "attributes"
      <*> v .: "name"
      <*> v .: "domain"

instance ToJSON NodeProto where
  toJSON (NodeProto ot ins outs attrs n d) = object
    [ "op-type" .= ot
    , "inputs" .= ins
    , "outputs" .= outs
    , "attributes" .= attrs
    , "name" .= n
    , "domain" .= d
    ]

data TensorProto = TensorProto
  { tensorName :: Text
  , tensorElemType :: TensorElementType
  , tensorDims :: [Int]
  } deriving (Show, Eq, Generic)

instance FromJSON TensorProto where
  parseJSON = withObject "TensorProto" $ \v ->
    TensorProto
      <$> v .: "name"
      <*> v .: "elem-type"
      <*> v .: "dims"

instance ToJSON TensorProto where
  toJSON (TensorProto n et dims) = object
    [ "name" .= n
    , "elem-type" .= et
    , "dims" .= dims
    ]

data GraphProto = GraphProto
  { graphNodes :: [NodeProto]
  , graphName :: Text
  , graphInputs :: [ValueInfoProto]
  , graphOutputs :: [ValueInfoProto]
  , graphInitializers :: [TensorProto]
  , graphDoc :: Text
  } deriving (Show, Eq, Generic)

instance FromJSON GraphProto where
  parseJSON = withObject "GraphProto" $ \v ->
    GraphProto
      <$> v .: "nodes"
      <*> v .: "name"
      <*> v .: "inputs"
      <*> v .: "outputs"
      <*> v .: "initializers"
      <*> v .: "doc"

instance ToJSON GraphProto where
  toJSON (GraphProto nodes n ins outs inits d) = object
    [ "nodes" .= nodes
    , "name" .= n
    , "inputs" .= ins
    , "outputs" .= outs
    , "initializers" .= inits
    , "doc" .= d
    ]

data OperatorSetIdProto = OperatorSetIdProto
  { opsetDomain :: Text
  , opsetVersion :: Int
  } deriving (Show, Eq, Generic)

instance FromJSON OperatorSetIdProto where
  parseJSON = withObject "OperatorSetIdProto" $ \v ->
    OperatorSetIdProto
      <$> v .: "domain"
      <*> v .: "version"

instance ToJSON OperatorSetIdProto where
  toJSON (OperatorSetIdProto d v) = object
    [ "domain" .= d
    , "version" .= v
    ]

data ModelProto = ModelProto
  { modelIrVersion :: Int
  , modelOpsetImport :: [OperatorSetIdProto]
  , modelProducerName :: Text
  , modelProducerVersion :: Text
  , modelDomain :: Text
  , modelVersion :: Int
  , modelDoc :: Text
  , modelGraph :: GraphProto
  } deriving (Show, Eq, Generic)

instance FromJSON ModelProto where
  parseJSON = withObject "ModelProto" $ \v ->
    ModelProto
      <$> v .: "ir-version"
      <*> v .: "opset-import"
      <*> v .: "producer-name"
      <*> v .: "producer-version"
      <*> v .: "domain"
      <*> v .: "model-version"
      <*> v .: "doc"
      <*> v .: "graph"

instance ToJSON ModelProto where
  toJSON (ModelProto ir opset pn pv d mv doc g) = object
    [ "ir-version" .= ir
    , "opset-import" .= opset
    , "producer-name" .= pn
    , "producer-version" .= pv
    , "domain" .= d
    , "model-version" .= mv
    , "doc" .= doc
    , "graph" .= g
    ]
