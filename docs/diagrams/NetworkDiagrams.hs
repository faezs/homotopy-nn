{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

{-|
Module: NetworkDiagrams
Description: Generate diagrams for neural network architectures using Diagrams library

Generates SVG diagrams for:
1. SimpleMLP: Chain network (x₀ → h₁ → h₂ → y)
2. ConvergentNetwork: ResNet-like with fork (x₀,₁ → h ← x₀,₂)
3. ComplexNetwork: Multi-path with multiple convergence points
-}

module NetworkDiagrams where

import Diagrams.Prelude
import Diagrams.Backend.SVG.CmdLine
import Data.Colour.Palette.BrewerSet

-- Colors
nodeColor :: Colour Double
nodeColor = sRGB24 102 102 255  -- Blue

edgeColor :: Colour Double
edgeColor = sRGB24 26 26 26     -- Near-black

forkStarColor :: Colour Double
forkStarColor = sRGB24 255 153 51  -- Orange (for fork stars)

forkTangColor :: Colour Double
forkTangColor = sRGB24 153 204 255  -- Light blue (for fork tangs)

-- Basic building blocks
vertex :: String -> Diagram B
vertex label = text label # fontSize (local 0.3) # fc black
            <> circle 0.4 # fc nodeColor # lw 0.05 # lc edgeColor

forkStar :: String -> Diagram B
forkStar label = text label # fontSize (local 0.25) # fc black
              <> square 0.7 # fc forkStarColor # lw 0.05 # lc edgeColor

forkTang :: String -> Diagram B
forkTang label = text label # fontSize (local 0.25) # fc black
              <> circle 0.4 # fc forkTangColor # lw 0.05 # lc edgeColor

-- Arrow with label
labeledArrow :: P2 Double -> P2 Double -> Diagram B
labeledArrow start end =
    arrowBetween start end # lw 0.08 # lc edgeColor

-- 1. Simple MLP: Chain network
-- x₀ → h₁ → h₂ → y
simpleMLP :: Diagram B
simpleMLP = positioned
    [ (p2 (0, 0), vertex "x₀")
    , (p2 (2, 0), vertex "h₁")
    , (p2 (4, 0), vertex "h₂")
    , (p2 (6, 0), vertex "y")
    ]
    <>
    mconcat
    [ labeledArrow (p2 (0.4, 0)) (p2 (1.6, 0))   -- x₀ → h₁
    , labeledArrow (p2 (2.4, 0)) (p2 (3.6, 0))   -- h₁ → h₂
    , labeledArrow (p2 (4.4, 0)) (p2 (5.6, 0))   -- h₂ → y
    ]
    # centerXY
    # pad 1.2

-- MLP with labels
mlpLabeled :: Diagram B
mlpLabeled = simpleMLP
          <> positioned [(p2 (0, -1.5), mlpLabel)]
    where
      mlpLabel = text "Poset X: y ≤ h₂ ≤ h₁ ≤ x₀"
               # fontSize (local 0.25) # fc black

-- 2. Convergent Network: ResNet-like
-- Two branches converging at hidden layer
--   x₀,₁    x₀,₂
--     ↓      ↓
--     └──→ h ←──┘
--          ↓
--          y
convergentNetwork :: Diagram B
convergentNetwork = positioned
    [ (p2 (-1, 2), vertex "x₀,₁")
    , (p2 (1, 2), vertex "x₀,₂")
    , (p2 (0, 0), vertex "h")
    , (p2 (0, -2), vertex "y")
    ]
    <>
    mconcat
    [ labeledArrow (p2 (-0.8, 1.6)) (p2 (-0.2, 0.4))  -- x₀,₁ → h
    , labeledArrow (p2 (0.8, 1.6)) (p2 (0.2, 0.4))    -- x₀,₂ → h
    , labeledArrow (p2 (0, -0.4)) (p2 (0, -1.6))      -- h → y
    ]
    # centerXY
    # pad 1.2

-- 2b. Convergent Network with Fork Construction
-- Shows the fork star (A★) and tang (A)
convergentWithFork :: Diagram B
convergentWithFork = positioned
    [ (p2 (-2, 3), vertex "x₀,₁")
    , (p2 (2, 3), vertex "x₀,₂")
    , (p2 (0, 1.5), forkStar "A★")
    , (p2 (0, 0), forkTang "A")
    , (p2 (0, -1.5), vertex "h")
    , (p2 (0, -3), vertex "y")
    ]
    <>
    mconcat
    [ labeledArrow (p2 (-1.6, 2.7)) (p2 (-0.3, 1.7))  -- x₀,₁ → A★
    , labeledArrow (p2 (1.6, 2.7)) (p2 (0.3, 1.7))    -- x₀,₂ → A★
    , labeledArrow (p2 (0, 1.15)) (p2 (0, 0.4))       -- A★ → A
    , labeledArrow (p2 (0, -0.4)) (p2 (0, -1.1))      -- A → h
    , labeledArrow (p2 (0, -1.9)) (p2 (0, -2.6))      -- h → y
    ]
    <>
    positioned
    [ (p2 (-3.5, 1.5), forkLabel1)
    , (p2 (-3.5, 0.8), forkLabel2)
    ]
    # centerXY
    # pad 1.2
    where
      forkLabel1 = text "Fork star A★"
                 # fontSize (local 0.2) # fc (sRGB24 255 100 0)
      forkLabel2 = text "Fork tang A"
                 # fontSize (local 0.2) # fc (sRGB24 100 150 255)

-- 2c. Poset X after fork (removed A★, reversed arrows)
convergentPoset :: Diagram B
convergentPoset = positioned
    [ (p2 (-1.5, 2), vertex "x₀,₁")
    , (p2 (1.5, 2), vertex "x₀,₂")
    , (p2 (0, 0.5), forkTang "A")
    , (p2 (0, -1), vertex "h")
    , (p2 (0, -2.5), vertex "y")
    ]
    <>
    mconcat
    [ labeledArrow (p2 (-0.3, 0.8)) (p2 (-1.2, 1.6))  -- A → x₀,₁ (reversed!)
    , labeledArrow (p2 (0.3, 0.8)) (p2 (1.2, 1.6))    -- A → x₀,₂ (reversed!)
    , labeledArrow (p2 (0, 0.1)) (p2 (0, -0.6))       -- A → h (reversed!)
    , labeledArrow (p2 (0, -1.4)) (p2 (0, -2.1))      -- h → y (reversed!)
    ]
    <>
    positioned
    [ (p2 (3, 0.5), posetLabel1)
    , (p2 (3, 0), posetLabel2)
    , (p2 (3, -0.5), posetLabel3)
    ]
    # centerXY
    # pad 1.2
    where
      posetLabel1 = text "Poset X:" # fontSize (local 0.2) # fc black
      posetLabel2 = text "y ≤ h ≤ A ≤ x₀,₁" # fontSize (local 0.18) # fc black
      posetLabel3 = text "y ≤ h ≤ A ≤ x₀,₂" # fontSize (local 0.18) # fc black

-- 3. Complex Multi-Path Network
-- Multiple convergence points with multiple forks
complexNetwork :: Diagram B
complexNetwork = positioned
    [ (p2 (-3, 4), vertex "x₀,₁")
    , (p2 (3, 4), vertex "x₀,₂")
    , (p2 (-2, 2), vertex "c₁")
    , (p2 (2, 2), vertex "c₂")
    , (p2 (0, 0), vertex "c₃")
    , (p2 (-3, -2), vertex "xₙ,₁")
    , (p2 (-1.5, -2), vertex "xₙ,₂")
    , (p2 (0, -2), vertex "xₙ,₃")
    , (p2 (1.5, -2), vertex "xₙ,₄")
    , (p2 (3, -2), vertex "xₙ,₅")
    ]
    <>
    mconcat
    [ labeledArrow (p2 (-2.8, 3.6)) (p2 (-2.2, 2.4))  -- x₀,₁ → c₁
    , labeledArrow (p2 (2.8, 3.6)) (p2 (2.2, 2.4))    -- x₀,₂ → c₂
    , labeledArrow (p2 (-1.6, 1.8)) (p2 (-0.3, 0.3))  -- c₁ → c₃
    , labeledArrow (p2 (1.6, 1.8)) (p2 (0.3, 0.3))    -- c₂ → c₃
    , labeledArrow (p2 (2, 1.6)) (p2 (-1.8, 2.2))     -- c₂ → c₁ (skip)
    , labeledArrow (p2 (-0.5, -0.3)) (p2 (-2.7, -1.7)) -- c₃ → xₙ,₁
    , labeledArrow (p2 (-0.3, -0.3)) (p2 (-1.4, -1.7)) -- c₃ → xₙ,₂
    , labeledArrow (p2 (0, -0.4)) (p2 (0, -1.6))       -- c₃ → xₙ,₃
    , labeledArrow (p2 (0.3, -0.3)) (p2 (1.4, -1.7))   -- c₃ → xₙ,₄
    , labeledArrow (p2 (0.5, -0.3)) (p2 (2.7, -1.7))   -- c₃ → xₙ,₅
    ]
    # centerXY
    # pad 1.2

-- All diagrams combined for export
allDiagrams :: [(String, Diagram B)]
allDiagrams =
    [ ("mlp", mlpLabeled)
    , ("convergent", convergentNetwork)
    , ("convergent_fork", convergentWithFork)
    , ("convergent_poset", convergentPoset)
    , ("complex", complexNetwork)
    ]

main :: IO ()
main = do
    -- Generate all diagrams
    mapM_ generateDiagram allDiagrams
    putStrLn "Generated all network diagrams in docs/images/"
  where
    generateDiagram (name, dia) = do
        let opts = SVGOptions
                   { _size = mkWidth 800
                   , _svgDefinitions = Nothing
                   , _idPrefix = T.pack name
                   , _svgAttributes = []
                   , _generateDoctype = True
                   }
        renderSVG ("../images/" ++ name ++ ".svg") opts dia
        putStrLn $ "Generated: " ++ name ++ ".svg"
