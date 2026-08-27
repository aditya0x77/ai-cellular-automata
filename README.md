# Neural Game of Life

A convolutional neural network that learns to simulate **Conway's Game of Life**.

Instead of applying the handcrafted cellular automata rules, the model predicts the next board state directly from the current board using a convolutional neural network. The learned model is then used to evolve the simulation continuously.

---

## Demo

![Demo](gif.gif)

---

## Features

- Learns Conway's Game of Life from examples rather than explicit rules.
- Predicts the next generation using a convolutional neural network.
- Interactive visualization built with **VisPy**.
- Supports simulations on **1000 × 1000** grids.
- Implemented and trained using **PyTorch**.

---

## How It Works

The simulation begins with a randomly initialized Game of Life board.

For each generation:

1. The current board is passed to the trained CNN.
2. The network predicts the next board state.
3. The output is thresholded to obtain binary alive/dead cells.
4. The predicted board becomes the input for the next iteration.

This process repeats continuously to produce the evolving simulation.

---

## Model Architecture

```text
Input
  │
  ▼
Conv2D (1 → 16)
  │
ReLU
  │
Conv2D (16 → 16)
  │
ReLU
  │
Conv2D (16 → 1)
  │
Sigmoid
  │
Threshold
  ▼
Output
```

The network was trained on board states generated using the original Conway's Game of Life rules, allowing it to learn the underlying transition function directly from data.
