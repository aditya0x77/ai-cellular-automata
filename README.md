# AI Game of Life

A neural network that learns to simulate Conway's Game of Life.

Instead of applying the handcrafted cellular automata rules every generation, this project trains a convolutional neural network to predict the next board state directly from the current one.

The result is a real-time AI-powered simulation accelerated with PyTorch and visualized using VisPy.

---

## Features

- CNN predicts the next generation of the Game of Life
- GPU acceleration with PyTorch
- Real-time visualization using VisPy
- Large simulation grids (1000×1000)
- Demonstrates how neural networks can approximate deterministic algorithms

---

## Demo

Add a GIF here:

![Demo](assets/demo.gif)

---

## How it Works

1. A random Game of Life board is generated.
2. The pretrained CNN receives the current board as input.
3. The model predicts the next generation.
4. The output is thresholded into alive/dead cells.
5. The process repeats continuously.

Unlike the classic implementation, the evolution is performed entirely through neural network inference.

---

## Model

Architecture:

- Conv2D (1 → 16)
- ReLU
- Conv2D (16 → 16)
- ReLU
- Conv2D (16 → 1)
- Sigmoid
- Threshold (0.5)

The network was trained on examples generated using the original Conway's Game of Life rules.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-game-of-life.git

cd ai-game-of-life
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python game_of_life.py
```

If CUDA is available, the model automatically runs on the GPU.

---

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- VisPy

Example:

```bash
pip install torch numpy vispy
```

---

## Repository

```
.
├── game_of_life.py
├── game_of_life_cnn.pth
├── notebooks/
├── assets/
└── README.md
```

---

## Future Improvements

- Train deeper CNN architectures
- Compare AI accuracy against the exact Game of Life rules
- Export simulations as GIFs
- Interactive drawing mode
- Performance benchmarking (CPU vs GPU)
- Larger training datasets

---

## Inspiration

Conway's Game of Life is a classic cellular automaton where simple local rules generate surprisingly complex behavior.

This project explores whether a convolutional neural network can learn those rules purely from data.

---

## License

MIT License
