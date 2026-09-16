# Mini PyTorch — Custom Autograd Engine

A lightweight **from-scratch implementation of automatic differentiation and neural networks in Python**, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

The project builds the core ideas behind PyTorch's autograd system without relying on PyTorch itself. It implements a scalar computational graph, reverse-mode automatic differentiation, trainable neurons, fully connected layers, and a Multi-Layer Perceptron (MLP).

The implementation is then tested on **XOR**, **Synthetic Moons**, and **Iris** classification datasets.

---

## 🚀 Project Overview

Modern deep-learning frameworks such as PyTorch hide a lot of important mechanics behind simple APIs like:

```python
loss.backward()
optimizer.step()
```

This project goes underneath that abstraction and demonstrates how those mechanisms can be implemented from scratch.

The main components are:

```text
Scalar Value
     │
     ▼
Computational Graph
     │
     ▼
Reverse-Mode Autograd
     │
     ▼
Neuron
     │
     ▼
Fully Connected Layer
     │
     ▼
MLP
     │
     ▼
Forward Pass → Loss → Backward Pass → Parameter Update
```

The project is primarily intended as a learning implementation for understanding:

* Computational graphs
* Automatic differentiation
* Backpropagation
* Chain rule
* Gradients
* Neural-network architecture
* Parameter updates
* Activation functions
* Decision boundaries

---

## ✨ Features

### Custom Scalar Autograd

The `Params` class represents a scalar value while keeping track of its computational history.

Each node stores:

* `data` — forward-pass value
* `grad` — accumulated gradient
* `op` — operation that created the node
* `label` — human-readable node name
* `_children` — parent/previous nodes in the computation graph

This allows mathematical operations to automatically construct a computational graph.

---

### Supported Operations

The custom autograd engine supports:

```python
+
-
*
/
**
```

including reverse operations such as:

```python
__radd__
__rmul__
__rsub__
__rtruediv__
```

Raw Python numbers are automatically converted into `Params` objects so that operations can be handled consistently.

---

## 🧠 Activation Functions

The implementation includes three common activation functions:

### ReLU

```python
x.relu()
```

Mathematically:

$$
ReLU(x) = max(0,x)
$$

Derivative:

$$
\frac{d}{dx}ReLU(x)=
\begin{cases}
1 & x>0 \\
0 & x\leq0
\end{cases}
$$

### Tanh

```python
x.tanh()
```

Output range:

$$
(-1,1)
$$

Derivative:

$$
\frac{d}{dx}\tanh(x)=1-\tanh^2(x)
$$

### Sigmoid

```python
x.sigmoid()
```

Output range:

$$
(0,1)
$$

Derivative:

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

These activation functions and their derivatives are implemented directly inside the custom autograd engine.

---

# 🔄 Reverse-Mode Automatic Differentiation

The core of the project is the `backward()` method.

When:

```python
loss.backward()
```

is called, the engine:

1. Builds a topological ordering of the computational graph.
2. Resets gradients in the graph.
3. Sets the loss gradient to:

$$
\frac{\partial L}{\partial L}=1
$$

4. Traverses the graph in reverse.
5. Applies the chain rule to calculate gradients.

This is the fundamental mechanism behind reverse-mode automatic differentiation.

For example, multiplication follows:

$$
z=a\times b
$$

Therefore:

$$
\frac{\partial z}{\partial a}=b
$$

and

$$
\frac{\partial z}{\partial b}=a
$$

The implementation propagates these gradients backward through the graph.

---

# 🧮 Computational Graph

Every operation creates a new node.

For example:

```python
a = Params(2.0)
b = Params(3.0)

c = a * b
d = c + a
```

Conceptually, the graph becomes:

```text
       a ───────────────┐
       │                │
       ▼                ▼
       × ──────────────► +
       ▲                 │
       │                 │
       b                 ▼
                         d
```

The `trace()` function performs a depth-first traversal of the graph and collects:

* Nodes
* Directed edges

This structure can also be used for computational-graph visualization with Graphviz.

---

# 🧠 Neural Network Implementation

After implementing the scalar autograd engine, the project builds a neural-network abstraction on top of it.

## Neuron

A neuron computes:

$$
z = \sum_i w_i x_i+b
$$

followed by an activation function:

$$
y=f(z)
$$

The `Neuron` class supports:

* Random weight initialization
* Bias
* Tanh activation
* ReLU activation
* Parameter collection

Weights are initialized from a uniform distribution between `-1` and `1`, while biases start at zero.

Example:

```python
neuron = Neuron(3)

output = neuron([
    Params(1.0),
    Params(2.0),
    Params(3.0)
])
```

---

# 🏗️ Fully Connected Layer

A `Layer` consists of multiple independent neurons.

```python
layer = Layer(
    nin=2,
    nout=4
)
```

Conceptually:

```text
Input
 x1 ─────┬────► Neuron 1 ───► y1
         │
 x2 ─────┼────► Neuron 2 ───► y2
         │
         ├────► Neuron 3 ───► y3
         │
         └────► Neuron 4 ───► y4
```

The layer automatically collects parameters from all its neurons.

---

# 🧠 Multi-Layer Perceptron

The `MLP` class combines multiple fully connected layers.

For example:

```python
model = MLP(2, [4, 4, 1])
```

creates:

```text
2 Input Features
       │
       ▼
┌─────────────┐
│  Layer: 4   │
│   Neurons   │
└─────────────┘
       │
       ▼
┌─────────────┐
│  Layer: 4   │
│   Neurons   │
└─────────────┘
       │
       ▼
┌─────────────┐
│  Output: 1  │
│   Neuron    │
└─────────────┘
       │
       ▼
 Prediction
```

The implementation uses `tanh` throughout the MLP, with the output constrained to `(-1, 1)` to match the `-1/+1` target representation used in the experiments.

---

# 🔥 Training Loop

The project demonstrates the complete neural-network training pipeline:

```text
Input
  ↓
Forward Pass
  ↓
Prediction
  ↓
MSE Loss
  ↓
Backward Pass
  ↓
Gradients
  ↓
Parameter Update
  ↓
Repeat
```

The parameter update follows basic gradient descent:

$$
\theta := \theta-\eta\frac{\partial L}{\partial\theta}
$$

where:

* `θ` = model parameter
* `η` = learning rate
* `L` = loss

Example from the implementation:

```python
model.zero_grad()

loss.backward()

for p in model.parameters():
    p.data -= learning_rate * p.grad
```

---

# 🧪 Experiments

## 1. XOR

The first experiment uses the classic XOR problem.

Dataset:

| Input    | Target |
| -------- | -----: |
| `(0, 0)` |   `-1` |
| `(0, 1)` |   `+1` |
| `(1, 0)` |   `+1` |
| `(1, 1)` |   `-1` |

The notebook uses:

```python
xor_model = MLP(2, [4, 4, 1])
```

and trains it using MSE loss for 500 epochs with a learning rate of `0.1`.

The recorded training run reduced the loss from approximately:

```text
1.010474
```

to:

```text
0.004176
```

The resulting predictions were close to the expected `±1` targets:

```text
(0, 0) → -0.9627
(0, 1) → +0.9291
(1, 0) → +0.9330
(1, 1) → -0.9242
```

---

# 🌙 2. Synthetic Moons

The project also evaluates the MLP on a nonlinear synthetic classification dataset generated using:

```python
make_moons(
    n_samples=200,
    noise=0.15,
    random_state=42
)
```

The labels are converted from:

```text
{0, 1}
```

to:

```text
{-1, +1}
```

This dataset is useful for demonstrating why nonlinear activation functions and hidden layers are necessary for learning nonlinear decision boundaries.

---

# 🌸 3. Iris Dataset

The notebook also uses the classic Iris dataset.

The experiment selects:

* The first two classes
* The first two features

The resulting labels are converted to:

```text
-1 / +1
```

The selected Iris features are standardized using zero mean and unit variance before training.

---

# 📊 Visualization

The project contains utilities for visualizing datasets and learned decision boundaries.

### Dataset Visualization

```python
plot_data(X, y, title)
```

creates a 2D scatter plot of the dataset.

### Decision Boundary

```python
plot_boundary(model, X, y, title)
```

creates a mesh grid across the feature space, performs a forward pass at every grid point, and visualizes the model's output as a decision surface.

The decision boundary is represented by the zero contour:

$$
f(x)=0
$$

This is particularly useful because the model produces predictions in the range `(-1, 1)`.

---

# 📁 Project Structure

A suggested repository structure is:

```text
Mini-PyTorch/
│
├── Mini_Pytorch_improved.ipynb
├── README.md
└── requirements.txt
```

The notebook currently contains the complete implementation and experiments.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Mini-PyTorch.git
cd Mini-PyTorch
```

Install the required packages:

```bash
pip install numpy matplotlib scikit-learn
```

Or:

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
numpy
matplotlib
scikit-learn
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Mini_Pytorch_improved.ipynb
```

Run the notebook cells sequentially.

The notebook will:

1. Build the custom autograd engine.
2. Test arithmetic operations.
3. Construct computational graphs.
4. Build neurons.
5. Build fully connected layers.
6. Build an MLP.
7. Train the model on XOR.
8. Load Synthetic Moons and Iris.
9. Visualize datasets.
10. Visualize decision boundaries.

---

# 🧩 Key Concepts Demonstrated

This project provides a hands-on implementation of several important deep-learning concepts:

### Automatic Differentiation

Automatically calculates derivatives through a computational graph.

### Computational Graphs

Represents mathematical operations as connected nodes.

### Backpropagation

Propagates gradients from the loss toward the model parameters.

### Chain Rule

The mathematical foundation used to propagate derivatives through multiple operations.

### Gradient Descent

Updates model parameters using calculated gradients.

### Neural Networks

Builds neurons, layers, and an MLP from scratch.

### Activation Functions

Implements:

* ReLU
* Tanh
* Sigmoid

### Loss Function

Uses Mean Squared Error:

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
$$

### Decision Boundaries

Visualizes how the trained MLP separates different classes.

---

# 🎯 Why Build This?

Using PyTorch is extremely convenient, but it can hide the underlying mechanics.

For example, PyTorch allows:

```python
loss.backward()
```

without requiring the developer to manually implement the chain rule.

This project removes that abstraction and implements the fundamental mechanics manually.

The goal is therefore not to replace PyTorch, but to understand **what happens underneath a deep-learning framework**.

---

# 📚 Learning Outcomes

After working through this project, you should have a better understanding of:

* How computational graphs are constructed
* How gradients flow through a graph
* How reverse-mode autodiff works
* How the chain rule is applied
* How neurons calculate outputs
* How layers are composed
* How an MLP performs a forward pass
* How losses generate gradients
* How gradient descent updates parameters
* Why hidden layers are useful for nonlinear problems
* How neural networks learn decision boundaries

---

# ⚠️ Limitations

This is an educational implementation rather than a production deep-learning framework.

Current limitations include:

* Scalar-based computation rather than tensor operations
* No GPU acceleration
* No batching abstraction
* No optimizers such as Adam or RMSProp
* No automatic mixed precision
* No convolutional layers
* No recurrent/transformer architectures
* Limited operation support
* Manual parameter updates
* Designed primarily for learning and experimentation

---

# 🔮 Possible Future Improvements

Potential extensions include:

* [ ] Add tensor-based automatic differentiation
* [ ] Add matrix operations
* [ ] Implement Softmax
* [ ] Implement Cross-Entropy Loss
* [ ] Add SGD optimizer class
* [ ] Add Adam optimizer
* [ ] Add mini-batch training
* [ ] Add train/validation/test splits
* [ ] Add accuracy, precision, recall and F1 metrics
* [ ] Add model saving/loading
* [ ] Add gradient checking with numerical derivatives
* [ ] Improve computational graph visualization
* [ ] Add unit tests
* [ ] Add convolutional neural networks
* [ ] Add simple Transformer components

---

# 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook**
* **Graphviz** *(optional for computational graph visualization)*

---

# 🙏 Inspiration

This project is inspired by the educational philosophy behind Andrej Karpathy's **micrograd** — implementing the fundamentals of automatic differentiation and neural networks from scratch to understand how modern deep-learning frameworks work internally.

---

# 👨‍💻 Author

**Mohammad Sajid**

B.Tech — Artificial Intelligence & Machine Learning

Interested in:

* Machine Learning
* AI Engineering
* Deep Learning
* Generative AI
* LLM Applications
* AI Systems

---

## ⭐ If You Found This Useful

If this project helped you understand automatic differentiation, backpropagation, or neural networks better, consider giving the repository a ⭐.

**Built from scratch to understand what happens behind `loss.backward()`.**
