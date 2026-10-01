"""
An introduction to PyTorch: the exercises of a 90-minute hands-on session for
readers of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.
The concepts are presented with the slides (`slides/slides.typ`).

Launch with:  uv run marimo edit pytorch_intro.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="An introduction to PyTorch: exercises",
)

with app.setup(hide_code=True):
    import inspect
    import math
    import time
    from itertools import islice
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from matplotlib.colors import ListedColormap
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

    NOTEBOOK_DIR = mo.notebook_dir() or Path.cwd()
    DATA_DIR = NOTEBOOK_DIR / "data"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.set_printoptions(precision=3, sci_mode=False)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
        }
    )
    CLASS_CMAP = ListedColormap(["tab:blue", "tab:orange", "tab:green"])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 🔥 An introduction to PyTorch: exercises

    *Getting ready for **Build a Large Language Model (From Scratch)** by Sebastian Raschka*

    The concepts are presented on the slides, this notebook holds the exercises. Each exercise has:

    1. a **statement**;
    2. a cell to complete: look for the `TODO`s;
    3. an **automatic check** below that cell (⏳ to do, ✅ correct, ❌ not quite);
    4. a **solution**, hidden in the "💡 Solution" box.

    /// tip | About marimo
    This is a **reactive** [marimo](https://marimo.io) notebook: when you edit a cell, every cell that depends on it
    re-runs. A variable defined at the top level of a cell is **shared with the whole notebook**, so no other cell
    can define it again. That is why most exercise cells wrap their code in a function (`exercise_1`, `exercise_3`, …):
    its variables stay local, and every exercise can reuse the same names.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 1 · Tensors
    """)
    return


@app.cell(hide_code=True)
def _():
    exercise(
        1,
        "Shape gymnastics",
        r"""
        Starting from `x = torch.arange(24).reshape(2, 3, 4)`, build:

        - `a` of shape `(4, 2, 3)`: move the last dimension to the front;
        - `b` of shape `(6, 4)`: merge the first two dimensions;
        - `c` of shape `(2, 1, 3, 4)`: insert a dimension of size 1;
        - `d` of shape `(2, 3, 1)`: the mean over the last dimension, keeping that dimension. ⚠️ `x` holds integers.
        """,
        r"""
        ```python
        a = x.permute(2, 0, 1)
        b = x.reshape(6, 4)  # or x.view(6, 4), or x.flatten(0, 1)
        c = x.unsqueeze(1)  # or x[:, None]
        d = x.to(torch.float32).mean(dim=-1, keepdim=True)
        ```

        `mean` is only defined for floating-point tensors: on int64, it raises a `RuntimeError`.
        """,
    )
    return


@app.cell
def _():
    def exercise_1():
        x = torch.arange(24).reshape(2, 3, 4)

        a = x.permute(2, 0, 1)  # TODO: shape (4, 2, 3)
        b = x.view(6, -1)  # TODO: shape (6, 4)
        c = x.unsqueeze(1)  # TODO: shape (2, 1, 3, 4)
        d = x.to(torch.float32).mean(dim=-1, keepdim=True)  # TODO: shape (2, 3, 1)

        return check_1(a, b, c, d)

    exercise_1()
    return


@app.cell(hide_code=True)
def _():
    exercise(
        2,
        "Softmax by hand",
        r"""
        Complete `my_softmax(x)`, which applies a softmax over the **last** dimension of `x`, without using
        `torch.softmax`:

        $$\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

        Every row of the result must sum to 1. Hint: `keepdim=True`.
        """,
        r"""
        ```python
        def my_softmax(x):
            e = torch.exp(x)
            return e / e.sum(dim=-1, keepdim=True)  # (..., n) / (..., 1)
        ```

        - With `keepdim=True`, the sum has shape `(4, 1)` and broadcasts over each row. Without it, `(4, 5) / (4,)`
          raises an error (or, worse, silently divides by the wrong values for a square input).
        """,
    )
    return


@app.cell
def _():
    def my_softmax(x: torch.Tensor):
        """Softmax over the last dimension of `x`."""
        e = torch.exp(x)
        return e / torch.sum(e, dim=-1, keepdims=True)

    check_2(my_softmax)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 2 · Autograd
    """)
    return


@app.cell(hide_code=True)
def _():
    exercise(
        3,
        "A derivative with autograd",
        r"""
        Compute the derivative of $y = x^3 + 2x$ at $x = 2$ with autograd. By hand, $3x^2 + 2 = 14$.
        """,
        r"""
        ```python
        x = torch.tensor(2.0, requires_grad=True)
        y = x**3 + 2 * x
        y.backward()
        grad = x.grad  # tensor(14.)
        ```

        Gradients **accumulate**: run the forward pass again, then backpropagate, and `x.grad` becomes 28.

        ```python
        y = x**3 + 2 * x  # a new graph
        y.backward()
        x.grad  # tensor(28.)
        ```

        Calling `y.backward()` twice on the **same** `y` raises a `RuntimeError`: the first call frees the values
        that the forward pass saved for the backward pass.
        """,
    )
    return


@app.cell
def _():
    def exercise_3():
        x = torch.tensor(
            2.0, requires_grad=True
        )  # TODO: ask PyTorch to track gradients
        y = x**3 + 2 * x  # TODO: y = x³ + 2x
        # TODO: backpropagate from y
        y.backward()
        grad = x.grad  # TODO: read dy/dx

        return check_3(x, grad)

    exercise_3()
    return


@app.cell(hide_code=True)
def _():
    exercise(
        4,
        "Fit a line with an optimizer",
        r"""
        The data below follows $y = 3x + 1$, plus noise. Fit $\hat y = w x + b$ by minimising the mean squared error,
        with the optimizer `torch.optim.SGD` (learning rate 0.1). At each step:

        1. compute the loss;
        2. reset the gradients: `optimizer.zero_grad()`;
        3. backpropagate;
        4. update `w` and `b`: `optimizer.step()`.
        """,
        r"""
        ```python
        optimizer = torch.optim.SGD([w, b], lr=0.1)
        for step in range(200):
            loss = ((w * x + b - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        ```

        - The optimizer receives the tensors to train: here `[w, b]`, later `model.parameters()`.
        - `optimizer.step()` applies $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ to each of them,
          using the gradients stored in `.grad`.
        - Without `optimizer.zero_grad()`, the gradients of every step would **accumulate**.
        - `Adam` and `AdamW` (used in the book) have the same interface, with smarter update rules.
        """,
    )
    return


@app.cell
def _():
    def exercise_4():
        torch.manual_seed(0)
        x = torch.linspace(-1, 1, 50)
        y = 3 * x + 1 + 0.1 * torch.randn(50)

        w = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.SGD(
            [w, b], lr=0.1
        )  # TODO: SGD over [w, b], with a learning rate of 0.1
        losses = []
        for step in range(200):
            res = w * x + b
            loss = torch.mean(
                (res - y) ** 2
            )  # TODO: the mean squared error of w * x + b
            # TODO: reset the gradients
            optimizer.zero_grad()
            # TODO: backpropagate
            loss.backward()
            # TODO: update w and b
            optimizer.step()
            losses.append(loss.item())
        return check_4(x, y, w, b, optimizer, losses)

    exercise_4()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 3 · `nn.Module`
    """)
    return


@app.cell(hide_code=True)
def _():
    exercise(
        5,
        "The book's network (listing A.4)",
        r"""
        Complete `NeuralNetwork`: 50 inputs → 30 hidden units (ReLU) → 20 hidden units (ReLU) → 3 outputs (the logits).
        Then count its **trainable parameters**, using `.parameters()` and `.numel()`.
        """,
        r"""
        ```python
        class NeuralNetwork(nn.Module):
            def __init__(self, num_inputs, num_outputs):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(num_inputs, 30),
                    nn.ReLU(),
                    nn.Linear(30, 20),
                    nn.ReLU(),
                    nn.Linear(20, num_outputs),
                )

            def forward(self, x):
                return self.layers(x)

        model = NeuralNetwork(50, 3)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 2,213
        ```

        $(50 \times 30 + 30) + (30 \times 20 + 20) + (20 \times 3 + 3) = 1530 + 620 + 63 = 2213$:
        each `nn.Linear` has a weight matrix and a bias vector.
        """,
    )
    return


@app.cell
def _():
    class NeuralNetwork(nn.Module):
        def __init__(self, num_inputs: int, num_outputs: int):
            super().__init__()
            self.layers = nn.Sequential(
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, num_outputs),
                # TODO: 1st hidden layer (30 units) and its ReLU
                # TODO: 2nd hidden layer (20 units) and its ReLU
                # TODO: output layer
            )

        def forward(self, x: torch.Tensor):
            return self.layers(x)

    def exercise_5():
        model = NeuralNetwork(50, 3)
        n_params = sum(
            p.numel() for p in model.parameters()
        )  # TODO: count the trainable parameters
        return check_5(model, n_params)

    exercise_5()
    return


@app.cell(hide_code=True)
def _():
    exercise(
        6,
        "Your own `LayerNorm` (book, chapter 4)",
        r"""
        Complete `MyLayerNorm`. Over the **last** dimension, normalise `x` to zero mean and unit variance, then apply
        a trainable `scale` (initialised to ones) and `shift` (zeros):

        $$\mathrm{LayerNorm}(x) = \mathrm{scale} \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \mathrm{shift}$$

        Use the biased variance (`unbiased=False`). The check compares your module with `nn.LayerNorm`.
        """,
        r"""
        ```python
        class MyLayerNorm(nn.Module):
            def __init__(self, emb_dim):
                super().__init__()
                self.eps = 1e-5
                self.scale = nn.Parameter(torch.ones(emb_dim))
                self.shift = nn.Parameter(torch.zeros(emb_dim))

            def forward(self, x):
                mean = x.mean(dim=-1, keepdim=True)  # (b, t, 1)
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                norm_x = (x - mean) / torch.sqrt(var + self.eps)
                return self.scale * norm_x + self.shift
        ```

        Wrapping the tensors in `nn.Parameter` registers them: they appear in `.parameters()` and get trained.
        """,
    )
    return


@app.cell
def _():
    class MyLayerNorm(nn.Module):
        def __init__(self, emb_dim: int):
            super().__init__()
            self.eps = 1e-5
            # TODO: the trainable `scale` and `shift` parameters

        def forward(self, x: torch.Tensor):
            return x  # TODO: normalise over the last dimension, then scale and shift

    check_6(MyLayerNorm)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 4 · `Dataset` and `DataLoader`

    Exercises 7 and 8 use a 2-D classification problem that a linear model can not solve: three interleaved spirals.
    """)
    return


@app.cell
def _():
    x_train, y_train, x_val, y_val = make_spirals()
    print(
        f"X_train: {tuple(x_train.shape)} {x_train.dtype} | y_train: {tuple(y_train.shape)} {y_train.dtype}"
    )
    print(f"X_val  : {tuple(x_val.shape)} | y_val: {tuple(y_val.shape)}")

    fig, ax = plt.subplots(figsize=(3.6, 3.6), constrained_layout=True)
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=CLASS_CMAP, s=8)
    ax.set_aspect("equal")
    ax.set_title("Training set")
    fig
    return x_train, x_val, y_train, y_val


@app.cell(hide_code=True)
def _():
    exercise(
        7,
        "A `Dataset` and its `DataLoader` (listing A.6)",
        r"""
        Complete `ToyDataset`: `__len__` returns the number of samples, and `__getitem__` the pair
        `(features, label)` of sample `index`. Then wrap the training set in a `DataLoader` that yields
        **shuffled** batches of **32** samples.
        """,
        r"""
        ```python
        class ToyDataset(Dataset):
            def __init__(self, X, y):
                self.features = X
                self.labels = y

            def __getitem__(self, index):
                return self.features[index], self.labels[index]

            def __len__(self):
                return self.labels.shape[0]

        train_ds = ToyDataset(x_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        ```

        For tensors already in memory, `TensorDataset(X, y)` does the same job.
        """,
    )
    return


@app.cell
def _(x_train, y_train):
    class ToyDataset(Dataset):
        def __init__(self, x: torch.Tensor, y: torch.Tensor):
            self.features = x
            self.labels = y

        def __getitem__(self, index: int):
            return self.features[index], self.labels[
                index
            ]  # TODO: the (features, label) pair of sample `index`

        def __len__(self):
            return len(self.features)  # TODO: the number of samples

    def exercise_7():
        train_ds = ToyDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=32, shuffle=True
        )  # TODO: shuffled batches of 32 samples

        return check_7(train_ds, train_loader, x_train, y_train)

    exercise_7()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 5 · The training loop
    """)
    return


@app.cell(hide_code=True)
def _():
    exercise(
        8,
        "Train a classifier",
        r"""
        Train the multilayer perceptron below on the spirals: complete the **five steps** of the training loop
        (forward pass, loss, reset the gradients, backward pass, update), and record `loss.item()` at each step.
        Goal: a validation accuracy above **90 %**.

        Then experiment: a learning rate of 0.5, or a linear model (`nn.Linear(2, 3)` only).
        """,
        r"""
        ```python
        for features, labels in train_loader:
            logits = model(features)  # 1. forward
            loss = F.cross_entropy(logits, labels)  # 2. loss
            optimizer.zero_grad()  # 3. reset the gradients
            loss.backward()  # 4. backward
            optimizer.step()  # 5. update
            losses.append(loss.item())
        ```

        `F.cross_entropy` takes the **raw logits**: no softmax. A linear model gets stuck below 50 %: its decision
        regions can only be separated by straight lines.
        """,
    )
    return


@app.cell
def _(x_train, x_val, y_train, y_val):
    def exercise_8():
        model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=32, shuffle=True
        )
        losses = []

        for epoch in range(100):
            model.train()
            for features, labels in train_loader:
                prediction = model(features)
                loss = F.cross_entropy(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                # pass  # TODO: the five steps, then append loss.item() to losses

        return check_8(model, losses, x_val, y_val)

    exercise_8()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 6 · Towards LLMs: a tiny character-level GPT

    Time to put everything together: we train a (very) small GPT on Tiny Shakespeare, with **one token per
    character** (the book uses GPT-2's tokenizer, with 50,257 tokens).

    1. cut the text into training examples (exercise 9);
    2. compute the loss of the model (exercise 10);
    3. train it, and generate text (exercise 11).

    The model, `TinyGPT`, follows the architecture of chapter 4: you do not need to understand attention today.
    Its code, and that of the training function, is in the appendix.
    """)
    return


@app.cell(hide_code=True)
def _():
    exercise(
        9,
        "Sliding windows for next-token prediction (book, chapter 2)",
        r"""
        An LLM learns to predict the **next token**. Complete `NextTokenDataset`, modelled on the book's
        `GPTDatasetV1`: starting at positions `0, stride, 2 * stride, …`, cut windows of `max_length` tokens.
        The **target** of each window is the same window shifted by one token, so a window must leave room for its
        last target token.

        The check uses a naive tokenizer (one id per character) on the first sentence of the book's short story.
        """,
        r"""
        ```python
        for i in range(0, len(token_ids) - max_length, stride):
            self.features.append(token_ids[i : i + max_length])
            self.targets.append(token_ids[i + 1 : i + max_length + 1])
        ```

        Every position of a window is a training example: after `"I HAD al"`, predict `"w"`. Chapter 5 trains with
        `stride = max_length`, so that the windows do not overlap.
        """,
    )
    return


@app.cell
def _():
    class NextTokenDataset(Dataset):
        def __init__(self, token_ids: torch.Tensor, max_length: int, stride: int):
            self.features = []
            self.targets = []
            for i in range(0, len(token_ids) - max_length, stride):
                self.features.append(token_ids[i : i + max_length])
                self.targets.append(token_ids[i + 1 : i + max_length + 1])
            # TODO: a window of `max_length` tokens every `stride` tokens, and its target shifted by one

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx: int):
            return self.features[idx], self.targets[idx]

    def exercise_9():
        text = "I HAD always thought Jack Gisburn rather a cheap genius"
        token_ids = torch.tensor(
            [ord(ch) for ch in text]
        )  # a naive tokenizer: one id per character
        dataset = NextTokenDataset(token_ids, max_length=8, stride=4)

        return check_9(dataset, text, max_length=8, stride=4)

    exercise_9()
    return


@app.cell
def _():
    corpus = (DATA_DIR / "tinyshakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(corpus)

    def _():
        # Windows of context_length + 1 tokens: the same as NextTokenDataset with stride = max_length
        windows = tokenizer.encode(corpus).unfold(0, 65, 64)
        split = int(
            0.9 * len(windows)
        )  # 90 % training, 10 % validation, as in chapter 5
        train_loader = DataLoader(
            TensorDataset(windows[:split, :-1], windows[:split, 1:]),
            batch_size=32,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(windows[split:, :-1], windows[split:, 1:]), batch_size=32
        )
        return train_loader, val_loader

    lm_train_loader, lm_val_loader = _()

    print(f"{len(corpus):,} characters, vocabulary of {tokenizer.vocab_size} tokens")
    print(
        f"{len(lm_train_loader)} training batches, {len(lm_val_loader)} validation batches"
    )
    return lm_train_loader, lm_val_loader, tokenizer


@app.cell(hide_code=True)
def _():
    exercise(
        10,
        "The loss of a language model",
        r"""
        `TinyGPT` returns logits of shape `(batch, tokens, vocab_size)`, and the targets have shape `(batch, tokens)`.
        But `F.cross_entropy` expects logits of shape `(N, classes)` and targets of shape `(N,)`.
        Compute the loss of the **untrained** model.

        Sanity check: an untrained model guesses uniformly, so its loss should be close to $\ln(\text{vocab size})$.
        """,
        r"""
        ```python
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        ```

        After flattening, every position of every sequence is a classification example: `(32 × 64, 65)` logits
        and `(32 × 64,)` targets. Chapter 5 does exactly this in `calc_loss_batch`.
        """,
    )
    return


@app.cell
def _(lm_train_loader, tokenizer):
    def exercise_10():
        model = TinyGPT(tokenizer.vocab_size)
        inputs, targets = next(iter(lm_train_loader))  # (32, 64) and (32, 64)
        logits = model(inputs)  # shape (32, 64, 65)
        logits = logits.view(-1, 65)
        targets = targets.flatten()
        loss = F.cross_entropy(logits, targets)  # TODO

        return check_10(loss, logits, targets, tokenizer.vocab_size)

    exercise_10()
    return


@app.cell
def _(lm_train_loader, lm_val_loader, tokenizer):
    torch.manual_seed(123)

    lm_model = TinyGPT(tokenizer.vocab_size)

    def _():
        start = time.perf_counter()
        losses = train_lm(
            lm_model, lm_train_loader, lm_val_loader, num_steps=1000, lr=0.01
        )
        return mo.md(
            f"Trained `TinyGPT` ({sum(p.numel() for p in lm_model.parameters()):,} parameters) "
            f"in {time.perf_counter() - start:.1f} s on `{device}`: training loss **{losses['train']:.2f}**, "
            f"validation loss **{losses['validation']:.2f}**."
        )

    _()
    return (lm_model,)


@app.cell(hide_code=True)
def _():
    exercise(
        11,
        "Text generation (book, chapter 5)",
        r"""
        Complete `generate`, which extends the token ids `prompt` (shape `(batch, tokens)`) by
        `max_new_tokens` tokens. Starting from `prompt`, at each step:

        1. crop the sequence to its last `context_size` tokens (the model cannot see more);
        2. compute the logits, without recording the graph;
        3. keep the logits of the **last** position: `(batch, vocab_size)`;
        4. pick the next token: the most likely one (`argmax`) if `temperature == 0`, otherwise sample it from
           `softmax(logits / temperature)` with `torch.multinomial`;
        5. append it to the sequence.

        Then play with the prompt and the temperature in the cell after the check.
        """,
        r"""
        ```python
        def generate(model, prompt, max_new_tokens, context_size, temperature=0.0):
            idx = prompt
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -context_size:]
                with torch.no_grad():
                    logits = model(idx_cond)
                logits = logits[:, -1, :]
                if temperature > 0.0:
                    probas = torch.softmax(logits / temperature, dim=-1)
                    idx_next = torch.multinomial(probas, num_samples=1)  # (batch, 1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
        ```

        A temperature below 1 sharpens the distribution (safer text), above 1 flattens it (wilder text).
        Greedy decoding (`temperature=0`) often gets stuck in a loop.
        """,
    )
    return


@app.cell
def _(lm_model, tokenizer):
    def generate(
        model: TinyGPT,
        prompt: torch.Tensor,
        max_new_tokens: int,
        context_size: int,
        temperature=0.0,
    ):
        # prompt has shape (batch_size, prompt_length)
        result = prompt
        with torch.no_grad():
            for _ in range(max_new_tokens):
                inputs = result[:, -context_size:]
                predicted = model(inputs)
                next_token_logits = predicted[:, -1, :]
                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                result = torch.cat([result, next_token], dim=1)
                # TODO: the five steps of the statement
        return result

    check_11(generate, lm_model, tokenizer, lm_model.context_length)
    return (generate,)


@app.cell
def _(generate, lm_model, tokenizer):
    prompt = "ROMEO:"
    prompt = (
        tokenizer.encode(prompt).unsqueeze(0).to(device)
    )  # add the batch dimension with unsqueeze: (1, tokens)
    out = generate(
        lm_model,
        prompt,
        max_new_tokens=300,
        context_size=lm_model.context_length,
        temperature=0.8,
    )
    mo.callout(mo.plain_text(tokenizer.decode(out[0].cpu())), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    That's all! **Next:** chapter 2 of the book, working with text data. 🚀
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Appendix: helpers

    The cells below define the checks of the exercises, the data, etc. Their code is hidden because you do not need to read it but feel free to have a look.
    """)
    return


@app.function(hide_code=True)
def exercise(number, title, statement, solution):
    """
    Render an exercise: its statement, followed by its solution folded in an
    accordion so that it stays hidden until the reader opens it.

    @ai-generated
    """
    return mo.vstack(
        [
            mo.md(
                f"### ✍️ Exercise {number} · {title}\n\n" + inspect.cleandoc(statement)
            ),
            mo.accordion(
                {"💡 Solution (try first!)": mo.md(inspect.cleandoc(solution))}
            ),
        ]
    )


@app.function(hide_code=True)
def report(*checks):
    """
    Render the outcome of an exercise as a checklist callout.

    Each check is a ``(label, test)`` tuple. ``test()`` returns True
    (passed), False (failed), or None (not attempted yet); an exception
    counts as a failure and its message is shown.

    @ai-generated
    """
    lines, states = [], []
    for label, test in checks:
        note = ""
        try:
            result = test()
            ok = None if result is None else bool(result)
        except Exception as err:  # the student's code may raise anything
            ok = False
            message = (
                str(err).splitlines()[0][:160].replace("`", "'") if str(err) else ""
            )
            note = f" (`{type(err).__name__}`: {message})"
        states.append(ok)
        lines.append(f"- {({True: '✅', False: '❌', None: '⏳'})[ok]} {label}{note}")

    if all(state is True for state in states):
        kind, header = "success", "**✅ Correct!**"
    elif False in states:
        kind, header = "danger", "**❌ Not quite yet.**"
    else:
        kind, header = "neutral", "**⏳ To do:** complete the `TODO`s above."
    return mo.callout(mo.md(header + "\n\n" + "\n".join(lines)), kind=kind)


@app.function(hide_code=True)
def check_1(a, b, c, d):
    """Check exercise 1 against reference tensors. @ai-generated"""
    x = torch.arange(24).reshape(2, 3, 4)
    return report(
        (
            "`a`: last dimension first, `(4, 2, 3)`",
            lambda: None if a is None else torch.equal(a, x.permute(2, 0, 1)),
        ),
        (
            "`b`: first two dimensions merged, `(6, 4)`",
            lambda: None if b is None else torch.equal(b, x.reshape(6, 4)),
        ),
        (
            "`c`: new dimension, `(2, 1, 3, 4)`",
            lambda: None if c is None else torch.equal(c, x.unsqueeze(1)),
        ),
        (
            "`d`: mean over the last dimension, `(2, 3, 1)`",
            lambda: (
                None
                if d is None
                else d.shape == (2, 3, 1)
                and torch.allclose(d, x.to(torch.float32).mean(-1, keepdim=True))
            ),
        ),
    )


@app.function(hide_code=True)
def check_2(fn):
    """Check exercise 2: ``fn`` must match ``torch.softmax`` over the last dim. @ai-generated"""
    gen = torch.Generator().manual_seed(0)
    x2, x3 = torch.randn(4, 5, generator=gen), torch.randn(2, 3, 5, generator=gen)

    def matches(x):
        out = fn(x)
        return (
            None
            if out is None
            else out.shape == x.shape
            and torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)
        )

    return report(
        (
            "matches `torch.softmax(x, dim=-1)` for `x` of shape `(4, 5)`",
            lambda: matches(x2),
        ),
        ("works for a 3-D tensor `(2, 3, 5)`", lambda: matches(x3)),
    )


@app.function(hide_code=True)
def check_3(x, grad):
    """Check exercise 3. @ai-generated"""
    return report(
        (
            "`x` tracks gradients",
            lambda: None if grad is None and not x.requires_grad else x.requires_grad,
        ),
        (
            "$dy/dx = 14$ at $x = 2$",
            lambda: (
                None if grad is None else math.isclose(float(grad), 14.0, rel_tol=1e-5)
            ),
        ),
    )


@app.function(hide_code=True)
def check_4(x, y, w, b, optimizer, losses):
    """Check exercise 4 and, once attempted, plot the fit and the loss curve. @ai-generated"""
    untouched = w.item() == 0.0 and b.item() == 0.0
    result = report(
        (
            "an `SGD` optimizer over `[w, b]`",
            lambda: (
                None
                if optimizer is None
                else isinstance(optimizer, torch.optim.SGD)
                and {id(p) for g in optimizer.param_groups for p in g["params"]}
                == {id(w), id(b)}
            ),
        ),
        ("$w \\approx 3$", lambda: None if untouched else abs(w.item() - 3) < 0.15),
        ("$b \\approx 1$", lambda: None if untouched else abs(b.item() - 1) < 0.15),
        ("the loss decreases", lambda: None if untouched else losses[-1] < losses[0]),
    )
    if untouched:
        return result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.8), constrained_layout=True)
    ax1.scatter(x, y, s=10, color="tab:gray", label="data")
    ax1.plot(
        x,
        (w * x + b).detach(),
        color="tab:red",
        lw=2,
        label=f"fit: w = {w.item():.2f}, b = {b.item():.2f}",
    )
    ax1.legend(loc="upper left")
    ax2.semilogy(losses)
    ax2.set_xlabel("step")
    ax2.set_ylabel("MSE loss")
    return mo.vstack([result, fig])


@app.function(hide_code=True)
def check_5(model, n_params):
    """Check exercise 5. @ai-generated"""
    expected_shapes = [(30, 50), (30,), (20, 30), (20,), (3, 20), (3,)]
    shapes = [tuple(p.shape) for p in model.parameters()]
    return report(
        (
            "layers 50 → 30 → 20 → 3",
            lambda: None if not shapes else shapes == expected_shapes,
        ),
        (
            "ReLU activations between the layers",
            lambda: (
                None
                if not shapes
                else sum(isinstance(m, nn.ReLU) for m in model.modules()) == 2
            ),
        ),
        (
            "`model(x)` maps `(4, 50)` to `(4, 3)`",
            lambda: None if not shapes else model(torch.rand(4, 50)).shape == (4, 3),
        ),
        (
            "number of trainable parameters",
            lambda: None if n_params is None else n_params == 2213,
        ),
    )


@app.function(hide_code=True)
def check_6(layer_norm_cls):
    """Check exercise 6 against ``nn.LayerNorm``. @ai-generated"""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6) * 3 + 1
    mine, ref = layer_norm_cls(6), nn.LayerNorm(6)
    n_params = sum(p.numel() for p in mine.parameters())

    def same_output():
        out = mine(x)
        return None if torch.equal(out, x) else torch.allclose(out, ref(x), atol=1e-5)

    return report(
        (
            "trainable `scale` and `shift` (12 parameters)",
            lambda: None if n_params == 0 else n_params == 12,
        ),
        ("same output as `nn.LayerNorm(6)`", same_output),
    )


@app.function(hide_code=True)
def make_spirals(n_per_class=150, n_classes=3, noise=0.4, seed=0):
    """
    Generate a 2-D classification dataset of ``n_classes`` interleaved spirals
    and split it into training (80 %) and validation (20 %) sets.

    Returns float32 features of shape (N, 2) and int64 labels of shape (N,),
    which are the dtypes expected by ``nn.Linear`` and ``F.cross_entropy``.

    @ai-generated
    """
    generator = torch.Generator().manual_seed(seed)
    features, labels = [], []
    for c in range(n_classes):
        radius = torch.linspace(0.05, 1.0, n_per_class)
        theta = torch.linspace(4.0 * c, 4.0 * (c + 1), n_per_class)
        theta = theta + noise * torch.randn(n_per_class, generator=generator)
        features.append(
            torch.stack([radius * torch.sin(theta), radius * torch.cos(theta)], dim=1)
        )
        labels.append(torch.full((n_per_class,), c, dtype=torch.long))
    X, y = torch.cat(features), torch.cat(labels)
    perm = torch.randperm(len(X), generator=generator)
    n_train = int(0.8 * len(X))
    return X[perm[:n_train]], y[perm[:n_train]], X[perm[n_train:]], y[perm[n_train:]]


@app.function(hide_code=True)
def check_7(dataset, loader, X, y):
    """Check exercise 7. @ai-generated"""

    def first_item():
        item = dataset[0]
        return (
            None
            if item is None
            else torch.equal(item[0], X[0]) and torch.equal(item[1], y[0])
        )

    def batch_shapes():
        if loader is None:
            return None
        features, labels = next(iter(loader))
        return features.shape == (32, 2) and labels.shape == (32,)

    return report(
        (
            f"`len(dataset)` is {len(X)}",
            lambda: None if len(dataset) == 0 else len(dataset) == len(X),
        ),
        ("`dataset[0]` is `(X_train[0], y_train[0])`", first_item),
        ("batches of features `(32, 2)` and labels `(32,)`", batch_shapes),
        (
            "shuffled batches",
            lambda: (
                None if loader is None else isinstance(loader.sampler, RandomSampler)
            ),
        ),
    )


@app.function(hide_code=True)
def check_9(dataset, text, max_length, stride):
    """Check exercise 9 by decoding the windows back to text. @ai-generated"""

    def decode(ids):
        return "".join(map(chr, ids.tolist()))

    def window(k):
        if len(dataset) <= k:
            return None
        start = k * stride
        inputs, targets = dataset[k]
        return (
            decode(inputs) == text[start : start + max_length]
            and decode(targets) == text[start + 1 : start + max_length + 1]
        )

    n_expected = len(range(0, len(text) - max_length, stride))
    result = report(
        ("first window and its target", lambda: window(0)),
        ("second window and its target", lambda: window(1)),
        (
            f"{n_expected} windows",
            lambda: None if len(dataset) == 0 else len(dataset) == n_expected,
        ),
    )
    if len(dataset) == 0:
        return result
    rows = "\n".join(
        f"| {k} | `{decode(dataset[k][0])!r}` | `{decode(dataset[k][1])!r}` |"
        for k in range(min(3, len(dataset)))
    )
    return mo.vstack(
        [result, mo.md("| window | input | target |\n|---|---|---|\n" + rows)]
    )


@app.function(hide_code=True)
def plot_decision_boundary(ax, model, X, y, lim=1.2, resolution=200):
    """Colour the plane by the class predicted by ``model`` and overlay the points ``(X, y)``. @ai-generated"""
    xs = torch.linspace(-lim, lim, resolution)
    grid_x, grid_y = torch.meshgrid(xs, xs, indexing="xy")
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    model.eval()
    with torch.no_grad():
        predictions = model(grid).argmax(dim=1)
    ax.contourf(
        grid_x,
        grid_y,
        predictions.reshape(grid_x.shape),
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=CLASS_CMAP,
        alpha=0.25,
    )
    ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=CLASS_CMAP, s=8, edgecolors="k", linewidths=0.2
    )
    ax.set_aspect("equal")


@app.function(hide_code=True)
def check_8(model, losses, X_val, y_val):
    """
    Check exercise 8, with two independent pieces of feedback:

    - the validation accuracy and the decision regions of the model, as soon
      as it has been trained (i.e. one of its parameters has a gradient);
    - the training loss curve, only if ``losses`` has been recorded.

    @ai-generated
    """
    trained = any(p.grad is not None for p in model.parameters())
    if trained:
        model.eval()
        with torch.no_grad():
            accuracy = (
                (model(X_val).argmax(dim=1) == y_val).to(torch.float32).mean().item()
            )
        accuracy_check = (
            f"validation accuracy above 90 % (got {accuracy:.1%})",
            lambda: accuracy > 0.9,
        )
        fig, ax = plt.subplots(figsize=(4, 3.4), constrained_layout=True)
        plot_decision_boundary(ax, model, X_val, y_val)
        ax.set_title("Decision regions (validation set)")
        accuracy_part = mo.vstack([report(accuracy_check), fig])
    else:
        accuracy_part = report(("validation accuracy above 90 %", lambda: None))

    if losses:
        fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
        ax.plot(losses, lw=0.8)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("training loss")
        ax.set_title(f"Training loss (last: {losses[-1]:.3f})")
        loss_part = fig
    else:
        loss_part = mo.callout(
            mo.md(
                "**Training loss:** append `loss.item()` to `losses` at each step to see the training curve."
            ),
            kind="neutral",
        )
    return mo.hstack(
        [accuracy_part, loss_part], justify="start", align="start", gap=2, widths=[1, 1]
    )


@app.class_definition(hide_code=True)
class CharTokenizer:
    """
    Character-level tokenizer exposing the same ``encode``/``decode``
    interface as the ``tiktoken`` tokenizer used in the book.
    """

    def __init__(self, text):
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        """Map characters to a 1-D int64 tensor of ids, skipping unknown characters. @ai-generated"""
        return torch.tensor(
            [self.stoi[ch] for ch in text if ch in self.stoi], dtype=torch.long
        )

    def decode(self, ids):
        """Map a sequence (list or 1-D tensor) of ids back to a string. @ai-generated"""
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        return "".join(self.vocab[i] for i in ids)


@app.class_definition(hide_code=True)
class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention, following the ``CausalAttention`` class of chapter 3. @ai-generated"""

    def __init__(self, d_in, d_out, context_length, dropout=0.0):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).to(
                torch.bool
            ),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries, keys, values = self.W_query(x), self.W_key(x), self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return self.dropout(attn_weights) @ values


@app.class_definition(hide_code=True)
class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block (attention and feed-forward, with residual connections), as in chapter 4. @ai-generated"""

    def __init__(self, emb_dim=64, context_length=64, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.att = CausalSelfAttention(emb_dim, emb_dim, context_length, drop_rate)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), nn.GELU(), nn.Linear(4 * emb_dim, emb_dim)
        )
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        return x + self.drop_shortcut(self.ff(self.norm2(x)))


@app.class_definition(hide_code=True)
class TinyGPT(nn.Module):
    """
    Miniature version of the book's ``GPTModel`` (chapter 4), with single-head attention.
    The book's configuration dictionary is replaced by keyword arguments.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 64,
        emb_dim: int = 64,
        n_layers: int = 1,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(emb_dim, context_length, drop_rate)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        x = self.tok_emb(in_idx) + self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = self.trf_blocks(self.drop_emb(x))
        return self.out_head(self.final_norm(x))


@app.function(hide_code=True)
def check_10(loss, logits, targets, vocab_size):
    """Check exercise 10. @ai-generated"""
    ref = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return report(
        ("a scalar loss", lambda: None if loss is None else loss.ndim == 0),
        (
            "the cross-entropy over every position",
            lambda: None if loss is None else torch.allclose(loss, ref),
        ),
        (
            f"close to $\\ln({vocab_size}) = {math.log(vocab_size):.2f}$"
            + ("" if loss is None else f" (got {loss.item():.2f})"),
            lambda: (
                None if loss is None else abs(loss.item() - math.log(vocab_size)) < 0.5
            ),
        ),
    )


@app.function(hide_code=True)
def lm_loss(model, inputs, targets):
    """Cross-entropy of a batch of next-token predictions, as ``calc_loss_batch`` in chapter 5. @ai-generated"""
    logits = model(inputs.to(device))
    return F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())


@app.function(hide_code=True)
def train_lm(model, train_loader, val_loader, num_steps, lr):
    """
    Train a language model with AdamW for ``num_steps`` steps, cycling
    through ``train_loader``, and return the final training and validation
    losses, each estimated over 10 batches.

    @ai-generated
    """
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    batches = iter(train_loader)
    for _ in mo.status.progress_bar(
        range(num_steps), title="Training TinyGPT", remove_on_exit=True
    ):
        try:
            inputs, targets = next(batches)
        except StopIteration:  # end of an epoch: start a new pass over the data
            batches = iter(train_loader)
            inputs, targets = next(batches)
        loss = lm_loss(model, inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return {
            name: sum(lm_loss(model, x, y).item() for x, y in islice(loader, 10))
            / min(10, len(loader))
            for name, loader in (("train", train_loader), ("validation", val_loader))
        }


@app.function(hide_code=True)
def check_11(generate_fn, model, tokenizer, context_size):
    """
    Check exercise 11: shapes, greedy decoding against a reference implementation,
    cropping, and that `temperature=1.0` actually samples (repeated runs differ).

    @ai-generated
    """
    model_device = next(model.parameters()).device
    prompt = tokenizer.encode("ROMEO:").unsqueeze(0).to(model_device)
    long_prompt = (
        tokenizer.encode(("To be, or not to be: " * 10)[:100])
        .unsqueeze(0)
        .to(model_device)
    )

    def reference(idx, n):
        for _ in range(n):
            with torch.no_grad():
                logits = model(idx[:, -context_size:])[:, -1, :]
            idx = torch.cat((idx, logits.argmax(dim=-1, keepdim=True)), dim=1)
        return idx

    def appends(idx, n, **kwargs):
        out = generate_fn(model, idx, n, context_size, **kwargs)
        return None if out.shape == idx.shape else out.shape == (1, idx.shape[1] + n)

    def samples(n_runs=5):
        # Unseeded on purpose: five identical 20-token samples at T=1 are
        # vanishingly unlikely, and seeding would reset the notebook's RNG.
        shape_ok = appends(prompt, 20, temperature=1.0)
        if not shape_ok:
            return shape_ok
        outs = [
            generate_fn(model, prompt, 20, context_size, temperature=1.0)
            for _ in range(n_runs)
        ]
        return any(not torch.equal(out, outs[0]) for out in outs[1:])

    return report(
        ("appends `max_new_tokens` tokens", lambda: appends(prompt, 20)),
        (
            "greedy decoding picks the most likely tokens",
            lambda: (
                None
                if appends(prompt, 20) is None
                else torch.equal(
                    generate_fn(model, prompt, 20, context_size), reference(prompt, 20)
                )
            ),
        ),
        ("crops prompts longer than `context_size`", lambda: appends(long_prompt, 5)),
        (
            "samples with `temperature=1.0`",
            samples,
        ),
    )


if __name__ == "__main__":
    app.run()
