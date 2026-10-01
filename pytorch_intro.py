"""
An introduction to PyTorch: the exercises of a 90-minute hands-on session for
readers of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.
The concepts are presented with the slides (`slides/slides.typ`).

Launch with:  uv run marimo edit pytorch_intro.py

@ai-generated
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
    re-runs. Names starting with an underscore (`_x`, `_model`) are **local to their cell**, which lets every
    exercise reuse the same names. Otherwise, variables cannot be reused across cells.
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
        _a = _x.permute(2, 0, 1)
        _b = _x.reshape(6, 4)  # or _x.view(6, 4), or _x.flatten(0, 1)
        _c = _x.unsqueeze(1)  # or _x[:, None]
        _d = _x.to(torch.float32).mean(dim=-1, keepdim=True)
        ```

        `mean` is only defined for floating-point tensors: on int64, it raises a `RuntimeError`.
        """,
    )
    return


@app.cell
def _():
    _x = torch.arange(24).reshape(2, 3, 4)

    _a = _x.permute(2, 0, 1)  # TODO: shape (4, 2, 3)
    _b = _x.view(6, -1)  # TODO: shape (6, 4)
    _c = _x.unsqueeze(1)  # TODO: shape (2, 1, 3, 4)
    _d = _x.to(torch.float32).mean(dim=-1, keepdim=True)  # TODO: shape (2, 3, 1)

    check_1(_a, _b, _c, _d)
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
        _x = torch.tensor(2.0, requires_grad=True)
        _y = _x**3 + 2 * _x
        _y.backward()
        _grad = _x.grad  # tensor(14.)
        ```

        Recompute `_y` and call `_y.backward()` once more: `_x.grad` becomes 28, because gradients **accumulate**.
        """,
    )
    return


@app.cell
def _():
    _x = torch.tensor(2.0)
    _x.requires_grad = True # TODO: ask PyTorch to track gradients
    _y = _x**3 + 2*_x  # TODO: y = x³ + 2x
    # TODO: backpropagate from y
    _y.backward()
    _grad = _y.grad  # TODO: read dy/dx

    check_3(_x, _grad)
    return


@app.cell(hide_code=True)
def _():
    exercise(
        4,
        "Gradient descent by hand",
        r"""
        The data below follows $y = 3x + 1$, plus noise. Fit $\hat y = w x + b$ by minimising the mean squared error,
        with plain tensors: no `torch.nn`, no optimizer. At each step:

        1. compute the loss;
        2. backpropagate;
        3. update `w` and `b` with `lr * grad`, **without recording the operation in the graph**;
        4. reset the gradients.
        """,
        r"""
        ```python
        for _step in range(200):
            _loss = ((_w * _x + _b - _y) ** 2).mean()
            _loss.backward()
            with torch.no_grad():
                _w -= _lr * _w.grad
                _b -= _lr * _b.grad
            _w.grad.zero_()
            _b.grad.zero_()
            _losses.append(_loss.item())
        ```

        This is exactly what `torch.optim.SGD` does in `optimizer.step()` and `optimizer.zero_grad()`.
        """,
    )
    return


@app.cell
def _():
    torch.manual_seed(0)
    _x = torch.linspace(-1, 1, 50)
    _y = 3 * _x + 1 + 0.1 * torch.randn(50)

    _w = torch.zeros(1, requires_grad=True)
    _b = torch.zeros(1, requires_grad=True)
    _lr = 0.1
    _losses = []
    for _step in range(200):
        _loss = torch.tensor(0.0)  # TODO: the mean squared error of w * x + b
        # TODO: backpropagate
        # TODO: update w and b without recording the graph
        # TODO: reset the gradients
        _losses.append(_loss.item())

    check_4(_x, _y, _w, _b, _losses)
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

        _n_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)  # 2,213
        ```

        $(50 \times 30 + 30) + (30 \times 20 + 20) + (20 \times 3 + 3) = 1530 + 620 + 63 = 2213$:
        each `nn.Linear` has a weight matrix and a bias vector.
        """,
    )
    return


@app.cell
def _():
    class NeuralNetwork(nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()
            self.layers = nn.Sequential(
                # TODO: 1st hidden layer (30 units) and its ReLU
                # TODO: 2nd hidden layer (20 units) and its ReLU
                # TODO: output layer
            )

        def forward(self, x):
            return self.layers(x)

    torch.manual_seed(123)
    _model = NeuralNetwork(50, 3)
    _n_params = None  # TODO: count the trainable parameters

    check_5(_model, _n_params)
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
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            # TODO: the trainable `scale` and `shift` parameters

        def forward(self, x):
            return x  # TODO: normalise over the last dimension, then scale and shift

    check_6(MyLayerNorm)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 4 · `Dataset` and `DataLoader`

    Exercises 7 and 9 use a 2-D classification problem that a linear model **cannot** solve: three interleaved spirals.
    """)
    return


@app.cell
def _():
    X_train, y_train, X_val, y_val = make_spirals()
    print(
        f"X_train: {tuple(X_train.shape)} {X_train.dtype} | y_train: {tuple(y_train.shape)} {y_train.dtype}"
    )
    print(f"X_val  : {tuple(X_val.shape)} | y_val: {tuple(y_val.shape)}")

    _fig, _ax = plt.subplots(figsize=(3.6, 3.6), constrained_layout=True)
    _ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=CLASS_CMAP, s=8)
    _ax.set_aspect("equal")
    _ax.set_title("Training set")
    _fig
    return X_train, X_val, y_train, y_val


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

        _train_loader = DataLoader(_train_ds, batch_size=32, shuffle=True)
        ```

        For tensors already in memory, `TensorDataset(X, y)` does the same job.
        """,
    )
    return


@app.cell
def _(X_train, y_train):
    class ToyDataset(Dataset):
        def __init__(self, X, y):
            self.features = X
            self.labels = y

        def __getitem__(self, index):
            return None  # TODO: the (features, label) pair of sample `index`

        def __len__(self):
            return 0  # TODO: the number of samples

    _train_ds = ToyDataset(X_train, y_train)
    _train_loader = None  # TODO: shuffled batches of 32 samples

    check_7(_train_ds, _train_loader, X_train, y_train)
    return


@app.cell(hide_code=True)
def _():
    exercise(
        8,
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
            self.input_ids.append(token_ids[i : i + max_length])
            self.target_ids.append(token_ids[i + 1 : i + max_length + 1])
        ```

        Every position of a window is a training example: after `"I HAD al"`, predict `"w"`. Chapter 5 trains with
        `stride = max_length`, so that the windows do not overlap.
        """,
    )
    return


@app.cell
def _():
    class NextTokenDataset(Dataset):
        def __init__(self, token_ids, max_length, stride):
            self.input_ids = []
            self.target_ids = []
            # TODO: a window of `max_length` tokens every `stride` tokens, and its target shifted by one

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    _text = "I HAD always thought Jack Gisburn rather a cheap genius"
    _token_ids = torch.tensor(
        [ord(_ch) for _ch in _text]
    )  # a naive tokenizer: one id per character
    _dataset = NextTokenDataset(_token_ids, max_length=8, stride=4)

    check_8(_dataset, _text, max_length=8, stride=4)
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
        9,
        "Train a classifier",
        r"""
        Train the multilayer perceptron below on the spirals: complete the **five steps** of the training loop
        (forward pass, loss, reset the gradients, backward pass, update), and record `loss.item()` at each step.
        Goal: a validation accuracy above **90 %**.

        Then experiment: a learning rate of 0.5, or a linear model (`nn.Linear(2, 3)` only).
        """,
        r"""
        ```python
        for _features, _labels in _train_loader:
            _logits = _model(_features)  # 1. forward
            _loss = F.cross_entropy(_logits, _labels)  # 2. loss
            _optimizer.zero_grad()  # 3. reset the gradients
            _loss.backward()  # 4. backward
            _optimizer.step()  # 5. update
            _losses.append(_loss.item())
        ```

        `F.cross_entropy` takes the **raw logits**: no softmax. A linear model gets stuck below 50 %: its decision
        regions can only be separated by straight lines.
        """,
    )
    return


@app.cell
def _(X_train, X_val, y_train, y_val):
    torch.manual_seed(123)
    _model = nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3)
    )
    _optimizer = torch.optim.Adam(_model.parameters(), lr=0.01)
    _train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    _losses = []

    for _epoch in range(100):
        _model.train()
        for _features, _labels in _train_loader:
            pass  # TODO: the five steps, then append loss.item() to _losses

    check_9(_model, _losses, X_val, y_val)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 6 · Towards LLMs: a tiny character-level GPT

    We train a (very) small GPT on Tiny Shakespeare, with **one token per character** (the book uses GPT-2's
    tokenizer, with 50,257 tokens). The model, `TinyGPT`, follows the architecture of chapter 4: you do not need to
    understand attention today. Its code, and that of the training function, is in the appendix.
    """)
    return


@app.cell
def _():
    corpus = (DATA_DIR / "tinyshakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(corpus)
    GPT_CONFIG = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": 64,
        "emb_dim": 64,
        "n_layers": 1,
        "drop_rate": 0.0,
    }

    # Windows of context_length + 1 tokens: the same as NextTokenDataset with stride = max_length
    _windows = tokenizer.encode(corpus).unfold(0, 65, 64)
    _split = int(0.9 * len(_windows))  # 90 % training, 10 % validation, as in chapter 5
    lm_train_loader = DataLoader(
        TensorDataset(_windows[:_split, :-1], _windows[:_split, 1:]),
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    lm_val_loader = DataLoader(
        TensorDataset(_windows[_split:, :-1], _windows[_split:, 1:]), batch_size=32
    )

    print(f"{len(corpus):,} characters, vocabulary of {tokenizer.vocab_size} tokens")
    print(
        f"{len(lm_train_loader)} training batches, {len(lm_val_loader)} validation batches"
    )
    return GPT_CONFIG, lm_train_loader, lm_val_loader, tokenizer


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
        _loss = F.cross_entropy(_logits.flatten(0, 1), _targets.flatten())
        ```

        After flattening, every position of every sequence is a classification example: `(32 × 64, 65)` logits
        and `(32 × 64,)` targets. Chapter 5 does exactly this in `calc_loss_batch`.
        """,
    )
    return


@app.cell
def _(GPT_CONFIG, lm_train_loader):
    torch.manual_seed(123)
    _model = TinyGPT(GPT_CONFIG)
    _inputs, _targets = next(iter(lm_train_loader))  # (32, 64) and (32, 64)
    _logits = _model(_inputs)  # (32, 64, 65)

    _loss = None  # TODO

    check_10(_loss, _logits, _targets, GPT_CONFIG["vocab_size"])
    return


@app.cell
def _(GPT_CONFIG, lm_train_loader, lm_val_loader):
    torch.manual_seed(123)
    lm_model = TinyGPT(GPT_CONFIG)
    _start = time.perf_counter()
    _losses = train_lm(
        lm_model, lm_train_loader, lm_val_loader, num_steps=1000, lr=0.01
    )
    mo.md(
        f"Trained `TinyGPT` ({sum(p.numel() for p in lm_model.parameters()):,} parameters) "
        f"in {time.perf_counter() - _start:.1f} s on `{device}`: training loss **{_losses['train']:.2f}**, "
        f"validation loss **{_losses['validation']:.2f}**."
    )
    return (lm_model,)


@app.cell(hide_code=True)
def _():
    exercise(
        11,
        "Text generation (book, chapter 5)",
        r"""
        Complete `generate`, which extends the token ids `idx` (shape `(batch, tokens)`) by `max_new_tokens` tokens.
        At each step:

        1. crop `idx` to its last `context_size` tokens (the model cannot see more);
        2. compute the logits, without recording the graph;
        3. keep the logits of the **last** position: `(batch, vocab_size)`;
        4. pick the next token: the most likely one (`argmax`) if `temperature == 0`, otherwise sample it from
           `softmax(logits / temperature)` with `torch.multinomial`;
        5. append it to `idx`.

        Then play with the prompt and the temperature in the cell after the check.
        """,
        r"""
        ```python
        def generate(model, idx, max_new_tokens, context_size, temperature=0.0):
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
def _(GPT_CONFIG, lm_model, tokenizer):
    def generate(model, idx, max_new_tokens, context_size, temperature=0.0):
        for _ in range(max_new_tokens):
            pass  # TODO: the five steps of the statement
        return idx

    check_11(generate, lm_model, tokenizer, GPT_CONFIG["context_length"])
    return (generate,)


@app.cell
def _(GPT_CONFIG, generate, lm_model, tokenizer):
    torch.manual_seed(123)
    _prompt = "ROMEO:"
    _idx = (
        tokenizer.encode(_prompt).unsqueeze(0).to(device)
    )  # add the batch dimension: (1, tokens)
    _out = generate(
        lm_model,
        _idx,
        max_new_tokens=300,
        context_size=GPT_CONFIG["context_length"],
        temperature=0.8,
    )
    mo.callout(mo.plain_text(tokenizer.decode(_out[0].cpu())), kind="info")
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

    The cells below define the checks of the exercises, the data, and the `TinyGPT` model. Their code is hidden;
    you do not need to read it.
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
def check_4(x, y, w, b, losses):
    """Check exercise 4 and, once attempted, plot the fit and the loss curve. @ai-generated"""
    untouched = w.item() == 0.0 and b.item() == 0.0
    result = report(
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
def check_8(dataset, text, max_length, stride):
    """Check exercise 8 by decoding the windows back to text. @ai-generated"""

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
def check_9(model, losses, X_val, y_val):
    """Check exercise 9 and, once trained, plot the loss and the decision regions. @ai-generated"""
    if not losses:
        return report(("validation accuracy above 90 %", lambda: None))
    model.eval()
    with torch.no_grad():
        accuracy = (model(X_val).argmax(dim=1) == y_val).to(torch.float32).mean().item()
    result = report(
        (f"validation accuracy above 90 % (got {accuracy:.1%})", lambda: accuracy > 0.9)
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax1.plot(losses, lw=0.8)
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("training loss")
    plot_decision_boundary(ax2, model, X_val, y_val)
    ax2.set_title("Decision regions (validation set)")
    return mo.vstack([result, fig])


@app.class_definition(hide_code=True)
class CharTokenizer:
    """
    Character-level tokenizer exposing the same ``encode``/``decode``
    interface as the ``tiktoken`` tokenizer used in the book.

    @ai-generated
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

    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.norm1 = nn.LayerNorm(emb_dim)
        self.att = CausalSelfAttention(
            emb_dim, emb_dim, cfg["context_length"], cfg["drop_rate"]
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), nn.GELU(), nn.Linear(4 * emb_dim, emb_dim)
        )
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        return x + self.drop_shortcut(self.ff(self.norm2(x)))


@app.class_definition(hide_code=True)
class TinyGPT(nn.Module):
    """
    Miniature version of the book's ``GPTModel`` (chapter 4), configured with
    the same kind of dictionary, but with single-head attention.

    @ai-generated
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

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
    """Check exercise 11 against a reference implementation of greedy decoding. @ai-generated"""
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
            lambda: appends(prompt, 20, temperature=1.0),
        ),
    )


if __name__ == "__main__":
    app.run()
