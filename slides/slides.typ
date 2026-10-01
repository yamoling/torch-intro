// Slides accompanying the marimo notebook `pytorch_intro.py`.
// Compile with:  typst compile slides/slides.typ
#import "@preview/fletcher:0.5.6" as fletcher: diagram, edge, node

// ------------------------------------------------------------------ theme
#let torch-orange = rgb("#EE4C2C")
#let ink = rgb("#1F2330")
#let muted = rgb("#6B7080")
#let soft = rgb("#F7F5F3")
#let blue = rgb("#2F6FB3")
#let green = rgb("#2E8B57")

#set document(title: "An introduction to PyTorch", author: "")
#set page(paper: "presentation-16-9", margin: (x: 1.6cm, top: 1.9cm, bottom: 1.1cm))
#set text(font: "Inter", size: 19pt, fill: ink)
#set par(leading: 0.7em)
#set list(marker: text(fill: torch-orange)[▸], spacing: 0.75em)
#set enum(spacing: 0.75em)
#show raw: set text(font: "DejaVu Sans Mono")
#show raw.where(block: true): set text(size: 14pt)
#show raw.where(block: false): set text(size: 0.9em)
#show raw.where(block: true): it => block(
  fill: soft,
  inset: (x: 12pt, y: 10pt),
  radius: 6pt,
  width: 100%,
  stroke: (left: 3pt + torch-orange),
  it,
)
#show raw.where(block: false): it => box(
  fill: luma(120).transparentize(88%),
  inset: (x: 3pt),
  outset: (y: 3pt),
  radius: 3pt,
  it,
)
#show strong: set text(fill: torch-orange)
#set table(stroke: (x, y) => if y == 0 { (bottom: 1pt + ink) } else { (bottom: 0.4pt + luma(220)) }, inset: 7pt)
#show table.cell.where(y: 0): set text(weight: "bold")

#let section-state = state("section", "")

// A regular slide with a title bar, the current section, and a page counter.
#let slide(title, body, fill: auto) = {
  pagebreak(weak: true)
  set page(
    fill: fill,
    header: context {
      set text(size: 11pt, fill: muted)
      grid(
        columns: (1fr, auto),
        align(left, section-state.get()), align(right, [An introduction to PyTorch]),
      )
      v(-6pt)
      line(length: 100%, stroke: 0.6pt + luma(210))
    },
    footer: context {
      set text(size: 11pt, fill: muted)
      align(right, counter(page).display())
    },
  )
  text(size: 28pt, weight: "bold", title)
  v(0.4em)
  body
}

// A section divider, with a cue to switch to the notebook.
#let section(number, title, subtitle: none, notebook: none, book: none) = {
  pagebreak(weak: true)
  section-state.update([§ #number · #title])
  set page(fill: ink, header: none, footer: none)
  set text(fill: white)
  v(1fr)
  text(size: 90pt, weight: "black", fill: torch-orange)[#number]
  v(-1.2em)
  text(size: 40pt, weight: "bold", title)
  if subtitle != none {
    v(-0.4em)
    text(size: 20pt, fill: luma(200), subtitle)
  }
  v(1fr)
  set text(size: 15pt, fill: luma(200))
  grid(
    columns: (1fr, auto),
    if book != none [📖 Book: #book],
    if notebook != none {
      box(fill: torch-orange, inset: (x: 12pt, y: 8pt), radius: 5pt, text(
        fill: white,
        weight: "bold",
      )[→ notebook § #notebook])
    },
  )
}

#let callout(body, color: torch-orange, title: none) = block(
  fill: color.lighten(90%),
  stroke: (left: 4pt + color),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
  {
    if title != none { text(weight: "bold", fill: color, title) + linebreak() }
    body
  },
)

#let pill(body, color: torch-orange) = box(
  fill: color.lighten(85%),
  stroke: 0.8pt + color,
  inset: (x: 8pt, y: 4pt),
  radius: 12pt,
  text(size: 0.8em, fill: color.darken(20%), body),
)

// An exercise to solve in the notebook; its solution goes on the next slide.
#let exercise(number, title, minutes: 5, body) = slide(
  fill: torch-orange.lighten(95%),
  [#text(fill: torch-orange)[✍️ Exercise #number] · #title],
  {
    body
    v(1fr)
    align(right, pill[notebook · exercise #number · ≈ #minutes min])
  },
)

#let solution(number, title, body) = slide(
  fill: green.lighten(95%),
  [#text(fill: green)[💡 Solution #number] · #title],
  body,
)

// ------------------------------------------------------------------ title
#{
  set page(fill: ink, header: none, footer: none)
  set text(fill: white)
  v(1fr)
  text(size: 16pt, fill: torch-orange, weight: "bold", tracking: 2pt)[HANDS-ON SESSION · 90 MIN]
  v(-0.3em)
  text(size: 46pt, weight: "black")[An introduction to PyTorch]
  v(-0.6em)
  text(size: 22pt, fill: luma(210))[Getting ready for _Build a Large Language Model (From Scratch)_]
  v(0.4em)
  text(size: 16pt, fill: luma(170))[Sebastian Raschka · Manning, 2024 · Appendix A]
  v(1fr)
  text(size: 14pt, fill: luma(170))[Notebook: `uv run marimo edit pytorch_intro.py`]
}

// ------------------------------------------------------------------ agenda
#slide[Today's plan][
  #grid(
    columns: (1.25fr, 1fr),
    gutter: 1.2cm,
    table(
      columns: (auto, 1fr, auto),
      align: (center, left, right),
      table.header[§][Topic][Time],
      [0], [Why PyTorch?], [5'],
      [1], [Tensors: NumPy with superpowers], [20'],
      [2], [Autograd: gradients for free], [15'],
      [3], [`nn.Module`: building models], [10'],
      [4], [`Dataset` and `DataLoader`], [10'],
      [5], [The training loop], [15'],
      [6], [Towards LLMs: a tiny GPT], [10'],
      [7], [GPUs and wrap-up], [5'],
    ),
    [
      *You know:* Python, NumPy arrays, broadcasting.

      *You will be able to:* read and write every line of PyTorch code in the book.

      #v(0.5em)
      #callout(title: [Format])[
        Short slides introduce each concept, then you solve a *short exercise* in the marimo notebook
        (11 in total). The solution is on the next slide.
      ]
    ],
  )
]

// ------------------------------------------------------------------ § 0
#section("0", "Why PyTorch?", subtitle: [What the library gives us], book: [A.1])

#slide[PyTorch is three libraries in one][
  #let pillar(icon, title, body, color) = block(
    width: 100%,
    height: 8.5cm,
    fill: color.lighten(92%),
    stroke: (top: 5pt + color),
    inset: 14pt,
    radius: 4pt,
    [
      #text(size: 30pt, icon)
      #v(-0.3em)
      #text(size: 21pt, weight: "bold", fill: color, title)
      #v(-0.2em)
      #set text(size: 16pt)
      #body
    ],
  )
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.8cm,
    pillar(
      "🧮",
      [Tensor library],
      [NumPy-like n-d arrays that also run on *GPUs*. \ `torch.tensor`, `@`, `softmax`, …],
      torch-orange,
    ),
    pillar(
      "∂",
      [Autograd],
      [Records operations and computes *gradients* automatically. \ `requires_grad`, `.backward()`],
      blue,
    ),
    pillar(
      "🧱",
      [Deep-learning toolbox],
      [Layers, losses, optimizers, data loading. \ `torch.nn`, `torch.optim`, `torch.utils.data`],
      green,
    ),
  )
]

#slide[Where PyTorch shows up in the book][
  #set text(size: 16pt)
  #align(center, diagram(
    spacing: (9mm, 12mm),
    node-corner-radius: 5pt,
    node-inset: 9pt,
    node((0, 0), [*ch. 2* \ Text data], fill: soft, stroke: 1pt + torch-orange, name: <c2>),
    node((1, 0), [*ch. 3* \ Attention], fill: soft, stroke: 1pt + torch-orange, name: <c3>),
    node((2, 0), [*ch. 4* \ GPT model], fill: soft, stroke: 1pt + torch-orange, name: <c4>),
    node((3, 0), [*ch. 5* \ Pre-training], fill: soft, stroke: 1pt + torch-orange, name: <c5>),
    node((4, 0), [*ch. 6–7* \ Fine-tuning], fill: soft, stroke: 1pt + torch-orange, name: <c6>),
    edge(<c2>, <c3>, "-|>"),
    edge(<c3>, <c4>, "-|>"),
    edge(<c4>, <c5>, "-|>"),
    edge(<c5>, <c6>, "-|>"),
    node((0, 1), [`Dataset` \ `DataLoader` \ `nn.Embedding`], stroke: none),
    node((1, 1), [`@`, `transpose` \ `triu`, `masked_fill` \ `softmax`], stroke: none),
    node((2, 1), [`nn.Module` \ `nn.Linear` \ `LayerNorm`], stroke: none),
    node((3, 1), [autograd \ `cross_entropy` \ `AdamW`], stroke: none),
    node((4, 1), [`requires_grad_` \ `.to(device)` \ `no_grad`], stroke: none),
  ))
  #v(0.5em)
  #callout[Appendix A of the book is the PyTorch primer; today follows it, then *goes a step further* by training a tiny GPT.]
]

// ------------------------------------------------------------------ § 1
#section("1", "Tensors", subtitle: [NumPy with superpowers], book: [A.2])

#slide[A tensor is an `ndarray` with two extra powers][
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      - Same ideas: *shape*, *dtype*, indexing, broadcasting, vectorized ops.
      - *Power 1:* a `device` (CPU, CUDA GPU, Apple MPS).
      - *Power 2:* `requires_grad`: it can record its history for autograd.

      #v(0.4em)
      In an LLM, almost every tensor has shape \
      #align(center, text(size: 22pt)[`(batch, tokens, emb_dim)`])
    ],
    [
      ```python
      import torch
      x = torch.tensor([[1., 2.], [3., 4.]])
      x.shape    # torch.Size([2, 2])
      x.dtype    # torch.float32
      x.device   # device(type='cpu')
      x.ndim     # 2

      torch.zeros(2, 3)
      torch.arange(5)
      torch.randn(4, 8, 768)
      ```
    ],
  )
]

#slide[NumPy → PyTorch][
  #set text(size: 15.5pt)
  #table(
    columns: (1fr, 1.25fr, 1fr),
    table.header[NumPy][PyTorch][Note],
    [`np.array(x)`], [`torch.tensor(x)`], [copies],
    [`a.sum(axis=0)`], [`t.sum(dim=0)`], [*`dim`*, not `axis`],
    [`a.astype(np.float32)`], [`t.to(torch.float32)`], [or `t.float()`],
    [`a.reshape(3, 2)`], [`t.reshape(3, 2)`, `t.view(3, 2)`], [`view` never copies],
    [`np.transpose(a, (2, 0, 1))`], [`t.permute(2, 0, 1)`, `t.transpose(1, 2)`], [`transpose`: 2 dims],
    [`a[None]`, `np.expand_dims`], [`t.unsqueeze(0)` / `t.squeeze(0)`], [],
    [`np.concatenate`, `np.stack`], [`torch.cat`, `torch.stack`], [],
    [`a @ b`], [`t @ u`], [batched matrix multiplication],
    [`a.copy()`], [`t.clone()`], [],
    [`np.random.seed(0)`], [`torch.manual_seed(0)`], [],
    [—], [`t.to("cuda")`, `t.to("cuda:0")`, `t.detach()`], [the superpowers],
  )
]

#slide[Five tensor gotchas][
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.9cm,
    [
      + *float32* by default (NumPy: float64). \ Mixing them → `RuntimeError`.
      + Labels and token ids must be *int64* (`torch.long`).
      + `torch.from_numpy` *shares* memory; `torch.tensor` copies.
      + `view` needs contiguous memory: after `transpose`, use `.contiguous().view()` or `reshape`.
      + A trailing underscore means *in place*: `add_`, `masked_fill_`, `zero_`.
    ],
    [
      ```python
      from torch.nn import Linear
      a = np.random.rand(4, 50)  # float64
      x = torch.from_numpy(a)
      model = Linear(50, 10)
      model(x)                    # 💥
      model(x.to(torch.float32))  # ✓

      y = x.transpose(0, 1)
      y.view(-1)               # 💥
      y.contiguous().view(-1)  # ✓
      y.reshape(-1)            # ✓
      ```
    ],
  )
]

#slide[Shape gymnastics][
  // Each element of `t = torch.arange(6).reshape(2, 3)` keeps its colour, so one can follow where it goes.
  #let palette = ("#F4A3A3", "#F7C59F", "#F3E79B", "#B5E2B4", "#A9CBEF", "#CDB4E8").map(rgb)
  #let lbl(label, fill) = (label: label, fill: fill)
  #let cell(v) = if type(v) == int { cell(lbl(str(v), palette.at(v))) } else {
    let (label, fill) = (v.label, v.fill)
    box(
      width: 0.62cm,
      height: 0.62cm,
      fill: fill,
      stroke: 0.6pt + luma(150),
      radius: 2pt,
      align(center + horizon, text(
        font: "DejaVu Sans Mono",
        size: 12pt,
        fill: if fill == ink { white } else { ink },
        label,
      )),
    )
  }
  #let mat(rows, caption, frame: false) = {
    let g = grid(columns: rows.at(0).len(), gutter: 2pt, ..rows.flatten().map(cell))
    stack(
      spacing: 5pt,
      if frame { box(stroke: (paint: muted, dash: "dashed"), inset: 4pt, radius: 4pt, g) } else { g },
      align(center, text(size: 11pt, fill: muted, caption)),
    )
  }
  #let arrow = text(fill: muted, size: 18pt)[⟶]
  #let panel(code, ..items) = block(
    width: 100%,
    height: 3.7cm,
    fill: soft,
    radius: 4pt,
    inset: 9pt,
    {
      align(center, text(size: 14pt, raw(code, lang: "python")))
      v(1fr)
      align(center, grid(columns: items.pos().len(), column-gutter: 8pt, align: horizon, ..items))
      v(1fr)
    },
  )
  #let t = ((0, 1, 2), (3, 4, 5))
  #let gray = luma(235)
  #let (T, F, neg) = (lbl("T", ink), lbl("F", white), lbl("-1", gray))

  #text(size: 15pt)[`t = torch.arange(6).reshape(2, 3)` #h(0.5em) #text(fill: muted)[(colours follow each element)]]
  #v(-0.2em)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.4cm,
    panel("t.view(3, 2)", mat(t, `(2, 3)`), arrow, mat(((0, 1), (2, 3), (4, 5)), `(3, 2)`)),
    panel("t.transpose(0, 1)", mat(t, `(2, 3)`), arrow, mat(((0, 3), (1, 4), (2, 5)), `(3, 2)`)),
    panel("t.unsqueeze(0)", mat(t, `(2, 3)`), arrow, mat(t, `(1, 2, 3)`, frame: true)),
    panel("t.sum(dim=0)", mat(t, `(2, 3)`), arrow, mat(((lbl("3", gray), lbl("5", gray), lbl("7", gray)),), `(3,)`)),
    grid.cell(colspan: 2, panel(
      "t.masked_fill(mask, -1)",
      mat(t, `t`),
      text(size: 13pt, fill: muted)[where],
      mat(((F, T, F), (T, F, T)), `mask = t % 2 == 1`),
      arrow,
      mat(((0, neg, 2), (neg, 4, neg)), `(2, 3)`),
    )),
  )
  #v(0.1em)
  #set text(size: 15pt)
  - `view` keeps the memory order; `transpose` swaps axes (its result is non-contiguous).
  - `dim=0` is the axis that *disappears*; `keepdim=True` keeps it with size 1.
]

#exercise(1, [Shape gymnastics], minutes: 4)[
  Starting from `x = torch.arange(24).reshape(2, 3, 4)`, build:
  - `a` of shape `(4, 2, 3)`: move the last dimension to the front;
  - `b` of shape `(6, 4)`: merge the first two dimensions;
  - `c` of shape `(2, 1, 3, 4)`: insert a dimension of size 1;
  - `d` of shape `(2, 3, 1)`: the mean over the last dimension, keeping that dimension.

  #callout[`x` holds integers. Does `mean` like that?]
]

#solution(1, [Shape gymnastics])[
  ```python
  x = torch.arange(24).reshape(2, 3, 4)
  a = x.permute(2, 0, 1)  # (4, 2, 3)
  b = x.reshape(6, 4)     # (6, 4); or x.view(6, 4), x.flatten(0, 1)
  c = x.unsqueeze(1)      # (2, 1, 3, 4); or x[:, None]
  d = x.to(torch.float32).mean(dim=-1, keepdim=True)  # (2, 3, 1)
  ```
  - `mean` is only defined for floating-point tensors: on int64, it raises a `RuntimeError`.
  - `permute` reorders *all* the dimensions; `transpose` swaps *two* of them.
]

#exercise(2, [Softmax by hand], minutes: 5)[
  Write `my_softmax(x)`, a softmax over the *last* dimension of `x`, without `torch.softmax`:
  $ "softmax"(x)_i = e^(x_i) / (sum_j e^(x_j)) $
  - Every row of the result must sum to 1.
  - Hint: `keepdim=True`.
]

#solution(2, [Softmax by hand])[
  ```python
  def my_softmax(x):
      e = torch.exp(x)
      return e / e.sum(dim=-1, keepdim=True)    # (..., n) / (..., 1)
  ```
  - With `keepdim=True`, the sum has shape `(…, 1)` and broadcasts over each row.
  - Without it, `(4, 5) / (4,)` fails, and `(n, n) / (n,)` *silently* divides by the wrong values!
]

// ------------------------------------------------------------------ § 2
#section("2", "Autograd", subtitle: [Gradients for free], book: [A.3–A.4])

#slide[Training = following the gradient downhill][
  #show raw.where(block: true): set text(size: 13pt)
  #grid(
    columns: (1fr, 1.3fr),
    gutter: 0.8cm,
    [
      $ theta <- theta - eta thin nabla_theta cal(L)(theta) $
      #v(0.3em)
      - $theta$: millions of parameters; $cal(L)$: the loss; $eta$: the learning rate.

      #callout(title: [Autograd])[
        Operations on tensors with `requires_grad=True` are recorded in a *computation graph*; `.backward()` applies
        the chain rule through it.
      ]
    ],
    [
      ```python
      w1 = torch.tensor([2.2], requires_grad=True)
      b  = torch.tensor([0.0], requires_grad=True)

      z = x1 * w1 + b
      a = torch.sigmoid(z)
      loss = F.binary_cross_entropy(a, y)

      loss.backward()
      w1.grad   # ∂L/∂w1
      b.grad    # ∂L/∂b
      ```
    ],
  )
]

#slide[The computation graph (book, figure A.8)][
  #set text(size: 17pt)
  #align(center, diagram(
    spacing: (15mm, 10mm),
    node-stroke: 1pt,
    node((0, 0), $x_1$, shape: circle, name: <x>),
    node((0, 1), $w_1$, shape: circle, fill: torch-orange.lighten(80%), stroke: 1pt + torch-orange, name: <w>),
    node((1, 0.5), $times$, shape: circle, name: <mul>),
    node((2, 1.5), $b$, shape: circle, fill: torch-orange.lighten(80%), stroke: 1pt + torch-orange, name: <b>),
    node((2, 0.5), $+$, shape: circle, name: <add>),
    node((3, 0.5), $sigma$, shape: circle, name: <sig>),
    node((4, 0.5), [BCE], corner-radius: 4pt, name: <bce>),
    node((4, 1.5), $y$, shape: circle, name: <y>),
    node((5, 0.5), $cal(L)$, shape: circle, fill: soft, name: <L>),
    edge(<x>, <mul>, "-|>"),
    edge(<w>, <mul>, "-|>"),
    edge(<mul>, <add>, "-|>"),
    edge(<b>, <add>, "-|>"),
    edge(<add>, <sig>, "-|>", $z$),
    edge(<sig>, <bce>, "-|>", $a$),
    edge(<y>, <bce>, "-|>"),
    edge(<bce>, <L>, "-|>"),
    edge(
      <L>,
      (5, -0.6),
      (1, -0.6),
      <mul>,
      "--|>",
      stroke: torch-orange,
      [`loss.backward()`: chain rule],
      label-side: left,
    ),
  ))
  #v(0.4em)
  #align(center)[
    #pill[forward: build the graph] #h(0.5em) #pill(
      color: blue,
    )[backward: $(partial cal(L)) / (partial w_1) = (a - y) x_1$] #h(0.5em)
    #pill(color: green)[leaves store `.grad`]
  ]
]

#exercise(3, [A derivative with autograd], minutes: 3)[
  Compute the derivative of $y = x^3 + 2x$ at $x = 2$ with autograd.

  By hand: $3x^2 + 2 = 14$.

  ```python
  x = torch.tensor(2.0)  # TODO: ask PyTorch to track gradients
  y = ...                # TODO: y = x³ + 2x
  # TODO: backpropagate from y
  grad = ...             # TODO: read dy/dx
  ```
]

#solution(3, [A derivative with autograd])[
  ```python
  x = torch.tensor(2.0, requires_grad=True)
  y = x**3 + 2 * x
  y.backward()
  grad = x.grad          # tensor(14.)
  ```
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    callout(title: [Try it: gradients *accumulate*])[
      Run the forward pass again, then backpropagate: `x.grad` becomes 28.
    ],
    ```python
    y = x**3 + 2 * x   # a new graph
    y.backward()
    x.grad             # tensor(28.)
    ```,
  )
  #text(
    size: 16pt,
    fill: muted,
  )[Calling `y.backward()` twice on the *same* `y` fails: the first call frees the values saved by the forward pass.]
]

#slide[Three autograd gotchas][
  #show raw.where(block: true): set text(size: 13pt)
  #grid(
    columns: (1fr, 1.15fr),
    gutter: 0.9cm,
    [
      + Gradients *accumulate*: every `.backward()` adds to `.grad` → reset them at each step.
      + `with torch.no_grad():` stops graph recording (evaluation, text generation).
      + `.detach()` cuts a tensor from the graph (needed before `.numpy()`).
    ],
    [
      ```python
      w = torch.tensor(3., requires_grad=True)
      (w ** 2).backward()  # dL/dw = 2w = 6
      (w ** 2).backward()
      w.grad               # 12 = 6 + 6!
      w.grad.zero_()       # reset

      with torch.no_grad():
          z = w * 2        # not recorded
      z.requires_grad      # False

      w.detach().numpy()
      ```
    ],
  )
]

#slide[Optimizers: the update rule, packaged][
  #show raw.where(block: true): set text(size: 13pt)
  #grid(
    columns: (1fr, 1.1fr),
    gutter: 0.9cm,
    [
      An optimizer receives the tensors to train, and updates them from their `.grad`:
      $ theta <- theta - eta thin nabla_theta cal(L) $
      - `optimizer.zero_grad()`: reset the gradients (they *accumulate*);
      - `optimizer.step()`: apply the update rule.
      #v(0.3em)
      #text(fill: muted)[`SGD` is the rule above; `Adam` and `AdamW` (used in the book) are smarter variants.]
    ],
    [
      ```python
      w = torch.zeros(1, requires_grad=True)
      optimizer = torch.optim.SGD([w], lr=0.1)

      loss = ...            # forward pass
      optimizer.zero_grad()
      loss.backward()       # fills w.grad
      optimizer.step()      # w -= 0.1 * w.grad
      ```
    ],
  )
]

#exercise(4, [Fit a line with an optimizer], minutes: 5)[
  The data follows $y = 3x + 1$, plus noise. Fit $hat(y) = w x + b$ by minimising the mean squared error, with
  `torch.optim.SGD` (learning rate 0.1).

  ```python
  w = torch.zeros(1, requires_grad=True)
  b = torch.zeros(1, requires_grad=True)
  optimizer = ...  # TODO: SGD over [w, b]
  for step in range(200):
      loss = ...   # TODO: the mean squared error of w * x + b
      # TODO: reset the gradients
      # TODO: backpropagate
      # TODO: update w and b
  ```
]

#solution(4, [Fit a line with an optimizer])[
  ```python
  optimizer = torch.optim.SGD([w, b], lr=0.1)
  for step in range(200):
      loss = ((w * x + b - y) ** 2).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  ```
  - Result: $w approx 3$, $b approx 1$.
  - With a model, the same code trains `model.parameters()` instead of `[w, b]`: this is the training loop of § 5.
]

// ------------------------------------------------------------------ § 3
#section("3", [`nn.Module`], subtitle: [Building models], book: [A.5])

#slide[Anatomy of a module][
  #show raw.where(block: true): set text(size: 12.5pt)
  #grid(
    columns: (1.15fr, 1fr),
    gutter: 0.9cm,
    [
      ```python
      from torch import nn

      class TwoLayerNet(nn.Module):
          def __init__(self, n_in, n_hidden, n_out):
              super().__init__()
              self.hidden = nn.Linear(n_in, n_hidden)
              self.output = nn.Linear(n_hidden, n_out)

          def forward(self, x):
              x = torch.relu(self.hidden(x))
              return self.output(x)   # logits

      torch.manual_seed(123)
      model = TwoLayerNet(2, 16, 3)
      logits = model(X)   # not model.forward(X)
      ```
    ],
    [
      - `__init__`: create *parameters and sub-modules*.
      - `forward`: the computation.
      - `model.parameters()` finds all parameters *recursively* → give them to the optimizer.
      - `nn.Linear(in, out)`: $y = x W^top + b$, applied to the *last* dim: \ `(b, t, in) → (b, t, out)`.
      - `nn.Sequential(m1, nn.ReLU(), m2)` chains modules.
    ],
  )
]

#exercise(5, [The book's network (listing A.4)], minutes: 5)[
  Write `NeuralNetwork(num_inputs, num_outputs)`, a multilayer perceptron:

  #align(center, text(size: 22pt)[50 inputs → 30 (ReLU) → 20 (ReLU) → 3 logits])

  Then count its *trainable parameters* with `.parameters()` and `.numel()`.
  #v(0.3em)
  #text(fill: muted)[How many do you expect? Each `nn.Linear` has a weight matrix and a bias vector.]
]

#solution(5, [The book's network (listing A.4)])[
  #show raw.where(block: true): set text(size: 12.5pt)
  #grid(
    columns: (1.25fr, 1fr),
    gutter: 0.8cm,
    [
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

      ```
    ],
    [
      ```python
      model = NeuralNetwork(50, 3)
      sum(p.numel()
          for p in model.parameters()
          if p.requires_grad)  # 2213
      ```
      #set text(size: 17pt)
      #table(
        columns: (auto, auto, auto),
        align: (left, right, right),
        table.header[Layer][Weight][Bias],
        [50 → 30], [1500], [30],
        [30 → 20], [600], [20],
        [20 → 3], [60], [3],
        table.cell(colspan: 3, align: right)[*total: 2213*],
      )
    ],
  )
]

#slide[Parameters, buffers, and modes][
  #table(
    columns: (1.3fr, auto, auto),
    align: (left, center, center),
    table.header[Attribute][Trained][Moved by `.to()`],
    [`nn.Parameter(torch.ones(d))`], [✓], [✓],
    [`self.register_buffer("mask", …)`], [✗], [✓],
    [`self.plain = torch.zeros(d)`], [✗], [✗],
  )
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    callout(
      title: [`model.train()` / `model.eval()`],
    )[Switch layer *behaviour*: dropout is active only in training mode.],
    callout(color: blue, title: [`torch.no_grad()`])[Switches off graph *recording*. For inference, use both.],
  )
]

#exercise(6, [Your own `LayerNorm` (book, ch. 4)], minutes: 6)[
  Complete `MyLayerNorm(emb_dim)`. Over the *last* dimension, normalise `x` to zero mean and unit variance, then
  apply a trainable `scale` (ones) and `shift` (zeros):
  $ "LayerNorm"(x) = "scale" dot (x - mu) / sqrt(sigma^2 + epsilon) + "shift" $
  - Use the biased variance: `x.var(dim=-1, keepdim=True, unbiased=False)`.
  - The notebook compares your module with `nn.LayerNorm`.
]

#solution(6, [Your own `LayerNorm`])[
  #show raw.where(block: true): set text(size: 13pt)
  ```python
  class MyLayerNorm(nn.Module):
      def __init__(self, emb_dim):
          super().__init__()
          self.eps = 1e-5
          self.scale = nn.Parameter(torch.ones(emb_dim))
          self.shift = nn.Parameter(torch.zeros(emb_dim))

      def forward(self, x):
          mean = x.mean(dim=-1, keepdim=True)               # (b, t, 1)
          var = x.var(dim=-1, keepdim=True, unbiased=False)
          norm_x = (x - mean) / torch.sqrt(var + self.eps)
          return self.scale * norm_x + self.shift           # broadcasts over (b, t)
  ```
  `nn.Parameter` *registers* the tensors: they appear in `.parameters()`, so the optimizer trains them.
]

// ------------------------------------------------------------------ § 4
#section("4", [`Dataset` & `DataLoader`], subtitle: [Feeding the model in batches], book: [A.6])

#slide[Two responsibilities][
  #let stage(body, color: muted) = box(
    fill: color.lighten(92%),
    stroke: 1pt + color,
    inset: 9pt,
    radius: 4pt,
    align(center, body),
  )
  #let arrow = text(fill: muted, size: 22pt)[⟶]
  #align(center, text(size: 15pt, grid(
    columns: 7,
    column-gutter: 8pt,
    align: horizon,
    stage[Sampler \ (shuffled indices)],
    arrow,
    stage(color: torch-orange)[`dataset[i]` \ `→ (x_i, y_i)`],
    arrow,
    stage[collate \ `torch.stack`],
    arrow,
    stage(color: blue)[batch \ `X: (B, …)`, `y: (B,)`],
  )))
  #v(0.3em)
  #show raw.where(block: true): set text(size: 12.5pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      ```python
      from torch.utils.data import Dataset

      class MyDataset(Dataset):
          def __init__(self, ...):
              ...   # store or load the data

          def __getitem__(self, index):
              ...   # return sample `index`

          def __len__(self):
              ...   # return the number of samples
      ```
    ],
    [
      ```python
      train_loader = DataLoader(
          dataset=train_ds,
          batch_size=32,
          shuffle=True,    # reshuffle
          drop_last=True,  # skip runt
          num_workers=0,   # parallel
      )

      for x, y in train_loader:
          ...   # x: (32, 2), y: (32,)
      ```
    ],
  )
]

#exercise(7, [A `Dataset` and its `DataLoader` (listing A.6)], minutes: 5)[
  The data: three interleaved spirals, a 2-D problem that a linear model *cannot* solve.
  - `x_train`: `(360, 2)` float32 features; `y_train`: `(360,)` int64 labels.

  Complete `ToyDataset(x, y)`:
  - `__len__`: the number of samples;
  - `__getitem__(index)`: the pair `(features, label)` of sample `index`.

  Then wrap it in a `DataLoader` yielding *shuffled* batches of *32* samples.
]

#solution(7, [A `Dataset` and its `DataLoader`])[
  #grid(
    columns: (1.2fr, 1fr),
    gutter: 0.8cm,
    [
      ```python
      class ToyDataset(Dataset):
          def __init__(self, x, y):
              self.features = x
              self.labels = y

          def __getitem__(self, index):
              return (self.features[index],
                      self.labels[index])

          def __len__(self):
              return self.labels.shape[0]
      ```
    ],
    [
      ```python
      ds = ToyDataset(x_train, y_train)
      train_loader = DataLoader(
          ds, batch_size=32, shuffle=True)
      x, y = next(iter(train_loader))
      # x: (32, 2), y: (32,)
      ```
      #text(size: 16pt)[For tensors already in memory, `TensorDataset(x, y)` does the same job.]
    ],
  )
]

// ------------------------------------------------------------------ § 5
#section("5", "The training loop", subtitle: [Five lines to rule them all], book: [A.7])

#slide[The same loop, from appendix A to chapter 5][
  #show raw.where(block: true): set text(size: 13pt)
  #grid(
    columns: (1.35fr, 1fr),
    gutter: 0.6cm,
    [
      ```python
      model = NeuralNetwork(2, 3)
      optimizer = torch.optim.Adam(
          model.parameters(), lr=0.01)

      for epoch in range(num_epochs):
          model.train()
          for x, y in train_loader:
              logits = model(x)                  # 1
              loss = F.cross_entropy(logits, y)  # 2
              optimizer.zero_grad()              # 3
              loss.backward()                    # 4
              optimizer.step()                   # 5
          model.eval()
          with torch.no_grad():
              ...  # validation loss, accuracy
      ```
    ],
    [
      #set text(size: 15pt)
      #align(center, diagram(
        spacing: (4mm, 6.5mm),
        node-corner-radius: 4pt,
        node-inset: 7pt,
        node-stroke: 1pt + torch-orange,
        node((0, 0), [batch from `DataLoader`], name: <a>),
        node((0, 1), [1 · forward], name: <b>),
        node((0, 2), [2 · loss], name: <c>),
        node((0, 3), [3 · `zero_grad()`], name: <d>),
        node((0, 4), [4 · `backward()`], name: <e>),
        node((0, 5), [5 · `step()`], name: <f>),
        edge(<a>, <b>, "-|>"),
        edge(<b>, <c>, "-|>"),
        edge(<c>, <d>, "-|>"),
        edge(<d>, <e>, "-|>"),
        edge(<e>, <f>, "-|>"),
        edge(<f>, (0.9, 5), (0.9, 0), <a>, "-|>"),
      ))
    ],
  )
]

#slide[Loss and evaluation][
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      *`F.cross_entropy(logits, labels)`*
      - takes *raw logits* (no softmax!) and int64 class ids;
      - combines `log_softmax` and the negative log-likelihood.
      #v(0.3em)
      #callout(title: [Sanity check])[An untrained classifier over $C$ classes has a loss $approx ln C$.]
    ],
    [
      ```python
      logits = model(X)       # (B, C)
      labels = ...            # (B,) int64
      loss = F.cross_entropy(logits, labels)

      preds = logits.argmax(dim=1)
      correct = preds == labels
      acc = correct.to(torch.float32).mean()
      ```
    ],
  )
]

#exercise(8, [Train a classifier], minutes: 8)[
  Train a multilayer perceptron on the spirals. Already provided in the notebook:
  ```python
  model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                        nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3))
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  train_ds = TensorDataset(X_train, y_train)
  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
  ```
  - Write the *five steps* of the loop, and record `loss.item()` at each step.
  - Goal: a validation accuracy above *90 %*.
  - Then try a learning rate of 0.5, or a linear model (`nn.Linear(2, 3)` only).
]

#solution(8, [Train a classifier])[
  #grid(
    columns: (1.3fr, 1fr),
    gutter: 0.8cm,
    [
      ```python
      for epoch in range(100):
          model.train()
          for features, labels in train_loader:
              logits = model(features)
              loss = F.cross_entropy(logits, labels)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              losses.append(loss.item())
      ```
    ],
    [
      - ≈ 100 % validation accuracy, in about a second on a laptop CPU.
      - `lr=0.5`: the updates overshoot; accuracy stays at chance level (≈ 33 %).
      - Linear model: stuck below 50 %, since straight lines cannot separate spirals.
    ],
  )
]

// ------------------------------------------------------------------ § 6
#section("6", "Towards LLMs", subtitle: [A tiny character-level GPT, same tools], book: [ch. 2–5])

#slide[From spirals to a language model][
  #grid(
    columns: (1.2fr, 1fr),
    gutter: 0.8cm,
    [
      #set text(size: 16pt)
      #table(
        columns: (auto, 1fr, 1fr),
        table.header[][Spirals][Language model],
        [input], [2 floats], [`t` token ids (int64)],
        [1st layer], [`nn.Linear`], [`nn.Embedding`],
        [output], [3 logits], [`vocab` logits *per position*],
        [label], [class id], [next token id],
        [loss], [`cross_entropy`], [same, after `flatten`],
      )
      #v(0.3em)
      `nn.Embedding(V, d)`: a trainable lookup table \
      (= one-hot vector $times$ weight matrix).
    ],
    [
      #set text(size: 14pt)
      #align(center, diagram(
        spacing: (0mm, 4.2mm),
        node-corner-radius: 4pt,
        node-inset: 6pt,
        node-stroke: 1pt + blue,
        node((0, 0), [token ids `(b, t)`], stroke: none, name: <in>),
        node((0, 1), [`tok_emb` + `pos_emb`], name: <emb>),
        node(
          (0, 2),
          [LayerNorm → causal attention → +],
          fill: torch-orange.lighten(88%),
          stroke: 1pt + torch-orange,
          name: <att>,
        ),
        node(
          (0, 3),
          [LayerNorm → feed-forward → +],
          fill: torch-orange.lighten(88%),
          stroke: 1pt + torch-orange,
          name: <ff>,
        ),
        node((0, 4), [final LayerNorm], name: <ln>),
        node((0, 5), [`out_head` (Linear)], name: <out>),
        node((0, 6), [logits `(b, t, vocab)`], stroke: none, name: <lg>),
        edge(<in>, <emb>, "-|>"),
        edge(<emb>, <att>, "-|>"),
        edge(<att>, <ff>, "-|>"),
        edge(<ff>, <ln>, "-|>"),
        edge(<ln>, <out>, "-|>"),
        edge(<out>, <lg>, "-|>"),
      ))
      #align(center, text(fill: muted)[`TinyGPT`: ≈ 60 k parameters \ GPT-2 small: 124 M])
    ],
  )
]

#slide[Next-token prediction data (ch. 2)][
  #let cell(ch, color) = box(
    width: 1.05cm,
    height: 1.05cm,
    fill: color.lighten(85%),
    stroke: 0.8pt + color,
    radius: 3pt,
    align(center + horizon, text(font: "DejaVu Sans Mono", size: 17pt, ch)),
  )
  #let chars = "I HAD always".clusters()
  #let row(label, offset, color) = grid(
    columns: (3cm,) + (1.15cm,) * 13,
    align: horizon,
    text(size: 16pt, weight: "bold", fill: color, label),
    ..range(13).map(i => if i >= offset and i < offset + 8 { cell(chars.at(i).replace(" ", "␣"), color) } else { [] }),
  )
  #row([input], 0, blue)
  #v(0.2em)
  #row([target], 1, torch-orange)
  #v(0.6em)
  - A sliding window of `max_length` tokens, the *target* is the input *shifted by one*.
  - Every position is a training example: "after `I HAD al`, predict `w`".
  - The book's `GPTDatasetV1` is a 15-line `Dataset`; batches have shape `(batch, max_length)` and dtype int64.
]

#exercise(9, [Sliding windows (book, ch. 2)], minutes: 5)[
  Complete `NextTokenDataset(token_ids, max_length, stride)`, modelled on the book's `GPTDatasetV1`:
  - starting at positions `0, stride, 2 * stride, …`, cut windows of `max_length` tokens;
  - the *target* of each window is the same window *shifted by one* token.

  ```python
  self.input_ids, self.target_ids = [], []
  # TODO: fill both lists
  ```
  #text(fill: muted)[Careful: each window must leave room for its last target token.]
]

#solution(9, [Sliding windows])[
  ```python
  for i in range(0, len(token_ids) - max_length, stride):
      self.input_ids.append(token_ids[i : i + max_length])
      self.target_ids.append(token_ids[i + 1 : i + max_length + 1])
  ```
  - With `max_length=8` and `stride=4` on `"I HAD always thought…"`: \
    window 1 = `"D always"` → target `" always "`.
  - Chapter 5 uses `stride = max_length`: windows do not overlap, and no token is seen twice per epoch.
]

#exercise(10, [The loss of a language model], minutes: 4)[
  ```python
  logits = model(inputs)   # (batch, tokens, vocab_size) = (32, 64, 65)
  targets                  # (batch, tokens)             = (32, 64)
  ```
  But `F.cross_entropy` expects logits `(N, classes)` and targets `(N,)`.

  Compute the loss of the *untrained* `TinyGPT`.
  #v(0.3em)
  #callout(
    title: [Sanity check],
  )[An untrained model guesses uniformly: its loss should be close to $ln(65) approx 4.17$.]
]

#solution(10, [The loss of a language model])[
  ```python
  loss = F.cross_entropy(logits.flatten(0, 1),   # (32 × 64, 65)
                         targets.flatten())      # (32 × 64,)
  ```
  - After flattening, every position of every sequence is one classification example.
  - Chapter 5 does exactly this in `calc_loss_batch`.
  - Then the notebook trains `TinyGPT` for 1,000 steps (about 10 s on a laptop CPU): the loss drops from 4.2 to ≈ 1.8.
]

#slide[Generating text (ch. 5)][
  - *Autoregressive*: predict one token, append it to the input, and repeat.
  - Only the logits of the *last* position matter: `logits[:, -1, :]` has shape `(batch, vocab_size)`.
  - The model sees at most `context_size` tokens: crop the input to the last ones.
  - Picking the next token:
    - *greedy*: the most likely one, `argmax` (tends to *loop*);
    - *sampling*: draw from `softmax(logits / temperature)` with `torch.multinomial`. \
      Temperature $< 1$: sharper, safer; $> 1$: flatter, wilder.
]

#exercise(11, [Text generation (book, ch. 5)], minutes: 8)[
  Complete `generate(model, idx, max_new_tokens, context_size, temperature=0.0)`. \
  `idx` has shape `(batch, tokens)`. Repeat `max_new_tokens` times:
  + crop `idx` to its last `context_size` tokens;
  + compute the logits, *without recording the graph*;
  + keep the logits of the *last* position;
  + pick the next token: `argmax` if `temperature == 0`, otherwise sample from `softmax(logits / temperature)`;
  + append it to `idx` (`torch.cat`).

  Then generate Shakespeare from the trained `TinyGPT`, and play with the temperature.
]

#solution(11, [Text generation])[
  #show raw.where(block: true): set text(size: 12.5pt)
  ```python
  def generate(model, idx, max_new_tokens, context_size, temperature=0.0):
      for _ in range(max_new_tokens):
          idx_cond = idx[:, -context_size:]
          with torch.no_grad():
              logits = model(idx_cond)
          logits = logits[:, -1, :]  # (batch, vocab_size)
          if temperature > 0.0:
              probas = torch.softmax(logits / temperature, dim=-1)
              idx_next = torch.multinomial(probas, num_samples=1)  # (batch, 1)
          else:
              idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
          idx = torch.cat((idx, idx_next), dim=1)
      return idx
  ```
  #text(size: 17pt)[The book's version adds *top-k* sampling: keep only the $k$ most likely tokens before sampling.]
]

// ------------------------------------------------------------------ § 7
#section("7", "GPUs & wrap-up", book: [A.9])

#slide[Using a GPU: three extra lines][
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 1
  model = model.to(device)                                     # 2: parameters, buffers
  for features, labels in train_loader:
      features, labels = features.to(device), labels.to(device)  # 3: every batch
      ...
  ```
  - All tensors in an operation must be on the *same device*.
  - `.cpu()` before `.numpy()`; GPU calls are *asynchronous* (use `torch.cuda.synchronize()` when timing).
  - Apple silicon: `torch.device("mps")`. Several GPUs: `DistributedDataParallel` (A.9.3).
  - The book's code runs on a laptop CPU; a GPU just makes chapters 5–7 faster.
]

#slide[Take-aways][
  #set text(size: 17pt)
  #table(
    columns: (auto, 1fr),
    table.header[Concept][Remember],
    [Tensors], [NumPy + `dim` + devices + dtypes (float32; int64 for ids and labels)],
    [Shapes], [`view`/`reshape`, `transpose`, `unsqueeze`, `flatten`; `@` is batched],
    [Autograd], [`requires_grad` → graph → `.backward()` → `.grad`; grads *accumulate*; `no_grad()`],
    [`nn.Module`], [parameters in `__init__`, computation in `forward`, call `model(x)`; `train()`/`eval()`],
    [Data], [`Dataset` (`__len__`, `__getitem__`) + `DataLoader` (batches, shuffling)],
    [Training], [forward → loss → `zero_grad()` → `backward()` → `step()`],
  )
]

#{
  pagebreak(weak: true)
  set page(fill: ink, header: none, footer: none)
  set text(fill: white)
  v(1fr)
  text(size: 44pt, weight: "black")[Next: chapter 2]
  v(-0.6em)
  text(size: 22pt, fill: luma(210))[Working with text data: tokenization, embeddings, and data loaders]
  v(1fr)
  set text(size: 15pt, fill: luma(190))
  [Book code: `LLMs-from-scratch-main/` · PyTorch docs: #link("https://pytorch.org/docs")[pytorch.org/docs] · Tutorials: #link("https://pytorch.org/tutorials")[pytorch.org/tutorials]]
}
