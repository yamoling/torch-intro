# An introduction to PyTorch

A 90-minute hands-on session to prepare readers of _Build a Large Language Model (From Scratch)_ (S. Raschka), following its appendix A.

The slides present the concepts. At regular intervals, the audience solves a short exercise in the notebook; the next slide shows the solution.

| File                | Content                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `slides/slides.typ` | The Typst slides, with 11 exercise slides (orange), each followed by its solution (green)                          |
| `pytorch_intro.py`  | The marimo notebook: for each exercise, a statement, a cell to complete, an automatic check, and a hidden solution |
| `data/`             | `tinyshakespeare.txt` (Karpathy's char-rnn corpus), used to train a tiny GPT in exercises 10–11                    |

## Running

```sh
uv sync
uv run marimo edit pytorch_intro.py        # the exercises
just compile                               # -> slides/slides.pdf (or `just watch`)
```

Opening the notebook runs every cell, including a short training run of the tiny GPT (about 10 s on a laptop CPU).
