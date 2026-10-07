import subprocess


def run_notebook():
    subprocess.run(
        ["marimo", "edit", "pytorch_intro.py"],
        check=True,
    )
