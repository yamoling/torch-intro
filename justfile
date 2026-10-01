slides := "slides/slides.typ"

# Compile the slides to slides/slides.pdf
compile:
    typst compile {{slides}}

# Recompile the slides on every change
watch:
    typst watch {{slides}}
