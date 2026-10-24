# LaTeX source (Computers & Security submission)

Self-contained Elsevier `elsarticle` source tree. Upload this whole folder to Overleaf
(New Project → Upload Project → zip of `paper/tex/`), or build locally with a TeX Live install.

## Build
```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
(Overleaf does this automatically; `elsarticle.cls` ships with it.)

## Files
- `main.tex` — the paper. Section status:
  - **Drafted:** §2 Related Work, §3 Methodology, §4 Results (9 tables, 6 equations, 4 figures wired in).
  - **Stubs (write with co-author):** §1 Introduction, §5 Discussion, §6 Conclusion, Abstract.
    Marked in-source with `\todo{...}`.
- `references.bib` — bibliography. Entries tagged `% VERIFIED` are confirmed; `% TODO-AUTHORS`
  are secondary/survey cites whose author lists still need a final confirming pass before submission.
- `figures/` — vector PDFs (`fig1`--`fig4`), copied from `paper/figures/`. Regenerate with
  `.venv/bin/python paper/make_figures.py` (run from repo root), then re-copy the PDFs here:
  `cp paper/figures/*.pdf paper/tex/figures/`.

## Outstanding before submission
- Fill the four `\todo` prose stubs and the title/author/affiliation/DOI `\todo`s.
- Resolve the `% TODO-AUTHORS` BibTeX entries.
