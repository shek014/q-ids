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
  - **Drafted:** §2 Related Work, §3 Methodology, §4 Results (all 5 tables + both figures wired in).
  - **Stubs (write with co-author):** §1 Introduction, §5 Discussion, §6 Conclusion, Abstract.
    Marked in-source with `\todo{...}`.
- `references.bib` — bibliography. Entries tagged `% VERIFIED` are confirmed; `% TODO-AUTHORS`
  are secondary/survey cites whose author lists still need a final confirming pass before submission.
- `figures/` — vector PDFs (`fig1_evasion_curve.pdf`, `fig2_arms_race.pdf`), copied from
  `paper/figures/`. Regenerate with `.venv/bin/python paper/make_figures.py`, then re-copy here.

## Outstanding before submission
- Fill the four `\todo` prose stubs and the title/author/affiliation/DOI `\todo`s.
- Resolve the `% TODO-AUTHORS` BibTeX entries.
