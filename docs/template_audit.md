# Course report template adaptation

Official source: https://www.overleaf.com/read/swwcqhtsdwbg (resolved project 6556cf1443f932295cf9d008), read through Computer/Overleaf on 2026-11-03.

The read-only project contains `report.tex`, `reference.bib` and `style/dsc180reportstyle.sty`. Inspected the actual source editor and rendered template. The course methodology page explicitly permits changing headings; blue instructional prose must be removed. Only the Q1 code link is required, not a website.

The local `reports/latex/style/dsc180reportstyle.sty` is an attributed **typography subset**, not an unchanged archive. It retains the course title/abstract format, letter paper, 1-inch margins, 12pt article body, paragraph spacing and natbib layout. Unused demonstration, contents, algorithm and appendix helpers were omitted. TeX Gyre Pagella/Heros/Cursor replace XCharter/Arial/Fira Code for portable local compilation; standard plainnat replaces the custom bibliography style. These are disclosed formatting deviations. A final submission requiring the exact unmodified style should use the full Overleaf style and bibliography files; mentor/TA acceptance of these adaptations is not independently verified.

The direct public source-ZIP URL returned HTTP 403; the read-only source remained accessible in the browser. No template account or remote document was modified. The local report was compiled with the already installed Tectonic engine because it imports supporting files, which the single-document built-in compiler does not support. Build command is in README. The source is editable and the actual PDF is inspected after compilation.
