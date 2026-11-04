# Official template comparison

Source: the [public DSC Capstone Overleaf template](https://www.overleaf.com/read/swwcqhtsdwbg), project `6556cf1443f932295cf9d008`, opened read-only on 2026-11-03. Official instructions it contains:
- use the template;
- remove blue instruction text;
- choose headings with your mentor;
- the Quarter 1 report links the code, not a website.

## Style file: official, with one marked local block

`reports/latex/style/dsc180reportstyle.sty` was transcribed from the template's editor. To verify the transcription, remove its six-line provenance header and restore the three original font lines in place of the block marked `LOCAL`. The result is byte-identical to the official file: SHA-256 `8b6a24551500813f874ca797ffdd16426d1b00725cbed3b87d2667a96e2597f7`, computed in the browser from the official text and locally from the restored copy, 14,166 characters.

The local block changes only font loading:

| Official line | Local replacement | Reason |
|---|---|---|
| `\setmainfont{XCharter}` | `\setmainfont{XCharter-Roman.otf}[Bold/Italic/BoldItalic files]` | XeTeX outside Overleaf finds the bundled XCharter files by name |
| `\setsansfont{Arial}` | Arial if installed, else TeX Gyre Heros | Portability; Arial is used on macOS |
| `\setmonofont[Scale=0.9]{Fira Code}` | `\setmonofont{FiraMono-Regular.otf}[Scale=0.9, ...]` | Fira Code is not in the TeX bundle; Fira Mono is the same family without ligatures |

The report (`reports/latex/checkpoint_report.tex`) follows the template's structure: the title and `\And` author block with the mentor last, the abstract, a centered `Code:` link, `\maketoc`, `\clearpage`, the sections, `\makereference` and `\bibliographystyle{style/dsc180bibstyle}`. Its only additions to the template are `xeCJK` with FandolSong for Chinese terms (the official style ships this line commented out), `tikz` for Figure 1, and a one-line `\nodot` helper used by one bibliography title. The template lists an email under each name. The report omits them because none was available here; add the student's and mentor's UCSD emails if desired.

## Bibliography style: template settings on econ.bst 3.1.1

The template's `style/dsc180bibstyle.bst` is econ.bst version 2.1 by Shiro Takeda (79,783 characters), with customized formatting hooks. A file download from the read-only project was not available. Instead, `reports/latex/style/dsc180bibstyle.bst` starts from econ.bst 3.1.1 as shipped in the Tectonic TeX bundle (LPPL 1.3c). It copies the template's values for the 20 formatting hooks whose meaning is the same in both versions, including:
- bold author names;
- the `Author. Year. "Title."` layout;
- `, and` author lists, with et al. after more than four authors;
- `: pages`;
- italic book and journal titles.

The URL hook keeps 3.1.1's `\url{...}` wrapper with the template's `". "` separator. The file is renamed, as the LPPL requires for modified files, and its header lists every changed hook. Differences in econ's core routines between 2.1 and 3.1.1 may remain; they were not otherwise checked.

## Build evidence

Tectonic 0.16.9 builds the report. The build log contains one underfull line in the bibliography, informational UTF-8 notices from the template's `algorithm2e` package, and a note that Arial is read from the macOS system font path. Every page was rendered and inspected: there is no blue instruction text, no clipped content and no unresolved reference.
