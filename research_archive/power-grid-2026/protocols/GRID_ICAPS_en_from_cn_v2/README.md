# English manuscript from author-approved Chinese V2

- `ICAPS2027_English_from_author_CN_V2.docx`: editable English reading draft.
- `manuscript_en.md`: text and LaTeX equations for subsequent AAAI typesetting.
- `figures/`: framework, results and completion case, each as SVG/PDF/PNG.
- `绘图指南落实与投稿准备.md`: figure takeaways, guide application and next materials.
- `translation_check.json` / `validation.json`: source fidelity and visual checks.
- `source_snapshot/`: retained author-approved source and extracted source paragraphs.
- `scripts/`: reproducible translation mappings and figure generation.

The Chinese source and all historical drafts remain unchanged. The framework diagram is embedded in the English manuscript; the two empirical figures are separate insertion-ready assets. Final figures are vector-first, generated at publication width with consistent font and muted functional colors.

The reading draft retains five editable OMML equations. Two isolated operators use Unicode compatibility forms to avoid a portable LibreOffice symbol-font fallback; NFKC-normalized equation text remains equal to the source. ASCII operators and complete LaTeX expressions are retained in Markdown for formal typesetting.

Scientific results are inherited from the approved source. No additional replay, training or experimental selection was performed.
