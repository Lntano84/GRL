# Reference audit

Last checked: **2026-09-18**

Every entry cited by the DASFAA manuscript was checked against a publisher/official proceedings
record or, where the publisher record was incomplete, DBLP plus the author-hosted paper. The
manuscript now cites 12 unique works and `references.bib` contains no uncited entries.

| key | verified record |
| --- | --- |
| `borgs2014tim` | SIAM SODA record, pp. 946--957, DOI `10.1137/1.9781611973402.70` |
| `tang2014timplus` | SIGMOD 2014, pp. 75--86, DOI `10.1145/2588555.2593670` |
| `tang2015imm` | SIGMOD 2015, pp. 1539--1554, DOI `10.1145/2723372.2723734` |
| `tang2018online` | SIGMOD 2018, pp. 991--1005, DOI `10.1145/3183713.3183749` |
| `nguyen2016stop` | SIGMOD 2016, pp. 695--710, DOI `10.1145/2882903.2915207` |
| `shahrouz2021gim` | IEEE TPDS 32(10), pp. 2386--2399, DOI `10.1109/TPDS.2021.3066215` |
| `min2020curipples` | ICS 2020, article 12, 11 pages, DOI `10.1145/3392717.3392750` |
| `manchanda2020gcomb` | NeurIPS 33, pp. 20000--20011; official NeurIPS proceedings |
| `loukides2020laico` | ACM TOIT 20(4), article 39, pp. 39:1--39:31, DOI `10.1145/3408315` |
| `leskovec2007cost` | KDD 2007, pp. 420--429, DOI `10.1145/1281192.1281239` |
| `chen2023rl4ccim` | IJCAI 2023, pp. 5531--5540, DOI `10.24963/ijcai.2023/614` |
| `zhu2026overexposure` | Information Sciences 744, article 123375, DOI `10.1016/j.ins.2026.123375` |

Material corrections made during the audit:

- Replaced the nonexistent `Efficient and Effective Influence Maximization on GPUs`/Li et al.
  record with the published gIM paper by Shahrouz, Salehkaleybar, and Hashemi.
- Replaced the incorrect GCOMB title and author list with the official NeurIPS record.
- Corrected the LAICO citation to *Overexposure-Aware Influence Maximization* in ACM TOIT.
- Corrected the source-model title and DOI for the 2026 Information Sciences paper.
- Corrected the complex-contagion RL record from an incomplete AAMAS attribution to the full
  IJCAI paper with all six authors.
- Removed the unverified `liang2024rethinking` claim, the duplicate TIM/TIM+ entry, and the unused
  `li2017almost` entry.

The short venue names in `references.bib` are intentional: they preserve the verified metadata
while keeping the LNCS submission within the 16-page limit.
