# e-Energy 匿名审阅 PDF 核对

匿名源稿已用 D 盘便携 Tectonic 0.17.0 编译。下载 ZIP 的 SHA-256 与官方发行 API 一致。最终版用同一本地缓存离线重编，acmart 为 1.83。便携编译器、包缓存和中间文件在工作区，系统 PATH 未改。

审阅稿为 3 页 US Letter、ACM 双栏，标记匿名作者。替换了类默认示例会名，清除了示例 ISBN 和 DOI。逐页检查布局、加载条件公式、表 1 与引文。PDF 提取的 12 行数值与冻结表一致，4 个引用链接与 10 个嵌入字体通过检查。PDF 作者字段为空，正文、普通元数据和 XMP 无内部仓库、路径与作者标识。TeX 仍报告 2.02 pt 的垂直排版警告，逐页查看未发现裁切或交叠。

审阅稿和源稿包仅保存在 D 盘协作区。供本人自查；按 12 月 15 日锁表计划复查最新征稿、引用与匿名附件。当前 [官方 CFP](https://energy.acm.org/conferences/eenergy/2027/pages/cfp.php) 以上一次 403 记录为准。编译工具见 [Tectonic 0.17.0 发行](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0)。

PDF SHA-256: 5a7aa3fac4f469d1b2a3a21913c9d40c7a15b700cb7bac7c4de6bd0bea741a33

LaTeX SHA-256: 05cdcbf2fc02d2deafb28aaa288c90b5bf4bea9afe0c30832d724695ce3c87b9
