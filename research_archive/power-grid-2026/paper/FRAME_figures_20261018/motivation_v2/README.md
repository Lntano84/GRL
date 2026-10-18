# 图1：动机图 V2 蓝灰版

本版按作者反馈移除原Set2橙绿配色，采用ColorBrewer官方Blues和Greys，只用蓝色突出当前最佳动作的A站归属，B/C用灰色和字母区分。下方短线标出追加查询。减轻候选边框与策略标题的粗细，保留原V1的两组对比。未修改论文正文或实验记录。

## 查看与编辑

- FRAME_动机图_V2.png：完整预览。
- FRAME_动机图_V2.pdf：170毫米宽的矢量排版文件，保存为单页，未嵌入栅格图像。
- FRAME_动机图_V2.svg：矢量版本，含diagrams.net嵌入源。
- FRAME_动机图_V2.drawio：用Draw.io打开；文字、箭头、括号、图例为原生可编辑对象。候选序列为Matplotlib生成的SVG组件，原文件在assets内。
- source/build.py：生成SVG组件和可编辑图形布局。
- source/export.py：调用实际Draw.io程序导出，pypdf设置纸面尺寸，Poppler渲染检查。
- 论文宽度检查.png、灰度检查.png：实际检查的预览版本。
- 图注与素材署名.md：完整图注、作者和许可信息。
- visual_review.json：源文件校验、矢量属性及视觉检查记录。

## 免费图标与署名

power grid by Ary Prasetyo from Noun Project (CC BY 3.0).

- 素材：https://thenounproject.com/icon/power-grid-4199252/
- 许可：https://creativecommons.org/licenses/by/3.0/
- 用户在注册后从免费With Attribution选项下载SVG并提供；未使用VIP素材或订阅。
- 原始下载SVG按逐字节保存于assets/noun_powergrid_4199252.svg。
- 派生文件只裁切图形周围空白并缩放；原图底部署名改放到整张动机图下方的可见署名行。正式论文使用时须保留作者、来源、许可证及修改说明。

## 图意范围

这是机制示意图，方格数量不对应实际预算参数。左侧一格为候选批次，右侧一格为单个候选；右侧候选顺序一致，全局扩展与同站补查均追加3次。示意图没有展示测量曲线，也没有宣称同站补查始终得到最佳结果。全部旧稿、原图和实验结果继续保留。
