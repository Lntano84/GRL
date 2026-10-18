# 图1：动机图 V1

这张图说明两个问题：不同状态需要不同搜索深度；相关变电站的连接方案可能分散在候选序列中。图为概念示意，不是实验数据图。

## 文件

- FRAME_动机图_V1.png：查看预览。
- FRAME_动机图_V1.pdf：170毫米宽的矢量排版文件。
- FRAME_动机图_V1.svg：矢量版本，含diagrams.net嵌入源。
- FRAME_动机图_V1.drawio：用Draw.io打开；文字、分隔线、括号和箭头是原生可编辑对象。候选条带和图标为嵌入SVG，组件原文件在assets内，可以替换或进一步编辑。
- source/build.py：用Matplotlib生成候选条带，构造可编辑版式。
- source/export.py：调用实际Draw.io程序导出，再用pypdf固定物理尺寸、Poppler渲染检查。
- 图注与素材署名.md：正式插入论文时使用的图注和素材署名。
- visual_review.json：最终尺寸、矢量检查与实际视觉检查记录。

## 视觉设计

ColorBrewer 2.0 Set2三类颜色，原始官方色表在assets/colorbrewer.json。变电站用A/B/C字母辅助区分；已查询、未查询、追加查询用边框区分。全局扩展与同站补查的顺序完全相同，均追加3次查询。

电力图标采用The Noun Project的Georgiana Ionescu作品Electricity Pole（1364802），使用Wikimedia Commons保存并核验授权的CC BY 3.0 US版本，已缩放和改色。具体来源、作者及许可证链接见图注与素材署名。

## 本轮修改

依据作者对文字密度的反馈，删去右侧重复说明，精简左侧四行注释，改用图形图例。实际检查全尺寸、论文宽度和灰度版本。图中没有加入实验结论或量化节省数据；没有重跑实验，没有改论文正文。原密集版保留于canonical输出的intermediate文件夹。
