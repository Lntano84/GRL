# FRAME方法图V2：分区与多色版

按作者反馈重新学习指南案例，改变原V1三框平铺的布局。全局搜索占左侧主要区域，前缀筛查与分批核查在组内展开；同站补查位于右侧，具体展示站归属与非连续候选。前缀为蓝色、核查为紫色、补查为绿色，均来自ColorBrewer官方配色；正文和旧图保留。

## 查看、编辑与排版

- FRAME_方法图_V2.png：完整预览。
- FRAME_方法图_V2.pdf：170毫米宽、单页纯矢量排版文件。
- FRAME_方法图_V2.svg：矢量图，含嵌入的diagrams.net源。
- FRAME_方法图_V2.drawio：原生可编辑布局、文字、箭头、形状。
- assets/：Matplotlib生成的候选SVG组件、免费Noun图标原件、裁切派生图和ColorBrewer源。
- source/：构建、实际Draw.io导出与归档脚本。
- 论文宽度检查.png、灰度检查.png、visual_review.json：实际查看并核对的预览和记录。
- 图注与素材来源.md：外部图注、素材署名、精确规则对应说明。

## 参考的布局做法

指南：https://dy8q0bnq8y.feishu.cn/wiki/OGpcw6zaRiQ6OwkNxfYcl7IKnBh

本次重新阅读方法图工作流和七个案例，并实际查看案例图片。

1. Agentic Reasoning综述案例：按内容归类，区域大小服务信息层次。
2. Agentic Reasoning方法案例：使用低饱和度功能分区和不同箭头。
3. WAPITI：紧凑图例、清楚的模块命名与主要信息流。
4. Language in the Flow of Time：输入分区、具体序列对象和分组。
5. V-IRL：反馈回路和功能不同的形状。
6. Reasoning Boundary：将抽象机制转成可识别的小对象。
7. Mem-Gallery：主体区域与辅助区域具有不同规模和阅读层级。

本图采用两项贡献为主的非等宽布局，未复制案例内容或加入与FRAME无关的装饰。

## 图标署名在外部图注

power grid by Ary Prasetyo from Noun Project (CC BY 3.0), cropped and scaled.

素材：https://thenounproject.com/icon/power-grid-4199252/

许可：https://creativecommons.org/licenses/by/3.0/

原始SVG保持不变。派生图裁切、缩放，作者署名置于配套外部图注末尾，正式论文使用时随图保留。

## 制作

Draw.io 32.4.1生成布局并实际导出SVG/PDF，Matplotlib生成矢量组件，Poppler渲染核对。未使用图像生成模型，未新增实验或调整模型、规则、预算。
