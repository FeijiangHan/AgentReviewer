# Dynamic Persona现状评估与升级方案（基于 plan.md）

## 1) 当前实现是否“充分”实现动态 Persona 抽取？

结论：**已实现最小可运行版本，但未达到 plan.md 的“稳健/生产级”标准**。

### 已实现
- 参考文献作者候选抽取（启发式规则）。
- OpenAlex / Semantic Scholar / Crossref 三源基础补全。
- 置信度过滤 + top-k 匹配。
- PersonaCard 生成并注入 reviewer prompt。
- 动态 trace 输出（候选、筛选、最终选中等）。

### 主要缺口（对照 plan.md）
- **Reference 解析鲁棒性不足**：仅字符串启发式，缺少结构化 PDF reference parser（如 GROBID/CERMINE）。
- **身份消歧较弱**：缺少“name+paper+coauthor+机构一致性”的联合打分。
- **Persona 完整性门控不足**（本次已补）：过去仅按 confidence，不保证 persona 维度完整。
- **backup 机制语义不严格**：过去 fallback 容易退化为 generic reviewer，而不是优先后备候选。
- **检索层未引入 agentic web search**：仍是 API 元数据聚合，缺少主页/讲座/履历等证据链融合。

---

## 2) 面向“输入 PDF 自动匹配 reviewer”的 API/网站建议

> 目标是“先高精度召回，再严格过滤”，优先避免脆弱网页爬取。

### A. 学术图谱类（推荐主干）
1. **OpenAlex API**
   - 用途：work↔author↔institution 图谱、topic 概念、作者主页链接。
   - 优点：免费、覆盖广、结构稳定。
2. **Semantic Scholar Graph API**
   - 用途：author/paper 查询、h-index/paperCount 等信号。
   - 优点：CS 相关覆盖较好，可与 OpenAlex 交叉验证。
3. **Crossref API**
   - 用途：reference title/DOI 规范化。
   - 优点：DOI 解析强，适合前置“去噪”。

### B. 会议生态工具（可选）
1. **OpenReview 生态（尤其 Expertise 方向）**
   - 适合“给定投稿 + 候选池”做匹配评分。
   - 一般需要与会务系统/候选数据库结合，不是纯公网“一键 PDF→最终审稿人”SaaS。

### C. 文献解析组件（应前置）
1. **GROBID（强烈推荐）**
   - 用于 PDF → 结构化 references（作者、标题、年份、venue）。
2. CERMINE / ParsCit（备选）

**建议组合**：`PDF -> GROBID references -> Crossref DOI 规范化 -> OpenAlex/S2 作者图谱扩展 -> 匹配与过滤`。

---

## 3) Persona 构建升级：从“模板填空”到“检索增强生成”

建议采用两阶段：

1. **Evidence Bundle 构建（可由 Agent 搜索完成）**
   - 输入：作者名、机构候选、关键论文列表。
   - 搜索源优先级：
     1) 机构主页 / 实验室主页
     2) OpenAlex/S2 publication graph
     3) Google Scholar 页面（只读公开信息）
     4) 公开视频/讲座摘要（可选）
   - 输出：标准化证据对象（来源 URL、抓取时间、证据片段、置信度）。

2. **Persona Synthesis（LLM）**
   - Prompt 只允许使用 evidence bundle 中出现的事实。
   - 每一条 persona 维度都要带 evidence pointer（source id）。
   - 不得输出隐私推断或非学术属性。

---

## 4) 建议的“严格 persona 维度”

最小必备维度（每项都应 evidence-linked）：
1. 身份锚点：`name`, `affiliation`。
2. 研究主题画像：`research_areas`（>=2）。
3. 方法偏好：`methodological_preferences`（>=2）。
4. 审稿关注点：`common_concerns`（>=2）。
5. 风格：`style_signature`（文本）。
6. 证据：`evidence_sources`（>=2）。
7. 可信度：`confidence`（>=0.5）。

### 严格筛选策略
- 对每个候选计算 `completeness_score`。
- 若 `< threshold`，**剔除**该候选。
- 若最终不足 K，按排序继续取后备候选；仍不足时才触发 generic fallback。

---

## 5) 本次代码落地内容

- 新增 persona 完整性校验器：`persona/persona_validator.py`。
- Dynamic pipeline 增加：
  - `min_persona_completeness` 参数；
  - 校验 trace（accepted/missing dimensions/completeness）；
  - backup pool 追踪字段。
- Reference miner 增强：
  - 去编号、去 `et al.`、支持 `and/&/;` 分隔、归一化去重。
- Reviewer matcher 新增 `rank()`，支持“先全量排序再严格过滤”。

> 这使系统从“仅 confidence 阈值”提升为“confidence + 完整性双门控”。


## 6) 本轮进一步实现（针对最新反馈）

已按反馈完成两项关键改动：

1. **取消 confidence 硬过滤**
   - 现在 `confidence` 仅作为排序信号，不再做阈值淘汰。
   - 避免低置信度作者被过早丢弃，提升候选多样性。

2. **增加可选 LLM 搜索式补全（不强依赖学术图谱 API）**
   - 新增 `LLMSearchEnricher`，可在动态 persona pipeline 中调用 provider 的 JSON 生成接口进行候选补全。
   - 搜索提示词只允许围绕 reference 抽出的候选姓名，遵守匿名论文约束（不从论文作者字段取人）。
   - 通过环境变量 `DYNAMIC_PERSONA_USE_LLM_SEARCH=1` 启用。

> 说明：若底层模型未启用真实 web 工具，该补全会退化为 best-effort（返回未知字段），但不会破坏 pipeline。
