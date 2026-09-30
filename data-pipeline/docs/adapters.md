# 内置 Adapter 清单

共 48 个注册入口，不等于 48 份独立数据集。旧格式与扩展格式可能来自同一数据集，
应按原文件结构选择入口，不重复导入同一监督。输入文件和标签表由使用者准备。

核心 adapter：

```text
c3, exam_csv, dureader_yesno, squad, chinese_jev, t2ranking
```

## 医学

```text
cblue_qic, cblue_sts, cblue_qtr, cblue_qqr, cblue_ctc
cmexam, medqa_cn, cmid_coarse, explain_cpe, cmb_exam, webmedqa, tcm_sd
cmedqa2_matching
cmexam_full, cmb_exam_full, medqa_full, mlec_qa, cnmleqa, empec
imcs21_clinical, dialmed_recorded, remedi_human, medical_ds_disease
psy_insight_cn, lcmdc_triage, meddg_entity_type
cmeee_typing, cmeie_typing, cdn_retrieved_pairs, text2dt_typing
alternatecd_mdcf, alternatecd_causal_pair, psyqa_strategy
```

医学 adapter 与其他来源使用同一核心：只根据输出的 `group_key` 和显式
`integration.link_keys` 建立传递组件。核心不会仅因两个记录的 `original_id`
或 state 相同就自动合并母组；需要的关联必须由 adapter 明确输出。

## 法律

```text
cail2019_elements, jec_qa, cail2018, cail2019_scm, lcr_cn
```

## 金融、新闻与评论

```text
finfe, finnsp, tnews, chnsenticorp_htl
```

## 调用前需确认的配置

| 入口 | 输入或额外条件 |
|---|---|
| `chinese_jev` | 已有的 `state / questions / gold` cases；保留 `group_key` 并将多问题记录展开为单 decision |
| `t2ranking` | `queries / qrels / collection` 三张本地表，路径相对于配置目录；可选 `query_fraction`（0 到 1，按 query 哈希抽样，默认 1.0）与 `sample_seed`（默认 42） |
| `squad` | 默认 `mode=review` 只输出复查材料；`mode=answerability` 要求布尔 `is_impossible` 和一致的参考答案 |
| `cmedqa2_matching` | `paths` 是原候选关系 CSV；`auxiliary_paths.questions / answers` 提供两张正文表；同一 query 的候选须连续 |
| `cail2019_elements` | `labels_available=true`、原子领域 `divorce / labor / loan`，以及按行对应的 `auxiliary_paths.tags / tag_names` |
| `lcr_cn` | JSONL 中需有原文档 `url`，当前 adapter 按原规范文档分组，不采用条款级隔离 |

完整字段读取逻辑见 [核心格式](../src/chinese_jev_data_pipeline/core.py)、
[医学](../src/chinese_jev_data_pipeline/domain_medical.py)、
[医学匹配](../src/chinese_jev_data_pipeline/medical_matching_sources.py)、
[法律](../src/chinese_jev_data_pipeline/domain_legal.py)和
[金融与生活](../src/chinese_jev_data_pipeline/domain_everyday.py)。
