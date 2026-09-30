"""Native clinical/dialogue supervision; no generated negatives or future summaries.

All turns of a dialogue share its original group. Short greetings are not global
link anchors: exact-input deduplication is still performed by the integrator.
"""
from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path

from .core import group_key, one_hot, text
from .medical_extended import _case, _records

IMCS_ACTS = {
    'Request-Basic_Information': '询问基本信息', 'Inform-Basic_Information': '告知基本信息',
    'Request-Symptom': '询问症状', 'Inform-Symptom': '告知症状',
    'Request-Existing_Examination_and_Treatment': '询问已有检查或治疗',
    'Inform-Existing_Examination_and_Treatment': '告知已有检查或治疗',
    'Request-Drug_Recommendation': '询问药物建议', 'Inform-Drug_Recommendation': '提供药物建议',
    'Request-Medical_Advice': '询问就医建议', 'Inform-Medical_Advice': '提供就医建议',
    'Request-Etiology': '询问病因', 'Inform-Etiology': '说明病因',
    'Request-Precautions': '询问注意事项', 'Inform-Precautions': '说明注意事项',
    'Diagnose': '作出诊断陈述', 'Other': '其他对话行为',
}
SYMPTOM_STATES = {'0': '阴性：否认该症状', '1': '阳性：存在该症状', '2': '不确定：尚未确定该症状'}
REMEDI_INTENTS = {'Inform': '告知信息', 'Inquire': '询问信息', 'Recommend': '提出建议',
                  'Chitchat': '寒暄闲聊', 'QuestionAnswering': '回答问题',
                  'Other': '其他行为', 'Diagnosis': '诊断陈述'}
PSY_STRATEGIES = {'Question': '提问', 'Reflection of Feelings': '反映感受',
                  'Others': '其他策略', 'Providing Suggestions': '提供建议',
                  'Unknown': '策略未知', 'Information': '提供信息',
                  'Affirmation and Reassurance': '肯定与安慰', 'Role-play': '角色扮演',
                  'Restatement or Paraphrasing': '复述或释义', 'Self-disclosure': '自我表露'}
PSY_EMOTIONS = {'Anxiety': '焦虑', 'Depression': '抑郁情绪', 'Sadness': '悲伤',
                'Anger': '愤怒', 'Fear': '恐惧', 'Guilty': '内疚', 'Shame': '羞耻',
                'Neutral': '中性', 'Unknown': '未知', 'Others': '其他情绪', 'Happiness': '快乐'}


def _dialogue_group(source, original_id, full_text):
    family = source['source_family']
    return group_key(family + ':dialogue', str(original_id)), [group_key('medical_dialogue', full_text)]


def _choice(source, sid, original_id, group, links, state, prompt, criteria, label, task, field):
    if label not in criteria:
        raise ValueError('unknown original label')
    result = _case(source, sid, group, state,
                   dict(type='choice', instructions=prompt, criteria=criteria), one_hot(criteria, label),
                   task_family=task, semantic_label=label, medical_field=field,
                   option_count=len(criteria), answer_count=1, links=links, original_id=original_id)
    result['_meta']['integration']['link_exact_state'] = False
    return result


def _memberships(source, sid, original_id, group, links, state, criteria, selected, task, field, prompt):
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError('expected nonempty distinct original multilabel annotations')
    if not set(selected) <= set(criteria):
        raise ValueError('unknown original multilabel annotation')
    for candidate, description in criteria.items():
        label = 'true' if candidate in selected else 'false'
        result = _case(source, f'{sid}:{candidate}', group, state,
                       dict(type='noul', instructions=f'{prompt}：{description}？',
                            criteria={'false': '原标注不含此标签', 'true': '原标注包含此标签'}),
                       one_hot(['false', 'true'], label), task_family=task,
                       semantic_label=label, medical_field=field, option_count=len(criteria),
                       answer_count=len(selected), links=links, original_id=original_id)
        result['_meta']['integration'].update(link_exact_state=False, candidate_label=candidate)
        yield result


def convert_imcs21(source, paths, audit, review):
    """Native DAC and per-mentioned-symptom status, with historical context only."""
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('IMCS needs the original dialogue-ID dictionary')
        for did, row in data.items():
            sid = f'{Path(path).name}:{did}'
            try:
                turns = row['dialogue']
                if not isinstance(turns, list) or not turns:
                    raise ValueError('empty IMCS dialogue')
                full = '\n'.join(text(t['speaker'], 'speaker') + '：' + text(t['sentence'], 'sentence') for t in turns)
                group, links = _dialogue_group(source, did, full)
                history = []
                for idx, turn in enumerate(turns):
                    tid = f'{sid}:{idx}'
                    history.append(turn['speaker'] + '：' + turn['sentence'])
                    state = '对话截至当前句：\n' + '\n'.join(history)
                    try:
                        yield _choice(source, tid + ':act', str(did), group, links, state,
                                      '判断当前最后一句的医疗对话行为。', IMCS_ACTS, turn['dialogue_act'],
                                      'medical_dialogue_act', 'pediatrics')
                    except (ValueError, TypeError, KeyError) as exc:
                        audit(source, tid + ':act', str(exc))
                    try:
                        symptoms, states = turn['symptom_norm'], turn['symptom_type']
                        if not isinstance(symptoms, list) or not isinstance(states, list) or len(symptoms) != len(states):
                            raise ValueError('IMCS symptom names and labels must align')
                        labels = {}
                        for symptom, status in zip(symptoms, states):
                            text(symptom, 'symptom')
                            status = str(status)
                            if status not in SYMPTOM_STATES:
                                raise ValueError('unknown IMCS symptom status')
                            if symptom in labels and labels[symptom] != status:
                                raise ValueError('conflicting status for same symptom in current turn')
                            labels[symptom] = status
                        for symptom, status in labels.items():
                            yield _choice(source, tid + ':symptom:' + symptom, str(did), group, links, state,
                                          f'根据当前最后一句及此前语境，症状“{symptom}”被表述为何种状态？',
                                          SYMPTOM_STATES, status, 'clinical_symptom_status', 'pediatrics')
                    except (ValueError, TypeError, KeyError) as exc:
                        audit(source, tid + ':symptom', str(exc))
            except (ValueError, TypeError, KeyError) as exc:
                audit(source, sid, str(exc))


def convert_dialmed(source, paths, audit, review):
    """Restore the recorded medication set from the original masked dialogue."""
    label_path = source.get('_resolved_auxiliary_paths', {}).get('labels')
    if not label_path:
        raise ValueError('DialMed requires its independent 70-label vocabulary')
    ids = json.loads(Path(label_path).read_text(encoding='utf-8-sig'))
    if len(ids) != 70 or set(ids.values()) != set(range(70)):
        raise ValueError('DialMed requires the full original 70-drug vocabulary')
    criteria = {k: k for k in sorted(ids, key=ids.get)}
    for path, index, row in _records(paths):
        sid = f'{path.name}:{index}'
        try:
            turns = row['dialog']
            if not isinstance(turns, list) or not turns:
                raise ValueError('DialMed needs its nonempty original masked dialog')
            state = '\n'.join(text(x, 'masked turn') for x in turns)
            # Never read original_dialog or disease into input. The source already
            # masks medication mentions; do not create a new masking convention.
            group, links = _dialogue_group(source, sid, state)
            yield from _memberships(source, sid, sid, group, links, state, criteria, row['label'],
                                    'recorded_masked_medication', 'general_medicine',
                                    '依据遮盖药名后的对话，原记录的用药集合是否包含以下药品（不是适宜性或禁忌判断）')
        except (ValueError, TypeError, KeyError) as exc:
            audit(source, sid, str(exc))


def convert_remedi(source, paths, audit, review):
    """Human base labels; official seed-5 657/100/800 dialogue partition."""
    split = source['original_split']
    if split not in ('train', 'dev', 'test'):
        raise ValueError('ReMeDi needs official train/dev/test original_split')
    for path in paths:
        rows = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(rows, list) or len(rows) != 1557:
            raise ValueError('ReMeDi official partition requires complete 1,557-dialogue base revision')
        random.Random(5).shuffle(rows)
        rows = {'train': rows[:657], 'dev': rows[657:757], 'test': rows[757:]}[split]
        for row in rows:
            sid = f'{Path(path).name}:{row.get("dialogue", "unknown")}'
            try:
                did = str(row['dialogue'])
                turns = row['information']
                full = '\n'.join(text(t['role'], 'role') + '：' + text(t['sentence'], 'sentence') for t in turns)
                group, links = _dialogue_group(source, did, full)
                history = []
                for i, turn in enumerate(turns):
                    history.append(turn['role'] + '：' + turn['sentence'])
                    tid = f'{sid}:{i}'
                    try:
                        acts = turn['actions']
                        if not isinstance(acts, list) or not acts or any('intent' not in a for a in acts):
                            raise ValueError('empty or incomplete ReMeDi action annotation')
                        selected = sorted({a['intent'] for a in acts})
                        state = '对话截至当前句：\n' + '\n'.join(history)
                        # Full original multilabel set; absence means unannotated
                        # intent in this closed annotation schema, not medical falsehood.
                        yield from _memberships(source, tid, did, group, links, state,
                                                REMEDI_INTENTS, selected, 'medical_dialogue_intent',
                                                'general_medicine', '当前最后一句的原对话意图标注是否包含')
                    except (ValueError, TypeError, KeyError) as exc:
                        audit(source, tid, str(exc))
            except (ValueError, TypeError, KeyError) as exc:
                audit(source, sid, str(exc))


def convert_psy_insight(source, paths, audit, review):
    """Chinese emotion/strategy labels; continuous source sessions share groups."""
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        previous_group = None
        for row in data:
            did = str(row.get('dialog_id', 'unknown'))
            sid = f'{Path(path).name}:{did}'
            try:
                turns = row['dialog']
                full = '\n'.join(text(t['speaker'], 'speaker') + '：' + text(t['content'], 'content') for t in turns)
                group, links = _dialogue_group(source, did, full)
                # Source documentation: both flags describe continuity with the
                # preceding session. Preserve both even if texts do not exactly match.
                if row.get('is_same_session') == 1 or row.get('is_same_qa') == 1:
                    if previous_group is None:
                        raise ValueError('Psy-Insight continuation has no preceding source session')
                    links.append(previous_group)
                previous_group = group
                history = []
                for i, turn in enumerate(turns):
                    history.append(turn['speaker'] + '：' + turn['content'])
                    tid = f'{sid}:{i}'
                    try:
                        if turn['speaker'] == 'Supporter':
                            criteria, labels, task, prompt = PSY_STRATEGIES, turn['strategy'], 'psychological_support_strategy', '当前最后一句的原支持策略标注是否包含'
                        elif turn['speaker'] == 'Seeker':
                            criteria, labels, task, prompt = PSY_EMOTIONS, turn['emotional label'], 'psychological_emotion', '当前最后一句的原情绪标注是否包含（不是精神疾病诊断）'
                        else:
                            raise ValueError('unknown Psy-Insight speaker')
                        state = '对话截至当前句：\n' + '\n'.join(history)
                        for case in _memberships(source, tid, did, group, links, state, criteria, labels, task, 'mental_health', prompt):
                            case['_meta']['supervision'] = 'source_model_assisted_annotation'
                            yield case
                    except (ValueError, TypeError, KeyError) as exc:
                        audit(source, tid, str(exc))
            except (ValueError, TypeError, KeyError) as exc:
                audit(source, sid, str(exc))


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('pickle globals/classes are forbidden; only primitive data accepted')


def convert_medical_ds(source, paths, audit, review):
    """MZ/DXY full-observation disease classification, never active diagnosis."""
    label_path = source.get('_resolved_auxiliary_paths', {}).get('labels')
    if not label_path:
        raise ValueError('medical_ds requires independent disease vocabulary')
    criteria = {x.strip(): x.strip() for x in Path(label_path).read_text().splitlines() if x.strip()}
    if len(criteria) < 2:
        raise ValueError('disease vocabulary needs at least two labels')
    for path in paths:
        with Path(path).open('rb') as stream:
            rows = _PrimitiveUnpickler(stream, encoding='utf-8').load()
        if isinstance(rows, dict):
            rows = rows[source['original_split']]
        if not isinstance(rows, list):
            raise ValueError('medical_ds expects primitive case list')
        for i, row in enumerate(rows):
            # A single pickle can contain all partitions. Its row index is only
            # unique within a split; never merge patient 0 across train/test.
            native_id = row.get('consult_id') if isinstance(row, dict) else None
            sid = ('consult:' + str(native_id) if native_id is not None else
                   f'{Path(path).name}:{source["original_split"]}:{i}')
            try:
                # Releases use both flat and nested goal forms.
                goal = row.get('goal', row)
                merged = {}
                for key in ('explicit_inform_slots', 'implicit_inform_slots'):
                    symptoms = goal[key]
                    if not isinstance(symptoms, dict):
                        raise ValueError('symptoms must be a dictionary')
                    for symptom, status in symptoms.items():
                        text(symptom, 'symptom')
                        if status not in (True, False, '1', '0', '2'):
                            raise ValueError('unknown symptom status')
                        status = '1' if status is True else '0' if status is False else status
                        if symptom in merged and merged[symptom] != status:
                            raise ValueError('conflicting explicit/implicit symptom')
                        merged[symptom] = status
                if not merged:
                    raise ValueError('empty symptoms')
                state = '已获得的完整症状信息：\n' + '\n'.join(f'{k}：{SYMPTOM_STATES[v]}' for k, v in sorted(merged.items()))
                group = group_key('medical_text', state)
                label = row.get('disease_tag', goal.get('disease_tag'))
                yield _choice(source, sid, sid, group, [group], state,
                              '依据给定的完整症状信息，判定原数据封闭疾病集合中的病例标签。',
                              criteria, label, 'closed_set_disease_classification', 'general_medicine')
            except (ValueError, TypeError, KeyError) as exc:
                audit(source, sid, str(exc))


ADAPTERS = {'imcs21_clinical': convert_imcs21, 'dialmed_recorded': convert_dialmed,
            'remedi_human': convert_remedi, 'psy_insight_cn': convert_psy_insight,
            'medical_ds_disease': convert_medical_ds}


def convert_lcmdc(source, paths, audit, review):
    """Original website routing labels, not expert diagnosis adjudication."""
    level = source.get('label_level', 1)
    if level not in (1, 3):
        raise ValueError('LCMDC supports original level 1 or level 3 labels')
    label_path = source.get('_resolved_auxiliary_paths', {}).get('labels')
    if not label_path:
        raise ValueError('LCMDC requires original independent label dictionary')
    ids = json.loads(Path(label_path).read_text(encoding='utf-8-sig'))
    n = 14 if level == 1 else 120
    if len(ids) != n or set(ids.values()) != set(range(n)):
        raise ValueError('LCMDC requires its complete original label vocabulary')
    criteria = {k: k for k in sorted(ids, key=ids.get)}
    for path in paths:
        with Path(path).open(encoding=source.get('encoding', 'gb18030'), newline='') as stream:
            for i, row in enumerate(csv.DictReader(stream), 1):
                sid = f'{Path(path).name}:{i}'
                try:
                    title, info = text(row['titles'], 'title'), text(row['infos'], 'info')
                    label = row[f'labels_{level}_word']
                    if label not in ids or int(row[f'labels_{level}']) != ids[label]:
                        raise ValueError('LCMDC numeric and text label disagree')
                    state = f'问题标题：{title}\n问题描述：{info}'
                    group = group_key('medical_text', state)
                    result = _choice(source, sid, sid, group, [group], state,
                                     '根据医疗问题判断原网站分诊科室。' if level == 1 else '根据医疗问题判断原网站的细粒度就诊类目（类目不等于确诊）。',
                                     criteria, label, 'medical_department_triage' if level == 1 else 'medical_detailed_routing',
                                     'general_medicine')
                    result['_meta']['supervision'] = 'source_website_routing_label'
                    yield result
                except (ValueError, TypeError, KeyError) as exc:
                    audit(source, sid, str(exc))


ADAPTERS['lcmdc_triage'] = convert_lcmdc


def convert_meddg_entities(source, paths, audit, review):
    """Classify supplied original entities into five native annotation classes."""
    criteria = {'Symptom': '症状', 'Medicine': '药品', 'Examination': '检查',
                'Attribute': '属性', 'Disease': '疾病'}
    for path in paths:
        with Path(path).open('rb') as stream:
            rows = _PrimitiveUnpickler(stream, encoding='utf-8').load()
        if not isinstance(rows, list):
            raise ValueError('MedDG expects the original nested dialogue list')
        for i, turns in enumerate(rows):
            sid = f'{Path(path).name}:{i}'
            try:
                full = '\n'.join(text(t['id'], 'speaker') + '：' + text(t['Sentence'], 'sentence') for t in turns)
                group, links = _dialogue_group(source, sid, full)
                for j, turn in enumerate(turns):
                    tid = f'{sid}:{j}'
                    try:
                        labels = {}
                        for category in criteria:
                            values = turn[category]
                            if not isinstance(values, list):
                                raise ValueError('entity annotation must be a list')
                            for entity in values:
                                text(entity, 'entity')
                                if entity in labels and labels[entity] != category:
                                    raise ValueError('one entity has conflicting source types')
                                labels[entity] = category
                        for entity, category in labels.items():
                            yield _choice(source, tid + ':' + entity, sid, group, links,
                                          f'当前语句：{turn["Sentence"]}\n给定原标注的规范实体：{entity}',
                                          '判断给定实体在原医学标注中的类型。这是给定实体分类，不是判断医学结论真假。',
                                          criteria, category, 'medical_entity_type', 'gastroenterology')
                    except (ValueError, TypeError, KeyError) as exc:
                        audit(source, tid, str(exc))
            except (ValueError, TypeError, KeyError) as exc:
                audit(source, sid, str(exc))


ADAPTERS['meddg_entity_type'] = convert_meddg_entities
