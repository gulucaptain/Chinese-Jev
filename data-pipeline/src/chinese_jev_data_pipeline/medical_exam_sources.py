"""Native full medical exam releases; answer/explanation fields stay out of input.

Adds explicit multiple-answer decisions and author-release parsers. Acquiring
files is separate from conversion. No generated distractors or relabelled data.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .core import group_key, one_hot, text
from .medical_extended import _case, _exam_input, _records

_TRADITIONAL_VISUAL = re.compile(r'(?:如|見|下|上|該|此|附)圖|圖示|圖中|圖片|見下表|如下表|見表\s*\d')


def _links(query):
    links = {group_key('medical_text', query)}
    # Native MLEC/CMExam flattened shared cases lack parent identifiers. A long
    # exact clinical prefix links views without consulting answers. This is a
    # conservative overlap guard, not a claim to recover all shared cases.
    compact = re.sub(r'\s+', '', query)
    if len(compact) >= 100:
        links.add(group_key('medical_exam_prefix80', compact[:80]))
    return sorted(links)


def _options(options):
    if not isinstance(options, dict):
        raise ValueError('exam options must be a dictionary')
    options = dict(options)
    # CMB represents many four-option questions with an empty trailing E.
    # Remove only contiguous trailing placeholders; interior holes are errors.
    while options:
        last = sorted(options)[-1]
        if isinstance(options[last], str) and not options[last].strip():
            del options[last]
        else:
            break
    return options


def _emit(source, sid, row, options, answer, *, medical_field=None, allow_multiple=True,
          extra_meta=None, original_id=None):
    if not isinstance(options, dict):
        raise ValueError('exam options must be a dictionary')
    if _TRADITIONAL_VISUAL.search(str(row.get('question', ''))) or any(
            _TRADITIONAL_VISUAL.search(str(v)) for v in options.values()):
        raise ValueError('explicit traditional Chinese image/table reference needs review')
    query, options, selected = _exam_input(row, options, answer)
    if not allow_multiple and len(selected) != 1:
        raise ValueError('source requires a single answer')
    group = group_key('medical_text', query)
    metadata = dict(medical_field=medical_field or source['medical_field'],
                    option_count=len(options), answer_count=len(selected),
                    links=_links(query), original_id=original_id or sid)
    if len(selected) == 1:
        question = dict(type='choice', instructions='根据材料中的医学试题，选择一个正确答案。', criteria=options)
        cases = [_case(source, sid, group, query, question, one_hot(options, selected[0]),
                       task_family='medical_exam_single', semantic_label='single_answer', **metadata)]
    else:
        context = query + '\n\n原题全部选项：\n' + '\n'.join(f'{k}：{v}' for k, v in options.items())
        cases = []
        for key in options:
            label = 'true' if key in selected else 'false'
            question = dict(type='noul', instructions=f'这是多项选择题。依据原题完整选项，选项 {key} 是否应被选中？',
                            criteria={'false': '原答案未选择此选项', 'true': '原答案选择此选项'})
            cases.append(_case(source, f'{sid}:{key}', group, context, question,
                               one_hot(['false', 'true'], label), task_family='medical_exam_multi',
                               semantic_label=label, **metadata))
    for case in cases:
        case['_meta']['integration'].update(extra_meta or {})
        yield case


def convert_cmexam_full(source, paths, audit, review):
    for value in paths:
        path = Path(value)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'Question', 'Options', 'Answer'} <= set(reader.fieldnames or []):
                raise ValueError('CMExam expects original CSV column names')
            for index, row in enumerate(reader, 2):
                sid = f'{path.name}:{index}'
                try:
                    options = {}
                    for line in text(row['Options'], 'Options').splitlines():
                        match = re.fullmatch(r'([A-Z])\s+(.+)', line.strip())
                        if match is None or match[1] in options:
                            raise ValueError('invalid or duplicate CMExam option line')
                        options[match[1]] = match[2]
                    yield from _emit(source, sid, {'question': row['Question']}, options, row['Answer'])
                except (ValueError, KeyError, TypeError) as exc:
                    audit(source, sid, str(exc))


def convert_cmb_full(source, paths, audit, review):
    answers = None
    answer_path = source.get('_resolved_auxiliary_paths', {}).get('answers')
    if answer_path:
        rows = json.loads(Path(answer_path).read_text(encoding='utf-8-sig'))
        answers = {}
        for row in rows:
            if row['id'] in answers:
                raise ValueError('duplicate CMB answer id')
            answers[row['id']] = row
    for path, index, row in _records(paths):
        sid = f'{path.name}:{row.get("id", index)}'
        try:
            row = dict(row)
            if answers is not None:
                gold = answers[row['id']]
                for key in ('exam_type', 'exam_class', 'exam_subject', 'question_type'):
                    if row[key] != gold[key]:
                        raise ValueError('CMB question/answer hierarchy does not match')
                if 'answer' in row and row['answer'] != gold['answer']:
                    raise ValueError('CMB duplicate answer sources disagree')
                row['answer'] = gold['answer']
            kind = row['question_type']
            if kind not in ('单项选择题', '多项选择题', 'C型选择题'):
                raise ValueError('unknown CMB question type')
            for field in ('exam_subject', 'exam_type', 'exam_class'):
                text(row[field], field)
            # C-type original questions provide one letter and a full candidate
            # list; retain these as ordinary one-answer decisions.
            _, _, selected = _exam_input(row, _options(row['option']), row['answer'])
            if (kind == '多项选择题') != (len(selected) > 1):
                raise ValueError('CMB declared question type and answer count disagree')
            yield from _emit(source, sid, row, _options(row['option']), row['answer'], medical_field=row['exam_subject'],
                             extra_meta={k: row[k] for k in ('exam_type', 'exam_class', 'question_type')})
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def convert_medqa_full(source, paths, audit, review):
    for path, index, row in _records(paths):
        sid = f'{path.name}:{index}'
        try:
            selected = row['answer_idx']
            if not isinstance(selected, str) or selected not in row['options']:
                raise ValueError('invalid MedQA answer_idx')
            if row.get('answer') != row['options'][selected]:
                raise ValueError('MedQA answer text disagrees with answer_idx')
            yield from _emit(source, sid, row, row['options'], selected, allow_multiple=False)
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def convert_mlec_qa(source, paths, audit, review):
    for path, index, row in _records(paths):
        sid = f'{path.name}:{row.get("qid", index)}'
        try:
            text(row['qid'], 'qid')
            if row['qtype'] not in ('A1型题', 'A2型题', 'B1型题', 'A3/A4型题'):
                raise ValueError('unknown MLEC question type')
            answer = row['answer']
            if isinstance(answer, list):
                if not answer or not all(isinstance(x, str) and len(x) == 1 for x in answer):
                    raise ValueError('invalid MLEC answer list')
                answer = ''.join(answer)
            native = dict(row, question=row['qtext'])
            yield from _emit(source, sid, native, row['options'], answer,
                             extra_meta={'question_type': row['qtype']}, original_id=row['qid'])
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def convert_cnmleqa(source, paths, audit, review):
    keys = ('opa', 'opb', 'opc', 'opd', 'ope')
    for path, index, row in _records(paths):
        sid = f'{path.name}:{row.get("id", index)}'
        try:
            answer = row['answer']
            if answer not in keys:
                raise ValueError('CNMLEQA answer must identify opa..ope')
            options = dict(zip('ABCDE', [row[key] for key in keys]))
            yield from _emit(source, sid, row, options, 'ABCDE'[keys.index(answer)], allow_multiple=False,
                             extra_meta={'exam_year': row.get('year'), 'original_question_type': row['question_type'],
                                         'upstream_source': row['source']}, original_id=str(row['id']))
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


def convert_empec(source, paths, audit, review):
    for path, index, row in _records(paths):
        sid = f'{path.name}:{index}'
        try:
            year = text(row['year'], 'year')
            years = source.get('include_years')
            if years is not None and year not in years:
                continue
            raw = text(row['question'], 'question')
            markers = list(re.finditer(r'(?:^|\n)\s*([A-D])\.', raw))
            if [m[1] for m in markers] != list('ABCD'):
                raise ValueError('EMPEC expected four ordered A..D option lines')
            question_text = text(raw[:markers[0].start()], 'question stem')
            options = {m[1]: raw[m.end():markers[i + 1].start() if i + 1 < len(markers) else len(raw)].strip()
                       for i, m in enumerate(markers)}
            yield from _emit(source, sid, dict(row, question=question_text), options, row['answer'],
                             medical_field=text(row['subject'], 'subject'), allow_multiple=False,
                             extra_meta={'exam_year': year, 'profession': row['profession']})
        except (ValueError, KeyError, TypeError) as exc:
            audit(source, sid, str(exc))


ADAPTERS = {
    'cmexam_full': convert_cmexam_full,
    'cmb_exam_full': convert_cmb_full,
    'medqa_full': convert_medqa_full,
    'mlec_qa': convert_mlec_qa,
    'cnmleqa': convert_cnmleqa,
    'empec': convert_empec,
}
