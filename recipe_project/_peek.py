import json
lines = open('data/silver_labels/teacher_output.jsonl','r',encoding='utf-8').readlines()
total = len(lines)
mods = []
for l in lines:
    rec = json.loads(l)
    to = rec.get('teacher_output')
    if to and to.get('has_modification'):
        mods.append(rec)
nulls = sum(1 for l in lines if json.loads(l).get('teacher_output') is None)
print(f'Total: {total}, With mods: {len(mods)}, Errors (null): {nulls}')
for m in mods[:10]:
    text = m['top_comment_text'][:70]
    aspects = [(x['aspect'], x['span']) for x in m['teacher_output']['modifications']]
    print(f'{text} -> {aspects}')
