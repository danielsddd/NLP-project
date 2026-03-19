import json, random
recs = [json.loads(l) for l in open('data/silver_labels/teacher_output.jsonl','r',encoding='utf-8')]
mods = [r for r in recs if r.get('teacher_output') and r['teacher_output'].get('has_modification')]
random.seed(42)
random.shuffle(mods)
for r in mods[:10]:
    print('='*60)
    print(f"TOP: {r['top_comment_text']}")
    for i, rep in enumerate(r.get('replies_texts', []), 1):
        print(f"  REPLY {i}: {rep}")
    print(f"MODS:")
    for m in r['teacher_output']['modifications']:
        src = m['source_comment']
        print(f"  {m['aspect']:15s} | span: \"{m['span']}\" | from: {src} | conf: {m['confidence']}")
    print()
