from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
base = Path(sys.argv[2])
selected = {}
candidates = []
for mode in ('projector','lora'):
    rows = []
    for lr in ('3e-5','1e-4','3e-4'):
        key = f'{mode}-{lr}-17'
        run = base/'train/full_image17' if key == 'projector-1e-4-17' else root/'train'/key
        events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
        epochs = [e for e in events if e['event']=='epoch']
        if len(epochs) != 3 or events[-1]['event'] != 'complete':
            raise RuntimeError(f'Incomplete tuning run {key}')
        best = max(epochs,key=lambda e: e['score'])
        parameters = next(e['count'] for e in events if e['event']=='parameters')
        rows.append(dict(method=mode, lr=lr, path=str(run), epoch=best['epoch'], score=best['score'],
                         map_switch=best['validation']['flip'][0], num_params=parameters))
    best = max(rows, key=lambda r: (r['score'], r['map_switch'], r['lr']=='1e-4'))
    selected[mode] = best
    candidates.extend(rows)
(root/'selection.json').write_text(json.dumps(dict(candidates=candidates, selected=selected),indent=2)+'\n')
