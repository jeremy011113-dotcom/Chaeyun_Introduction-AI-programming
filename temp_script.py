import json
with open('week5_3_nonconvex_optimization.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('temp_code.py', 'w', encoding='utf-8') as out:
    for c in nb.get('cells', []):
        if c.get('cell_type') == 'code':
            out.write('# CELL\n')
            out.write(''.join(c.get('source', [])))
            out.write('\n\n')
