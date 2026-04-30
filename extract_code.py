import json
nb = json.load(open('week5_2_mnist_cnn.ipynb', encoding='utf-8'))
with open('mnist_code.py', 'w', encoding='utf-8') as f:
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            f.write('# CELL\n')
            f.write(''.join(c['source']))
            f.write('\n\n')
