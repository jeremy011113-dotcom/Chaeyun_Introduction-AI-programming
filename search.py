import json, glob

with open('search_results.txt', 'w', encoding='utf-8') as out:
    for f in glob.glob('*.ipynb'):
        try:
            with open(f, encoding='utf-8') as nbf:
                nb = json.load(nbf)
            for c in nb.get('cells', []):
                src = "".join(c.get('source', []))
                if '기초' in src or '도전' in src:
                    out.write(f"=== {f} ===\n{src}\n\n")
        except Exception as e:
            pass
