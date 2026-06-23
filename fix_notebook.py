import json
import os

files_to_fix = [
    "docs/functiongemma/finetuning-with-functiongemma.ipynb",
    "docs/functiongemma/function-calling-with-hf.ipynb",
    "docs/functiongemma/full-function-calling-sequence-with-functiongemma.ipynb"
]

for file_path in files_to_fix:
    full_path = os.path.join(r"C:\Users\HP\Desktop\GSOC_Prop\deepmind_contri\cookbook", file_path)
    if not os.path.exists(full_path):
        continue
        
    with open(full_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if line.strip().startswith('%pip install') or line.strip().startswith('!pip install'):
                    # Insert the dependency fix right before the first pip install we find
                    source.insert(i, "%pip install \"numpy<2.0.0\" \"requests==2.32.4\"\n")
                    break
            # Only fix the first pip install cell
            if any(line.strip().startswith('%pip install') or line.strip().startswith('!pip install') for line in source):
                break
                
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write('\n')
