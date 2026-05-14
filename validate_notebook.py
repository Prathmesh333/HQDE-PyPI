import json


NOTEBOOK_PATH = "examples/cbt_deberta_hqde_kaggle.ipynb"


with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Valid JSON")
print(f"  File: {NOTEBOOK_PATH}")
print(f"  Cells: {len(nb['cells'])}")
print(f"  Format: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"  Kernel: {nb['metadata'].get('kernelspec', {}).get('name', 'N/A')}")

code_cells = sum(1 for cell in nb["cells"] if cell["cell_type"] == "code")
markdown_cells = sum(1 for cell in nb["cells"] if cell["cell_type"] == "markdown")

print(f"\n  Code cells: {code_cells}")
print(f"  Markdown cells: {markdown_cells}")

print("\nNotebook JSON is valid.")
print("Runtime still needs validation on the target Kaggle/Colab hardware.")
