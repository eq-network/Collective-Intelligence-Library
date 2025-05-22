import ast
import os
from pathlib import Path

def extract_imports(file_path):
    """Extract all imports from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            print(f"Syntax error in {file_path}")
            return []
    
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for name in node.names:
                    if node.level > 0:  # Relative import
                        imports.append(f"RELATIVE({node.level}): {node.module}.{name.name}")
                    else:  # Absolute import
                        imports.append(f"{node.module}.{name.name}")
    
    return imports

def analyze_project(project_dir):
    """Analyze a Python project for imports."""
    project_dir = Path(project_dir)
    
    # Dictionary to store file -> imports mapping
    file_imports = {}
    
    # Walk through Python files
    for file_path in project_dir.glob('**/*.py'):
        relative_path = file_path.relative_to(project_dir)
        module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
        
        imports = extract_imports(file_path)
        file_imports[module_path] = imports
    
    # Look for circular imports
    potential_cycles = []
    
    for module, imports in file_imports.items():
        for imported in imports:
            # Clean up relative imports for comparison
            clean_import = imported
            if imported.startswith("RELATIVE"):
                # Extract just the module name for comparison
                clean_import = imported.split(":")[1].strip()
            
            # Check if the imported module also imports this module
            if clean_import in file_imports and module in file_imports[clean_import]:
                potential_cycles.append((module, clean_import))
    
    return file_imports, potential_cycles

if __name__ == "__main__":
    project_dir = "C:\\Users\\Jonas\\Documents\\GitHub\\Mycorrhiza"
    
    print(f"Analyzing imports in: {project_dir}\n")
    file_imports, potential_cycles = analyze_project(project_dir)
    
    # Print all imports for core.category and core.property
    print("=== Imports for core.category ===")
    if "core.category" in file_imports:
        for imp in file_imports["core.category"]:
            print(f"  - {imp}")
    else:
        print("  Module not found")
    
    print("\n=== Imports for core.property ===")
    if "core.property" in file_imports:
        for imp in file_imports["core.property"]:
            print(f"  - {imp}")
    else:
        print("  Module not found")
    
    print("\n=== Potential Circular Imports ===")
    if potential_cycles:
        for module1, module2 in potential_cycles:
            print(f"  - {module1} <--> {module2}")
    else:
        print("  No circular imports detected!")