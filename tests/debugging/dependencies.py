"""
Python Project Dependency Analyzer

This script analyzes import dependencies between Python modules in a project,
identifies circular dependencies, and generates visualizations to help
understand and resolve architectural issues.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, NamedTuple
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re

class Import(NamedTuple):
    """Represents an import statement and its metadata."""
    module: str  # The imported module name
    line: int    # Line number where the import appears
    is_relative: bool  # Whether it's a relative import
    level: int   # Relative import level (0 for absolute)
    names: List[str]  # Names imported from the module

class ModuleInfo(NamedTuple):
    """Information about a Python module."""
    file_path: Path
    module_name: str
    package_path: List[str]

def find_python_files(project_dir: Path) -> List[Path]:
    """Find all Python files in the project directory recursively."""
    if not project_dir.is_dir():
        raise ValueError(f"Not a directory: {project_dir}")
    
    python_files = list(project_dir.glob('**/*.py'))
    return python_files

def determine_module_name(file_path: Path, project_dir: Path) -> ModuleInfo:
    """
    Convert a file path to a Python module name.
    
    Args:
        file_path: Path to the Python file
        project_dir: Root directory of the project
    
    Returns:
        ModuleInfo containing the module name and related information
    """
    try:
        rel_path = file_path.relative_to(project_dir)
    except ValueError:
        # File is not within project_dir
        return ModuleInfo(file_path, file_path.stem, [])
    
    # Convert path to module notation
    parts = list(rel_path.parts)
    
    # Handle __init__.py files
    if parts[-1] == '__init__.py':
        module_name = '.'.join(parts[:-1])
        package_path = parts[:-1]
    else:
        # Regular Python file
        module_name = '.'.join(parts[:-1] + [rel_path.stem])
        package_path = parts[:-1]
    
    return ModuleInfo(file_path, module_name, package_path)

def extract_imports(file_path: Path) -> Tuple[List[Import], Optional[str]]:
    """
    Extract all imports from a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        Tuple of (list of imports, error message if any)
    """
    imports = []
    error = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
            
            # Process all import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    line_num = getattr(node, 'lineno', 0)
                    for name_node in node.names:
                        imports.append(Import(
                            module=name_node.name,
                            line=line_num,
                            is_relative=False,
                            level=0,
                            names=[name_node.asname or name_node.name]
                        ))
                elif isinstance(node, ast.ImportFrom):
                    line_num = getattr(node, 'lineno', 0)
                    if node.module is None and node.level > 0:
                        # Relative import like "from .. import xyz"
                        module_name = ''
                    else:
                        module_name = node.module or ''
                    
                    imported_names = [n.name for n in node.names]
                    imports.append(Import(
                        module=module_name,
                        line=line_num,
                        is_relative=node.level > 0,
                        level=node.level,
                        names=imported_names
                    ))
    except Exception as e:
        error = f"Error parsing {file_path}: {str(e)}"
    
    return imports, error

def resolve_relative_import(
    import_info: Import,
    importer_module: ModuleInfo
) -> str:
    """
    Resolve a relative import to its absolute module name.
    
    Args:
        import_info: The import information
        importer_module: Information about the importing module
    
    Returns:
        The resolved absolute module name
    """
    if not import_info.is_relative:
        return import_info.module
    
    # Copy the package path
    package_parts = importer_module.package_path.copy()
    
    # Handle relative import levels
    if import_info.level > len(package_parts):
        # Invalid relative import - goes beyond the top-level package
        return f"INVALID_RELATIVE_IMPORT({import_info.module})"
    
    # Remove parts based on the relative level
    if import_info.level > 1:
        package_parts = package_parts[:-import_info.level + 1]
    
    # Add the imported module
    if import_info.module:
        package_parts.append(import_info.module)
    
    return '.'.join(package_parts)

def build_dependency_graph(
    project_dir: Path,
    display_file_paths: bool = False
) -> Tuple[nx.DiGraph, Dict[str, ModuleInfo], Dict[str, List[Import]], List[str]]:
    """
    Build a dependency graph for the Python project.
    
    Args:
        project_dir: The root directory of the project
        display_file_paths: Whether to use file paths as node labels instead of module names
    
    Returns:
        Tuple of (dependency graph, module information, module imports, errors)
    """
    graph = nx.DiGraph()
    
    # Find all Python files
    python_files = find_python_files(project_dir)
    
    # Maps from module names to module information
    modules = {}
    
    # Maps from module names to their imports
    module_imports = {}
    
    # List of error messages
    errors = []
    
    # First pass: collect all module information
    for file_path in python_files:
        module_info = determine_module_name(file_path, project_dir)
        modules[module_info.module_name] = module_info
        node_label = str(file_path.relative_to(project_dir)) if display_file_paths else module_info.module_name
        graph.add_node(module_info.module_name, label=node_label)
    
    # Second pass: process imports and add edges
    for module_name, module_info in modules.items():
        imports, error = extract_imports(module_info.file_path)
        if error:
            errors.append(error)
        
        module_imports[module_name] = imports
        
        for imp in imports:
            # Resolve the import to an absolute module name
            if imp.is_relative:
                target_module = resolve_relative_import(imp, module_info)
            else:
                target_module = imp.module
            
            # Check if this is an internal module
            found_module = False
            for internal_module in modules.keys():
                # Check if it's a direct import of an internal module
                if target_module == internal_module:
                    graph.add_edge(module_name, internal_module, line=imp.line)
                    found_module = True
                    break
                
                # Check if it's a submodule or parent module import
                elif (target_module.startswith(f"{internal_module}.") or 
                      internal_module.startswith(f"{target_module}.")):
                    # Add an edge to the most specific module
                    if len(target_module) > len(internal_module):
                        graph.add_edge(module_name, target_module, line=imp.line)
                    else:
                        graph.add_edge(module_name, internal_module, line=imp.line)
                    found_module = True
                    break
    
    return graph, modules, module_imports, errors

def find_circular_imports(
    graph: nx.DiGraph
) -> List[List[str]]:
    """
    Find circular imports in the dependency graph.
    
    Args:
        graph: The dependency graph
        
    Returns:
        List of circular import chains
    """
    try:
        # Find all simple cycles in the graph
        cycles = list(nx.simple_cycles(graph))
        
        # Sort cycles for consistent output
        cycles.sort(key=lambda c: (len(c), c[0]))
        return cycles
    except Exception as e:
        print(f"Error detecting cycles: {e}")
        return []

def generate_module_import_report(
    modules: Dict[str, ModuleInfo],
    module_imports: Dict[str, List[Import]],
    specific_modules: Optional[List[str]] = None
) -> str:
    """
    Generate a detailed report of module imports.
    
    Args:
        modules: Dict mapping module names to module information
        module_imports: Dict mapping module names to their imports
        specific_modules: Optional list of modules to focus on
    
    Returns:
        Formatted report string
    """
    lines = ["Module Import Report", "===================", ""]
    
    # Filter modules if specific ones requested
    target_modules = specific_modules or sorted(modules.keys())
    
    for module_name in target_modules:
        if module_name not in modules:
            lines.append(f"Module not found: {module_name}")
            continue
            
        module_info = modules[module_name]
        imports = module_imports.get(module_name, [])
        
        lines.append(f"Module: {module_name}")
        lines.append(f"File: {module_info.file_path}")
        lines.append("Imports:")
        
        if not imports:
            lines.append("  No imports")
        else:
            for imp in imports:
                if imp.is_relative:
                    rel_str = f"relative (level {imp.level})"
                    resolved = resolve_relative_import(imp, module_info)
                    lines.append(f"  Line {imp.line}: from {'.' * imp.level}{imp.module} import {', '.join(imp.names)} ({rel_str} -> {resolved})")
                else:
                    if imp.module:
                        lines.append(f"  Line {imp.line}: from {imp.module} import {', '.join(imp.names)}")
                    else:
                        lines.append(f"  Line {imp.line}: import {', '.join(imp.names)}")
        
        lines.append("")
    
    return "\n".join(lines)

def highlight_cycles(
    graph: nx.DiGraph,
    cycles: List[List[str]],
    pos: Dict[str, Tuple[float, float]],
    ax: plt.Axes
) -> None:
    """
    Highlight cycles in the graph visualization.
    
    Args:
        graph: The dependency graph
        cycles: List of cycles to highlight
        pos: Node positions for the graph
        ax: Matplotlib Axes object
    """
    cycle_nodes = set()
    cycle_edges = []
    
    # Collect all nodes and edges involved in cycles
    for cycle in cycles:
        for i, node in enumerate(cycle):
            cycle_nodes.add(node)
            next_node = cycle[(i + 1) % len(cycle)]
            cycle_edges.append((node, next_node))
    
    # Draw cycle edges
    nx.draw_networkx_edges(
        graph, pos, 
        edgelist=cycle_edges,
        edge_color='red',
        width=2.0,
        arrows=True,
        ax=ax
    )
    
    # Draw cycle nodes
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=list(cycle_nodes),
        node_color='lightcoral',
        node_size=700,
        alpha=0.8,
        ax=ax
    )

def visualize_dependency_graph(
    graph: nx.DiGraph,
    cycles: List[List[str]],
    output_path: Optional[str] = None,
    title: str = "Python Module Dependencies",
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Create a visualization of the dependency graph.
    
    Args:
        graph: The dependency graph
        cycles: List of detected cycles
        output_path: Path to save the visualization (if None, display instead)
        title: Title for the visualization
        figsize: Figure dimensions
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Try different layouts based on graph size
    if len(graph) > 50:
        # For larger graphs, spectral layout often works better
        pos = nx.spectral_layout(graph)
        # Refine with force-directed layout
        pos = nx.spring_layout(graph, pos=pos, k=0.2, iterations=50)
    else:
        # For smaller graphs, spring layout with more iterations
        pos = nx.spring_layout(graph, k=0.3, iterations=100, seed=42)
    
    # Draw regular nodes and edges
    nx.draw_networkx_nodes(
        graph, pos,
        node_color='lightblue',
        node_size=500,
        alpha=0.8,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='gray',
        width=1.0,
        alpha=0.6,
        arrows=True,
        ax=ax
    )
    
    # Highlight cycles
    if cycles:
        highlight_cycles(graph, cycles, pos, ax)
    
    # Add labels with optional line numbers
    labels = {}
    for node in graph.nodes():
        labels[node] = graph.nodes[node].get('label', node)
    
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=9,
        font_family='sans-serif',
        ax=ax
    )
    
    # Add title and other information
    cycle_info = f"{len(cycles)} circular dependencies found" if cycles else "No circular dependencies"
    plt.title(f"{title}\n({len(graph)} modules, {graph.size()} dependencies, {cycle_info})")
    plt.axis('off')
    
    # Add legend
    if cycles:
        circle1 = plt.Circle((0, 0), 0.1, color='lightcoral', alpha=0.8)
        circle2 = plt.Circle((0, 0), 0.1, color='lightblue', alpha=0.8)
        plt.legend([circle1, circle2], ['Circular dependency', 'Normal module'], loc='lower right')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def get_cycle_details(
    graph: nx.DiGraph,
    cycles: List[List[str]],
    modules: Dict[str, ModuleInfo],
    module_imports: Dict[str, List[Import]]
) -> str:
    """
    Generate detailed information about circular imports.
    
    Args:
        graph: The dependency graph
        cycles: List of detected cycles
        modules: Dict mapping module names to module information
        module_imports: Dict mapping module names to their imports
        
    Returns:
        Formatted report string
    """
    if not cycles:
        return "No circular imports detected."
    
    lines = ["Circular Import Details", "======================", ""]
    
    for i, cycle in enumerate(cycles):
        lines.append(f"Circular Import #{i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
        lines.append("-" * 80)
        
        # Show the specific import statements creating the cycle
        for j, module in enumerate(cycle):
            next_module = cycle[(j + 1) % len(cycle)]
            
            # Find the import statement from module to next_module
            imports = module_imports.get(module, [])
            if not imports:
                lines.append(f"  {module} -> {next_module}: [Import information not available]")
                continue
            
            found = False
            for imp in imports:
                resolved_module = imp.module
                if imp.is_relative:
                    resolved_module = resolve_relative_import(imp, modules[module])
                
                # Check if this import contributes to the cycle
                if next_module == resolved_module or next_module.startswith(f"{resolved_module}."):
                    if imp.is_relative:
                        lines.append(f"  {module} -> {next_module}: Line {imp.line}: from {'.' * imp.level}{imp.module} import {', '.join(imp.names)}")
                    else:
                        if imp.module:
                            lines.append(f"  {module} -> {next_module}: Line {imp.line}: from {imp.module} import {', '.join(imp.names)}")
                        else:
                            lines.append(f"  {module} -> {next_module}: Line {imp.line}: import {', '.join(imp.names)}")
                    found = True
                    break
            
            if not found:
                # This should not happen with a proper graph, but handle it anyway
                edge_data = graph.get_edge_data(module, next_module)
                line_num = edge_data.get('line', '?') if edge_data else '?'
                lines.append(f"  {module} -> {next_module}: Line {line_num}: [Import statement not found]")
        
        lines.append("")
        lines.append("Suggested solutions:")
        lines.append("  1. Use TYPE_CHECKING in one of the modules:")
        lines.append("     ```python")
        lines.append("     from typing import TYPE_CHECKING")
        lines.append("     if TYPE_CHECKING:")
        lines.append("         from .other_module import SomeClass")
        lines.append("     ```")
        lines.append("  2. Move shared type definitions to a separate module")
        lines.append("  3. Restructure the modules to eliminate the circular dependency")
        lines.append("")
    
    return "\n".join(lines)

def identify_core_modules(
    cycles: List[List[str]],
    min_occurrences: int = 2
) -> List[str]:
    """
    Identify modules that appear in multiple circular dependencies.
    
    Args:
        cycles: List of detected cycles
        min_occurrences: Minimum number of cycles a module must appear in
        
    Returns:
        List of core problematic modules
    """
    if not cycles:
        return []
    
    # Count occurrences of each module
    module_counts = defaultdict(int)
    for cycle in cycles:
        for module in cycle:
            module_counts[module] += 1
    
    # Filter modules that appear in multiple cycles
    core_modules = [module for module, count in module_counts.items() 
                    if count >= min_occurrences]
    
    # Sort by occurrence count (descending)
    core_modules.sort(key=lambda m: module_counts[m], reverse=True)
    
    return core_modules

def main():
    """Main function to run the dependency analyzer."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        project_dir = Path(sys.argv[1])
    else:
        project_dir = Path.cwd()
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = "dependency_graph.svg"
    
    print(f"Analyzing Python dependencies in: {project_dir}")
    print("Scanning for Python files...")
    
    # Build dependency graph
    graph, modules, module_imports, errors = build_dependency_graph(project_dir)
    
    # Find circular imports
    print("Detecting circular imports...")
    cycles = find_circular_imports(graph)
    
    # Print summary
    print(f"\nFound {len(graph)} Python modules with {graph.size()} dependencies.")
    print(f"Detected {len(cycles)} circular import chains.")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors while parsing files:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"- {error}")
        if len(errors) > 5:
            print(f"... and {len(errors) - 5} more errors.")
    
    # Generate visualization
    print("\nGenerating visualization...")
    visualize_dependency_graph(graph, cycles, output_path)
    
    # If circular imports are found, show details
    if cycles:
        print("\n" + "=" * 80)
        print("CIRCULAR IMPORTS DETECTED")
        print("=" * 80)
        
        cycle_details = get_cycle_details(graph, cycles, modules, module_imports)
        print(cycle_details)
        
        # Identify core problematic modules
        core_modules = identify_core_modules(cycles)
        if core_modules:
            print("\nCore problematic modules (appearing in multiple circular dependencies):")
            for module in core_modules:
                print(f"- {module}")
            
            # Generate detailed report for core modules
            print("\nGenerating detailed report for core problematic modules...")
            core_report = generate_module_import_report(modules, module_imports, core_modules)
            core_report_path = "core_modules_report.txt"
            with open(core_report_path, "w", encoding="utf-8") as f:
                f.write(core_report)
            print(f"Report saved to {core_report_path}")
    
    # Generate full report
    full_report_path = "dependency_report.txt"
    with open(full_report_path, "w", encoding="utf-8") as f:
        # Write summary
        f.write(f"Python Dependency Analysis Report\n")
        f.write(f"===============================\n\n")
        f.write(f"Project: {project_dir}\n")
        f.write(f"Modules: {len(graph)}\n")
        f.write(f"Dependencies: {graph.size()}\n")
        f.write(f"Circular Imports: {len(cycles)}\n\n")
        
        # Write cycle details if any
        if cycles:
            f.write(cycle_details)
            f.write("\n\n")
        
        # Write full module report
        f.write(generate_module_import_report(modules, module_imports))
    
    print(f"\nFull dependency report saved to {full_report_path}")
    print(f"Analysis complete!")

if __name__ == "__main__":
    main()