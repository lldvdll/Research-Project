"""check_names.py -- find names that will raise NameError at runtime.

WHY THIS EXISTS
    make_replay used `optimizer` without declaring it as a parameter. Every other builder had
    it; that one was missed when the optimiser factory was added. Python compiles this
    happily -- a missing global is a RUNTIME error -- so it only surfaced when experiment 12
    reached the replay arm, after backprop had already run. The earlier call-signature audit
    could not see it, because the call site was fine; the DEFINITION was wrong.

    This walks every function, works out what is local to it (parameters, assignments,
    comprehension targets, imports, with/except/for bindings), and reports any name that is
    read but is neither local, nor module-level, nor enclosing-scope, nor a builtin.

    Run it after any edit to src/. It takes under a second and needs nothing installed.

        python experiments/check_names.py
"""
import ast, builtins, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = sorted(list((ROOT / "src").glob("*.py")) + list((ROOT / "experiments").glob("*.py")))
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "self"}


def bound_names(node):
    """Names a function binds: parameters, assignments, imports, loops, with, except,
       comprehension targets, nested defs and classes."""
    out = set()
    a = node.args
    for x in a.posonlyargs + a.args + a.kwonlyargs:
        out.add(x.arg)
    if a.vararg:
        out.add(a.vararg.arg)
    if a.kwarg:
        out.add(a.kwarg.arg)
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out.update(n.names)
    return out


def module_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
    return out


problems = []
for f in TARGETS:
    tree = ast.parse(f.read_text(encoding="utf-8"))
    mod = module_names(tree)

    def own_loads(fn):
        """Name loads belonging to THIS function, not to functions nested inside it.
           Without this, a closure's parameters look undefined to its parent."""
        out, stack = [], list(ast.iter_child_nodes(fn))
        for a in [fn.args]:
            for d in a.defaults + [x for x in a.kw_defaults if x]:
                stack.append(d)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue                      # checked separately, with its own scope
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                out.append(n)
            stack.extend(ast.iter_child_nodes(n))
        return out

    def visit(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = enclosing | bound_names(child)
                for n in own_loads(child):
                    if n.id not in scope and n.id not in mod and n.id not in BUILTINS:
                        problems.append(
                            f"{f.name}:{n.lineno} in {child.name}(): "
                            f"'{n.id}' is read but never defined")
                visit(child, scope)
            else:
                visit(child, enclosing)

    visit(tree, set())

print(f"checked {len(TARGETS)} files")
if problems:
    for p in sorted(set(problems)):
        print("  NameError risk:", p)
    sys.exit(1)
print("  no undefined names")
