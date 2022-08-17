from pathlib import Path
import os


def load_code(path='.', modules=None):
    codes = dict()
    for module in Path(path).glob('*.py'):
        if modules is None or module.stem in modules:
            with open(module, 'r') as f:
                codes[module.stem] = f.read()
    return codes


def generate_condensed_run_script(code_string_vars, run_description=None):
    run_script = []
    if run_description is None:
        run_description = code_string_vars
    for i, (code_string_var, description) in enumerate(zip(code_string_vars, run_description)):
        run_script.append(f"print('{description}'); exec({code_string_var})")
    return '\n'.join(run_script)


def generate_run_script(path, scripts, prefixes=None, postfixes=None, capitalize=True):
    run_scripts = []
    if prefixes is None:
        prefixes = [f"{script.capitalize() if capitalize else script}: " for script in scripts]
    if postfixes is None:
        postfixes = ['Finished.'] * len(scripts)
    if isinstance(postfixes, str):
        postfixes = [postfixes] * len(scripts)
    for prefix, script, postfix in zip(prefixes, scripts, postfixes):
        run_script = []
        run_script.append(f"print('{prefix}', end='')")
        run_script.append(f"with open('{path}/{script}.py') as f:")
        run_script.append(f"    exec(compile(f.read(), '', 'exec'))")
        run_script.append(f"print('{postfix}')")
        run_script = '\n'.join(run_script)
        run_scripts.append(run_script)
    return '\n\n'.join(run_scripts)
