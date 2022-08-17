from utils import *

PATH = 'train'
MODS = [
    'dependencies',
    'setups',
    'heavy_setups',
    'utils',
    'train'
]
DESC = [
    'Importing dependencies: ',
    'Setting up light variables: ',
    'Setting up heavy variables: ',
    'Building utilities: ',
    'Training: '
]

CODE = load_code(path=PATH, modules=MODS)
with open('run.py', 'w+') as f:
    f.write(generate_run_script(
        path=PATH,
        scripts=MODS,
        prefixes=DESC,
        postfixes='Success.'
    ))
