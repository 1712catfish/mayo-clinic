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
    'Importing dependencies...',
    'Setting up light variables...',
    'Setting up heavy variables...',
    'Building utilities...',
    'Running distributed data preprocessing...'
]

CODE = load_code(path=PATH, modules=MODS)
print(generate_run_script(
    path=PATH,
    scripts=MODS,
    run_description=DESC,
))
