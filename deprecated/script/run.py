print('Importing dependencies: ', end='')
with open('train/setup_libraries.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Setting up light variables: ', end='')
with open('train/setup_configs.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Setting up heavy variables: ', end='')
with open('train/heavy_setups.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Building utilities: ', end='')
with open('train/train_utils.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Training: ', end='')
with open('train/train_utils.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')