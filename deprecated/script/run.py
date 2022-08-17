print('Importing dependencies: ', end='')
with open('train/dependencies.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Setting up light variables: ', end='')
with open('train/setups.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Setting up heavy variables: ', end='')
with open('train/heavy_setups.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Building utilities: ', end='')
with open('train/utils.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')

print('Training: ', end='')
with open('train/compile.py') as f:
    exec(compile(f.read(), '', 'exec'))
print('Success.')