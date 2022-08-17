print('Importing dependencies: ', end='')
with open('train/dependencies.py') as f:
    exec(f.read())
print('Success!')

print('Setting up light variables: ', end='')
with open('train/setups.py') as f:
    exec(f.read())
print('Success!')

print('Setting up heavy variables: ', end='')
with open('train/heavy_setups.py') as f:
    exec(f.read())
print('Success!')

print('Building utilities: ', end='')
with open('train/utils.py') as f:
    exec(f.read())
print('Success!')

print('Training: ', end='')
with open('train/train.py') as f:
    exec(f.read())
print('Success!')