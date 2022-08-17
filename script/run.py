print('Importing dependencies...')
with open('train/dependencies.py') as f:
    exec(f.read())
print('Setting up light variables...')
with open('train/setups.py') as f:
    exec(f.read())
print('Setting up heavy variables...')
with open('train/heavy_setups.py') as f:
    exec(f.read())
print('Building utilities...')
with open('train/utils.py') as f:
    exec(f.read())
print('Running distributed data preprocessing...')
with open('train/train.py') as f:
    exec(f.read())
