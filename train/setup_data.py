try:
    INTERACTIVE
except Exception:
    from build_utils.generic_utils import *
    from build_utils.data_utils import *

df = pd.read_csv(TRAIN_CSV)
df["image_path"] = df["image_id"].apply(lambda x: os.path.join(TRAIN_DIR, x + ".jpg"))
train_df, val_df = k_fold_train_test_split(df)

CLASS_WEIGHT = auto_class_weight(train_df.label)
print('Auto class weight:')
for k, v in CLASS_WEIGHT.items():
    print(f"  {k}: {v:.4f}")
CLASS_WEIGHT = {S2I_LBL_MAP[k]: v for k, v in CLASS_WEIGHT.items()}

train_ds = create_dataset(train_df, batch_size=BATCH_SIZE, repeated=True)
val_ds = create_dataset(val_df, ordered=True,
                        batch_size=BATCH_SIZE, drop_remainder=False,
                        cached=False, repeated=False, augmented=False)

test_df = pd.read_csv(TEST_CSV)
test_df["image_path"] = test_df["image_id"].apply(lambda x: os.path.join(TEST_DIR, x + ".jpg"))

N_TRAIN, N_VAL, N_TEST = len(train_df), len(val_df), len(test_df)
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
VALIDATION_STEPS = N_VAL // BATCH_SIZE

print(f'Found {N_TRAIN} in train_ds')
print(f'Found {N_VAL} in val_ds')
print(f'FOUND {N_TEST} in test_ds')
print(f'STEPS_PER_EPOCH:', STEPS_PER_EPOCH)
print(f'VALIDATION_STEPS:', VALIDATION_STEPS)

