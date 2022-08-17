from train.utils.data import *
from train.utils.generic_utils import *

df = pd.read_csv(TRAIN_CSV)
df["image_path"] = df["image_id"].apply(lambda x: os.path.join(IMAGE_DIR, "train", x + ".jpg"))
train_df, val_df = k_fold_train_test_split(df)

class_weight = auto_class_weight(train_df.label)
print('Auto class weight:')
for k, v in class_weight.items():
    print(f"  {k}: {v:.4f}")

N_TRAIN, N_VAL = len(train_df), len(val_df)
train_ds = create_dataset(train_df, batch_size=BATCH_SIZE, repeated=True)
val_ds = create_dataset(val_df, ordered=True,
                        batch_size=BATCH_SIZE, drop_remainder=False,
                        cached=False, repeated=False, augmented=False)

test_df = pd.read_csv(TEST_CSV)
test_df["image_path"] = test_df["image_id"].apply(lambda x: os.path.join(IMAGE_DIR, "test", x + ".jpg"))

steps_per_epoch = N_TRAIN // BATCH_SIZE
validation_step = N_VAL // BATCH_SIZE
