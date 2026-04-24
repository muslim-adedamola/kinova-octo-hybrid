import tensorflow_datasets as tfds

data_dir = "your/data/dir"  # change

train_ds = tfds.load(
    "kinova_dataset/default",
    split="train",
    data_dir=data_dir,
)

val_ds = tfds.load(
    "kinova_dataset/default",
    split="val",
    data_dir=data_dir,
)

train_eps = []
for ep in tfds.as_numpy(train_ds):
    ep_dir = ep["episode_metadata"]["episode_dir"].decode("utf-8")
    train_eps.append(ep_dir)

val_eps = []
for ep in tfds.as_numpy(val_ds):
    ep_dir = ep["episode_metadata"]["episode_dir"].decode("utf-8")
    val_eps.append(ep_dir)

print("TRAIN EPISODES:")
for x in train_eps:
    print(x)

print("\nVAL EPISODES:")
for x in val_eps:
    print(x)
