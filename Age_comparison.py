## importing libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

##Load the Dataset

Base_Dir = r"C:\Users\KartikLaptop\Downloads\Deeplearning\Age_Comparison\Directory"

# lables age
image_paths = []
age_labels = []

for filename in tqdm(os.listdir(Base_Dir)):
    if not filename.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(Base_Dir, filename)
    age = int(filename.split("_")[0])
    image_paths.append(image_path)
    age_labels.append(age)  

# convert to dataframe
df = pd.DataFrame({
    "image": image_paths,
    "age": age_labels
})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())
print(df["age"].describe())

#Exploratory Data Analysis
from PIL import Image
img = Image.open(df['image'][0])
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis('off')
plt.title("Sample Face Image")
#plt.show(block=False)

# plot classifications
plt.figure(figsize=(6, 4))
sns.histplot(df['age'], kde=True)
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Age Distribution")
plt.show(block=False)

#plot grid of images
plt.figure(figsize=(10, 10))

files = df.sample(n=25, random_state=None).reset_index(drop=True)

for i, row in files.iterrows():
    plt.subplot(5, 5, i + 1)

    img = load_img(row["image"])
    img = np.array(img)

    plt.imshow(img)
    plt.title(f"Age: {row['age']}")
    plt.axis("off")

plt.tight_layout()
plt.show(block=False)

# extraction functions
def extract_features(images):
    features = []
    for path in tqdm(images):
        img = load_img(path, color_mode="grayscale", target_size=(128, 128))
        img = np.array(img)             # extraction into numpy
        features.append(img)

    features = np.array(features)       # (N, 128, 128)
    features = features.reshape(len(features), 128, 128, 1)  # (N, 128, 128, 1)
    return features

# Normalization
X_CACHE = "X_features.npy"
Y_CACHE = "y_age.npy"

if os.path.exists(X_CACHE) and os.path.exists(Y_CACHE):
    print("Loading cached X and y...")
    X = np.load(X_CACHE)
    y_age = np.load(Y_CACHE)
else:
    print("Extracting features (first run only)...")
    X = extract_features(df["image"])  # grayscale + resize + numpy conversion

    print("Normalizing features...")
    X = X / 255.0                      # normalization happens HERE

    y_age = df["age"].to_numpy()

    np.save(X_CACHE, X)
    np.save(Y_CACHE, y_age)

print("X shape:", X.shape)   # should be (24106, 128, 128, 1)

input_shape = (128, 128, 1)

def create_pair_dataset(X, y, num_pairs=50000):
    X_left, X_right, y_rel = [], [], []

    n = len(X)
    for _ in range(num_pairs):
        i, j = np.random.choice(n, 2, replace=False)

        X_left.append(X[i])
        X_right.append(X[j])

        
        y_rel.append(1 if y[i] < y[j] else 0)

    return (
        np.array(X_left),
        np.array(X_right),
        np.array(y_rel)
    )

# Model creation
inputs = Input((input_shape))
#Convolutional layer
conv_1 = Conv2D(32, kernel_size=(3,3), activation='relu')(inputs) # relu improves model preformance
maxp_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3,3), activation='relu')(maxp_1) 
maxp_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3,3), activation='relu')(maxp_2) 
maxp_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3,3), activation='relu')(maxp_3) 
maxp_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

flatten = Flatten()(maxp_4) # matrix to single dimention vectors

# fully connected layers
dense_1 = Dense(256, activation='linear') (flatten)


dropout_1 = Dropout(0.3) (dense_1)


embedding = Dense(128, activation='relu', name='embedding')(dropout_1)
age_out   = Dense(1, activation='relu', name='age_out')(embedding)

model = Model(inputs=inputs, outputs=age_out)


model.compile(loss= tf.keras.losses.Huber(delta=5.0), optimizer='adam')

#plot model

from tensorflow import keras
keras.utils.plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True
)


img = Image.open("model_architecture.png")
plt.imshow(img)
plt.axis("off")
#plt.show(block=False)

# train model
MODEL_PATH = "age_model.keras"
HISTORY_PATH = "history.npy"

if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(HISTORY_PATH):
        print("Loading saved training history...")
        history_dict = np.load(HISTORY_PATH, allow_pickle=True).item()
    else:
        history_dict = None
else:
    print("Training model...")
    history = model.fit(
        X,
        y_age,
        batch_size=32,
        epochs=15,
        validation_split=0.2
    )

    model.save(MODEL_PATH)
    np.save(HISTORY_PATH, history.history)
    history_dict = history.history


feature_extractor = Model(
    inputs=model.input,
    outputs=model.get_layer("embedding").output
)

print("Feature extractor output shape:", feature_extractor.output_shape)

X_left, X_right, y_rel = create_pair_dataset(X, y_age, num_pairs=30000)

E_left  = feature_extractor.predict(X_left,  batch_size=64)
E_right = feature_extractor.predict(X_right, batch_size=64)


E_pairs = np.concatenate([E_left, E_right], axis=1)

#MLP
from keras.layers import Input
from keras.models import Model

cmp_input = Input(shape=(256,))
x = Dense(128, activation='relu')(cmp_input)
x = Dense(64, activation='relu')(x)
cmp_output = Dense(1, activation='sigmoid')(x)

cmp_model = Model(cmp_input, cmp_output)

cmp_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cmp_model.fit(
    E_pairs,
    y_rel,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)



# Plot Results
if history_dict is not None:
    plt.figure(figsize=(6, 4))
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
  #  plt.show(block=False)

    print("Saved plot as training_loss.png")
else:
    print("No training history available to plot.")

plt.close("all")

# Prediction with Test Data
image_index_1,image_index_2 = np.random.choice(len(X), size=2, replace=False)

img1 = X[image_index_1]
img2 = X[image_index_2]

true_age_1 = y_age[image_index_1]
true_age_2 = y_age[image_index_2]

pred_age_1 = model.predict(img1[np.newaxis, ...])[0][0]
pred_age_2 = model.predict(img2[np.newaxis, ...])[0][0]

#print("Original Age_1:", true_age_1,"Original Age_2:", true_age_2,"Predict Age_1:", pred_age_1,"Predict Age_2:", pred_age_2)

e1 = feature_extractor.predict(img1[np.newaxis, ...])
e2 = feature_extractor.predict(img2[np.newaxis, ...])

pair_emb = np.concatenate([e1, e2], axis=1)
rel = cmp_model.predict(pair_emb)[0][0]

younger = "Left image" if rel > 0.5 else "Right image"
older   = "Right image" if rel > 0.5 else "Left image"


plt.figure(figsize=(8, 4))

# Left image
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img1.squeeze(), cmap="gray")
ax1.axis("off")

ax1.text(
    0.5, 0.95,
    "Younger" if younger == "Left image" else "Older",
    transform=ax1.transAxes,
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    color="yellow",
    bbox=dict(facecolor="black", alpha=0.6, pad=4)
)
# Bottom-center: Original age 1
ax1.text(
    0.5, -0.10,
    f"Original Age: {true_age_1}",
    transform=ax1.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    color="black"
)
# Bottom-center: Predict age 1
ax1.text(
    0.5, -0.20,
    f"Predicted Age: {pred_age_1:.1f}",
    transform=ax1.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    color="blue"
)



# Right image
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(img2.squeeze(), cmap="gray")
ax2.axis("off")

ax2.text(
   0.5, 0.95,
    "Younger" if younger == "Right image" else "Older",
    transform=ax2.transAxes,
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    color="yellow",
    bbox=dict(facecolor="black", alpha=0.6, pad=4)
)
# Bottom-center: Original age 2
ax2.text(
    0.5, -0.10,
    f"Original Age: {true_age_2}",
    transform=ax2.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    color="black"
)

# Bottom-center: Predict age 2
ax2.text(
    0.5, -0.20,
    f"Predicted Age: {pred_age_2:.1f}",
    transform=ax2.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    color="blue"
)


plt.tight_layout()
plt.show()
