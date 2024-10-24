import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import gc

# Load the preprocessed data
dataset_path = "/content/preprocessed_recipes.csv"
df = pd.read_csv(dataset_path)
df_sampled = df.sample(frac=0.5, random_state=42)
# Separate features (ingredients) and labels (recipe_title)
X = df["ingredient_tokens"].values
y = df["name_tokens"].values

# Convert the labels to one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
# Build the model
model = Sequential()
# Add layers as needed, e.g., Dense, Dropout, etc.

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


def create_dataset(X, y_encoded, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y_encoded))
    dataset = dataset.batch(batch_size)
# Train the model
from tensorflow.keras.utils import to_categorical
model.fit(X, y_encoded, batch_size=32, epochs=10, validation_split=0.1)