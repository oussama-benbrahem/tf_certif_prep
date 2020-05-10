import tensorflow as tf
print(tf.__version__)

defaults = [tf.int32] * 55
dataset = tf.contrib.data.CsvDataset(['covtype.csv.train'], defaults)

col_names = ['elevation', 'aspect', 'slope']

def _parse_csv_row(*vals):
  soil_type = tf.convert_to_tensor(vals[14:54])
  feat_vals = vals[:10] + (soil_type_t, vals[54])
  features = dict(zip(col_names, feat_vals))
  # from one hot to 0-3 column
  class_label = tf.argmax(row_vals[10:14], axis=0)
  return features, class_label


dataset = dataset.map(_parse_csv_row).batch(64)

print(list(dataset.take(1)))

# Cover_Type / integer / 1 to 7
cover_type = tf.feature_column.categorical_column_with_identity(
    'cover_type', num_buckets=8)
cover_embedding = tf.feature_column.embedding_column(cover_type, dimension=10)

 numeric_features = [tf.feature_column.numeric_column(feat) for feat in numeric_cols]

 # Soil_Type (40 binary columns)
 soil_type = tf.feature_column.numeric_column(soil_type, shape=(40,))

# Defining features
columns = numeric_features + [soil_type, cover_embedding]
feature_layer = tf.keras.layers.DenseFeatures(columns)

# Building a model
model = tf.keras.Sequential([
                             feature_layer,
                             tf.keras.layers.Dense(256),
                             tf.keras.layers.Dense(16),
                             tf.keras.layers.Dense(8),
                             tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(dataset, steps_per_epoch=NUM_TRAIN_EXAMPLES/64)