import tensorflow as tf

model_path = "models/tvb_hksl_dev_no_masking.keras"
tflite_save_path = "models/tvb_hksl_dev.tflite"

# Step 1: Load your saved TensorFlow model
# If your model is a SavedModel:
model = tf.keras.models.load_model(model_path)

# Step 2: Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Step 3: Save the converted model to a .tflite file
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)