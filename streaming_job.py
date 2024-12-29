import os
import tensorflow as tf
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, collect_list
from pyspark.sql.types import StructType, StructField, FloatType, ArrayType, TimestampType, IntegerType

# Suprimir TensorFlow and Spark logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs

# Empezar sesión
spark = SparkSession.builder \
    .appName("Vibration Prediction") \
    .config("spark.sql.shuffle.partitions", "64") \
    .config("spark.streaming.backpressure.enabled", "true") \
    .config("spark.default.parallelism", "64") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # Suppress Spark logs

# Definir el esquema para el mensahe de Kafka
schema = ArrayType(
    StructType([
        StructField('Vibration_1', FloatType(), True),
        StructField('Vibration_2', FloatType(), True),
        StructField('Vibration_3', FloatType(), True)
    ])
)

# Leer Kafka stream
kafka_stream = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "sensor-data") \
    .option("maxOffsetsPerTrigger", 1) \
    .load()

# Parse Kafka mensaje JSON
parsed_stream = kafka_stream \
    .selectExpr("CAST(value AS STRING) as json_string", "CAST(timestamp AS TIMESTAMP) as kafka_timestamp") \
    .selectExpr("kafka_timestamp", 
                "from_json(json_string, 'array<struct<Vibration_1:float,Vibration_2:float,Vibration_3:float>>') as data") \
    .select('kafka_timestamp', 'data')

# Aplicar FFT para reducir (4096, 3) a (2048,)
def prepare_features(data):
    """Apply FFT to (4096, 3) data and flatten to (2048, ) shape."""
    try:
        data = np.array(data)  # Convertir a NumPy array
        print(f"Original data shape: {data.shape}") 

        # Verificar correct shape (4096, 3)
        if data.shape != (4096, 3):
            data = data[:4096] 
            if data.shape[0] < 4096:
                padding = np.zeros((4096 - data.shape[0], 3))  
                data = np.vstack([data, padding])  

        # Aplicar transformada fourier
        fft_data = np.abs(np.fft.rfft(data, axis=0))[:2048, :]  # (2048, 3)
        
        # Flatten a (2048,)
        flattened_data = fft_data.flatten()[:2048]
        
        if len(flattened_data) < 2048:
            padding = np.zeros(2048 - len(flattened_data)) 
            flattened_data = np.hstack([flattened_data, padding]) 

        print(f"Final prepared data shape: {flattened_data.shape}") 
        return flattened_data.tolist()
    except Exception as e:
        print(f"Error in prepare_features: {e}")
        return np.zeros(2048).tolist()  

prepare_features_udf = udf(prepare_features, ArrayType(FloatType()))

flattened_stream = parsed_stream \
    .withColumn('features', prepare_features_udf(
        col('data')
    ))

# Cargar el modelo de Keras
try:
    model_path = r"C:\Users\ialdabaldetreku\Documents\TFM\Producer\model\modelo_machine_learning.keras"
    model = tf.keras.models.load_model(model_path) 
    print(f"Keras model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    model = None

broadcast_model = spark.sparkContext.broadcast(model)

# Definir UDF para aplicar a modelo
def predict_keras(features):
    """Predict using Keras model, input shape (1, 2048) and output 0 (balanced) or 1 (unbalanced)."""
    try:
        features = np.array(features).reshape(1, 2048)  # Reshape to (1, 2048)
        print(f"Features shape for prediction: {features.shape}")  # Debugging log
        prediction = broadcast_model.value.predict(features)[0][0]  # Get prediction
        print(f"Prediction result: {prediction}")  # Debugging log
        return int(prediction >= 0.5)  # Convert to 0 or 1
    except Exception as e:
        print(f"Error in predict_keras: {e}")  # Log the error
        return -1  

predict_udf = udf(predict_keras, IntegerType())

# Aplicar el modelo
predictions = flattened_stream \
    .withColumn('prediction', predict_udf(
        col('features')
    )) \
    .select('kafka_timestamp', 'prediction')

# Escribir resultados en consola
query = predictions \
    .writeStream \
    .trigger(processingTime="1 second") \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .option("numRows", 20) \
    .start()

query.awaitTermination()