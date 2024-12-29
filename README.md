# TFM

En este documento no se explican los pasos a seguir para replicar la ejecución del proyecto debido a que preparar el entorno de todas las tecnologías que se emplean este trabajo es un proceso largo y complejo. Sirve a modo de exposición.

Los datos del proyecto se han tomado del siguiente proyecto de Kaggle: https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft/data

En el repositorio se pueden encontrar los archivos de código utilizados para llevar a cabo el proyecto. A continuación se resume la utilidad de cada uno:

- **Kafka Producer**: Es el código del proyecto de consola de .NET (escrito en C#) que actúa como productor de Kafka para enviar datos desde SQL Server a Kafka.

- **SQL**: Es el código del proyecto de consola de .NET (escrito en C#) para hacer el trasvase de datos de los archivos CSV a las tablas de SQL Server.

- **modelo_machine_learning.keras**: Es el modelo de Machine Learning que usa Spark en formato `.keras`.

- **Preprocesamiento-Entrenamiento.py**: Es el archivo de Python desarrollado en Kaggle para preprocesar y entrenar los modelos de entrenamiento.

- **docker-compose.yml**: Archivo de configuración de Docker donde se ejecutan Kafka y Cassandra.

- **streaming-job.py**: Archivo que ejecuta Spark para preprocesar los datos que envía Kafka y ejecutar el modelo.
