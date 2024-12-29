# TFM

En este documento no se explican los pasos a seguir para replicar la ejecución del proyecto debido a que preparar el entorno de todas las tecnologías que se emplean este trabajo es un proceso largo y complejo. Sirve a modo de exposición.

Los datos del proyecto se han tomado del siguiente proyecto de Kaggle: https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft/data

En el repositorio se pueden encontrar los archivos de código utilizados para llevar a cabo el proyecto. A continuación se resume la utilidad de cada uno:

-Kafka Producer: es el código del proyecto de consola de .net (está escrito en C#) que actúa como productor de Kafka para enviar datos desde SQL Server a Kafka.
-SQL: es el código del proyecto de consola de .net (está escrito en C#) para hacer el trasvase de datos de los archivos csv a las tablas de SQL Server.
-modelo_machine_learning.keras: es el modelo machine learning que usa spark en formato .keras.

 
