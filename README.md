# REPARTO DE TAREAS
## BILBO
1. Seguir investigando approaches como el de la fisica estadística
## MOLI
1. Crear una función para extraer datos a nivel de 5 minutos utilizando la API de interactive brokers con un buen historico (alrededor de 6 años estaría bien)
2. Una vez creada la función extraer datos de diferentes activos y guardarlos en una carpeta en formato csv. (top empresas s&p500, oro y plata, bitcoin y ethereum, divisas como USD-EUR.
3. Ayudar a Unai con temas de VectorBT y vectorización
## UNAI
1. Seguir creando un pipeline completo con VectorBT desde optimizacion con algoritmos geneticos, hasta el calculo de profits, visualización y testeo de overfitting (comparando con otros modelos random, buy and hold, etc)

# SIGUIENTES PASOS
1. Investigar optimización de portfolios con paquetes como PyFolio
2. Investigar si podemos añadir los impuestos de hacienda sobre ganancias en la optimización


# Pending tasks:

1. Write new code to be able to handle more intervals in extraction code
2. Add arguments when calling to the script to edit them from bash instead of doing it from the script.
3. Check why some logs are duplicated


# Crontab

The sleep command is used in order to avoid the script being executed before the data is uploaded. This happens rarely, but it is a case that must be handled.

    * * * * * sleep 10 && cd /path/to/script/ && bash run.sh # Every minute
    */30 * * * * sleep 10 && cd /path/to/script/ && bash run.sh # Every 30 minutes
    
# Biblio

## 1D denoising
   - https://www.sciencedirect.com/topics/computer-science/signal-denoising
   - https://pypi.org/project/noisereduce/
   - https://towardsdatascience.com/noise-cancellation-with-python-and-fourier-transform-97303314aa71
