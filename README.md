<h1 align='center'>
  Sistemas de extracción de similitud de fragmentos musicales
</h1>

<p align='center'>
    <img src='https://github.com/ahidalgocenteno/TFG-Antonio-Hidalgo-Centeno/assets/155966566/3bbb3c66-2ad4-436b-aef6-9ba56d9ef738' alt='ua-logo' height='70'>
    <img src='https://github.com/ahidalgocenteno/TFG-Antonio-Hidalgo-Centeno/assets/155966566/b517512c-1421-4730-ab96-0733ac9f5e81' alt='ua-logo' height='70'>
</p>

<p align='center'>
 Código del Trabajo Fin de Grado de Antonio Hidalgo Centeno. Para extracción de similitud de fragmentos musicales con descriptores de audio y aprendizaje automático.
</p>


<h2>Modo de Uso</h2>

<p>Para su correcto funcionamiento, este código requiere de ciertas dependencias, en el archivo requirements.txt se pueden encontrar las librerias necesarias para su ejecucción. Es recomendado utilizar un entorno virtual para su ejeccución. Para ello, puedes ejecutar el siguiente comando (python debe estar instalado):</p> 

<pre>python -m venv &lt;enviroment-name&gt;</pre>

<p>A continuación, debes activar el entorno virtual:</p>

<pre>&lt;enviroment-name&gt;/Scripts/activate</pre>

<p>Una vez creado, instala las dependencias con:</p>

<pre>python -m pip install -r requirements.txt</pre>

<p>Para poder realizar los entrenamientos en tiempos razonables, es recomendado usar GPU. Pytorch trabaja con CUDA, un Toolkit de desarrollo que permite trabajar con tarjetas gráficas de NVIDIA para optimizar la ejecucción del código. CUDA Toolkit Puede ser instalado en diferentes sistemas operativos. La versión de CUDA a instalar dependerá de las espeficicaciones de tu equipo. </p>

<p>Para determinar la versión de CUDA compatible con PyTorch puedes visitar el siguiente enlace: <a href="https://pytorch.org/get-started/locally/"> Instalación de Pytorch en Local </a> </p>

<p>Utilizando Windows y pip, la última versión de CUDA disponible en el momento de escritura es la 12.1. Para instalar la versión de CUDA necesaria puede visitarse el siguiente enlace: <a href="https://developer.nvidia.com/cuda-toolkit-archive"> Instalación de CUDA</a>.  </p>

<P> Una vez instalado CUDA, PyTorch puede ser instalado, con la instrucción generada por la página oficial de PyTorch mencionada anteriormente. PyTorch compatible con CUDA 12.1 puede ser instalado en el entorno virtual con la siguiente instrucción:</p>
<pre>python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121</pre>

<p>Una vez realizado todo lo anterior, puede comprobarse que CUDA y PyTorch funcionan conjuntamente con la siguiente instrucción: </p>

<pre>python check_gpu.py</pre>

<p>Si en la consola aparece el nombre de la gpu, toda la instalación ha ido correctamente. Con lo anterior realizado, puedes comenzar a ejecutar los diferentes módulos del proyecto.</p>

<h3>Descargar Datos</h3>

Se facilita un script para la descarga de los datos, 'download_data.py' que guarda en el directorio actual los datos de GTZAN, incluyendo audios, espectogramas y descriptores de audio. Adeás, realiza el split train, test y val. 

<pre> python download_data.py</pre>

<h3>Redes Neuronales para clasificación</h3>

El script 'main_cls.py'. Carga los datos y entrena el modelo para diferentes datos por clase. Se entrenan los modelos de CNN y CRNN.

<pre> python main_cls.py</pre>

<h3>Redes neuronales siamesas</h3>

Para entrenar las redes neuronales siamesas y obtener su precisión, se facilita el script 'main_siamese.py'. De la misma manera, se emplean las arquitecturas con convolucional y convolucional recurrente.

<pre>python main_siamese.py</pre>

<h3>Redes neuronales siamesas con descriptores de audio</h3>

Finalmente, se puede emplear el script 'main_features.py' para entrenar y evaluar las arquitecturas que cuentan también con la entrada de los descriptores de audio, de la base de datos de GTZAN.

<pre>python main_features.py</pre>
