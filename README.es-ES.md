# QRGAT
Código fuente de "Question-directed Reasoning with Relation-aware Graph Attention Network for Complex Question Answering over Knowledge Graph"

## Datos preprocesados
Hay dos conjuntos de datos utilizados en este trabajo: WebQuestionsSP y ComplexWebQuestions.
Los datos preprocesados se pueden acceder directamente en [este enlace](https://drive.google.com/file/d/1OL45pvg5EflFI8wqYu8MJa48spnYKgRD/view?usp=sharing) (~1.05 GB).

## Registros de experimentos y pesos del modelo
Todos los registros de entrenamiento de los experimentos del artículo y los puntos de control del modelo correspondientes se pueden acceder en [este enlace](https://drive.google.com/file/d/19aGf1OF2XrQI75yjn54BoLkvdt9mdP47/view?usp=sharing) (~3.14 GB).

## Comandos y reproducción de resultados
Todos los comandos están listados en el directorio `scripts`. Antes de volver a ejecutar el proceso de entrenamiento, asegúrese de que los datos preentrenados estén descargados y descomprimidos en el directorio `datasets`; antes de realizar la inferencia, asegúrese de que los puntos de control del modelo estén descargados y descomprimidos en el directorio `cache`.

Se pueden volver a ejecutar los procesos de entrenamiento mediante los scripts de shell con `train` en su nombre, o reproducir los resultados del experimento mediante scripts con `inference` en su nombre.


```
git clone https://github.com/zxgx/QRGAT.git
cd QRGAT

# descargar los datos preprocesados
tar -zxf preprocessed_data.tgz -C <data_dir>
cd datasets
ln -s <data_dir>/CWQ CWQ
ln -s <data_dir>/webqsp webqsp
cd ..

# descargar los puntos de control del modelo
tar -zxf qrgat.tgz -C cache

bash scripts/<any shell script>
```

# Actualización sobre EL
Para obtener los resultados con enlace de entidades (EL), utilizamos los resultados de EL de [RnG-KBQA](https://github.com/salesforce/rng-kbqa) para WebQSP.
Los scripts para anotar nuestros conjuntos de datos con enlace de entidades están en el directorio `datasets`.
Para ejecutar esta configuración, por favor (1) reemplace el enlace simbólico del conjunto de datos (explicado a continuación) con el conjunto de datos descargado y (2) use `--enable_entity_linking`

# Cita
Si encuentra este repositorio útil, por favor cite:
```
@article{zhang2024question,
  title={Question-Directed Reasoning With Relation-Aware Graph Attention Network for Complex Question Answering Over Knowledge Graph},
  author={Zhang, Geng and Liu, Jin and Zhou, Guangyou and Zhao, Kunsong and Xie, Zhiwen and Huang, Bo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2024},
  publisher={IEEE}
}
```
