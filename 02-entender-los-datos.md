# CRISP-DM — Fase 2: Entender los datos

**Proyecto:** AgroIA — Producción de piña en México (2003–2025)

En la fase anterior definimos el problema de negocio, los usuarios potenciales y los objetivos analíticos de AgroIA. Ahora examinaremos los datos históricos de producción de piña antes de limpiarlos o preparar variables para los modelos de inteligencia artificial.

## Objetivos de aprendizaje

Al finalizar esta actividad podrás:

1. Leer y escribir archivos CSV con `pandas`.
2. Utilizar los atributos y métodos básicos de un DataFrame de `pandas`.
3. Concatenar múltiples DataFrames y guardar el resultado en un archivo CSV.
4. Realizar un análisis exploratorio de los datos.

---

## 1. Recolección de datos

Para estudiar la evolución de la producción de piña en México, utilizaremos los datos oficiales de la **Dirección General del Servicio de Información Agroalimentaria y Pesquera (DGSIAP)**.

La información se encuentra disponible en archivos CSV organizados por año. Por lo tanto, para analizar el periodo **2003–2025**, necesitamos descargar los 23 archivos anuales.

### Descarga de los datos

Accede al portal oficial:

[DGSIAP — Datos Abiertos de Producción Agrícola](https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php)

1. Localiza la sección de descarga de datos agrícolas.
2. Descarga los archivos CSV correspondientes a cada año, desde 2003 hasta 2025.
3. Guarda los archivos en una carpeta llamada `dataset/` para utilizarlos durante el análisis.

---

## 2. Descripción de los datos

Antes de integrar los archivos, examinaremos su estructura para identificar las variables disponibles y verificar que los datos sean compatibles.

### 2.1. Leer un archivo CSV

Utilizaremos la biblioteca `pandas` para cargar uno de los archivos en un DataFrame.

```python
import pandas as pd

df = pd.read_csv("dataset/Cierre_agricola_mun_2003.csv", encoding="latin-1")
```

Un **DataFrame** es un objeto de la clase `pandas.DataFrame` que almacena datos en una estructura tabular de filas y columnas.

Algunos de sus atributos más útiles son:

- `shape`: contiene el número de filas (muestras) y columnas (variables).
- `columns`: contiene los nombres de las columnas (encabezados).
- `dtypes`: contiene los tipos de datos de cada columna.

Algunos de sus métodos más útiles son:

- `info()`: muestra un resumen del DataFrame, incluyendo número de registros, columnas, tipos de datos y valores no nulos.
- `head(n)`: muestra los primeros `n` registros.


### 2.2. Identificar encabezados, variables y muestras

El atributo `columns` permite obtener los nombres de las columnas (*headers*), mientras que `shape` devuelve el número de filas y columnas.

```python
print("Encabezados:", df.columns.tolist())
print("Número de variables:", df.shape[1])
print("Número de muestras:", df.shape[0])
```

### 2.3 Acceso a los datos de un DataFrame

Podemos acceder a los datos mediante columnas, filas o celdas utilizando:
- `iloc[]`: permite acceder mediante posiciones numéricas, comenzando desde cero.
- `loc[]`: permite acceder mediante etiquetas de filas y columnas.


**Acceder a una fila por su posición:**

```python
df.iloc[0]  # Primera fila
```

**Acceder a una columna:**

```python
df["Nommunicipio"]
```

**Acceder por fila y nombre de columna:**

```python
df.loc[0, "Nommunicipio"]
```

**Acceder a una celda por su posición:**

```python
df.iloc[0, 2]  # Primera fila, tercera columna
```

**Seleccionar varias filas y columnas:**

```python
df.iloc[0:5, 0:3]  # Primeras 5 filas y 3 columnas
```

**Seleccionar todas las filas y un rango de columnas:**

```python
df.iloc[:, 0:3]  # Todas las filas y primeras 3 columnas
```

### 2.4. Examinar todos los archivos CSV

#### Revisar el número de variables y muestras
Ahora utilizaremos un ciclo `for` para revisar los 23 archivos y comparar sus estructuras.

```python
for anio in range(2003, 2026):
    archivo = f"dataset/Cierre_agricola_mun_{anio}.csv"
    print(f"Leyendo año: {anio}")
    df = pd.read_csv(archivo, encoding="latin-1",low_memory=False)
    print("Encabezados:", df.columns.tolist())
    print("Número de variables:", df.shape[1])
    print("Número de muestras:", df.shape[0])
```

#### Revisar nombres de columnas

Queremos concatenar todos los archivos CSV en un solo DataFrame. Para ello, es importante que todos tengan los mismos nombres de columna.

Cuando `pandas` encuentra encabezados diferentes, los interpreta como variables distintas y crea columnas separadas.

Por ejemplo, si un archivo contiene la columna `Precio` y otro contiene `Preciomediorural`, al concatenarlos se crearán dos columnas:

| Año | Precio | Preciomediorural |
|---|---|---|
| 2003 | 4500 | NaN |
| 2025 | NaN | 6200 |

Los valores `NaN` representan datos faltantes.

Por lo tanto, revisaremos qué archivos tienen encabezados diferentes, tomando como referencia el archivo más reciente (2025).

```python
import pandas as pd

df_2025 = pd.read_csv("dataset/Cierre_agricola_mun_2025.csv", encoding="latin-1")
columnas = df_2025.columns

for anio in range(2003, 2025):
    archivo = f"dataset/Cierre_agricola_mun_{anio}.csv"
    df = pd.read_csv(archivo, encoding="latin-1",low_memory=False)

    if not df.columns.equals(columnas):
        print(f"\nAño {anio}: encabezados diferentes")

        for i in range(len(columnas)):
            if columnas[i] != df.columns[i]:
                print(columnas[i], "->", df.columns[i])
```

Para uniformizar los nombres de las columnas, se utiliza el método `rename` en cada dataFrame:
```python
df = df.rename(columns={
    "Precio": "Preciomediorural",
    "Nomcultivo Sin Um": "Nomcultivo"
})
```

#### Concatenar en un solo archivo CSV

Leeremos cada archivo CSV con `read_csv()` y lo agregaremos a una lista de DataFrames llamada `datos` mediante `append()`. Después, uniremos todos los DataFrames con `pd.concat()` y guardaremos el resultado en un único archivo CSV utilizando `to_csv()`.


```python
import pandas as pd

datos = []

for anio in range(2003, 2026):
    archivo = f"dataset/Cierre_agricola_mun_{anio}.csv"
    df = pd.read_csv(archivo, encoding="latin-1",low_memory=False)

    df = df.rename(columns={
        "Precio": "Preciomediorural",
        "Nomcultivo Sin Um": "Nomcultivo"
    })

    datos.append(df)

df = pd.concat(datos, ignore_index=True)
df.to_csv("dataset/Cierre_agricola_2003_2025.csv", index=False)

print("Número de muestras:", df.shape[0])
print("Número de variables:", df.shape[1])
df.info()
```

- `append()`: agrega cada DataFrame a la lista `datos`.
- `pd.concat()`: concatena los DataFrames por filas.
- `ignore_index=True`: genera un índice consecutivo para el DataFrame resultante.

El resultado es un único DataFrame `df` con los registros de 2003 a 2025, conservando los archivos CSV originales.

---

## 3. Explorar los valores de las variables

Ahora leeremos el archivo CSV que contiene los datos integrados de 2003 a 2025.

```python
import pandas as pd

df = pd.read_csv("dataset/Cierre_agricola_2003_2025.csv",low_memory=False)
df.info()
```

### Estadísticas de las variables numéricas

El método `describe()` calcula estadísticas descriptivas, incluyendo:

- `count`: número de valores no nulos.
- `mean`: promedio.
- `std`: desviación estándar.
- `min`: valor mínimo.
- `25%`, `50%`, `75%`: cuartiles.
- `max`: valor máximo.

Para mostrar las estadisticas de todas las columnas numéricas:
```python
print(df.describe().to_string(float_format="%.2f"))
```
Para un rango de columnas:
```python
print(df.iloc[:, 0:10].describe().to_string(float_format="%.2f"))
```

### Explorar las variables no numéricas

Con `df.info()` identificamos las columnas de tipo `object`. Para explorar sus valores, utilizaremos los siguientes métodos:

- `unique()`: devuelve los valores distintos de una columna.
- `nunique()`: cuenta cuántos valores distintos existen.
- `value_counts()`: cuenta cuántas veces aparece cada valor.

Por ejemplo, para explorar la variable `Nommodalidad`:

```python
print(df["Nommodalidad"].unique())
print(df["Nommodalidad"].nunique())
print(df["Nommodalidad"].value_counts())
```

Repite el procedimiento con las demás variables no numéricas que quieras analizar.

Script para todas las columnas no numéricas:

```python
for columna in df.select_dtypes(exclude="number").columns:
    print("\nVariable:", columna)
    print("Valores:", df[columna].unique())
    print("No. valores diferentes", df[columna].nunique())
#    print("Valores:", df[columna].value_counts())
```

---

## 4. Conclusiones de esta etapa

1. Tenemos 825,089 registros y 24 variables, correspondientes a distintos cultivos entre 2003 y 2025.
2. `Sembrada`, `Volumenproduccion` y `Rendimiento` aparecen como `object`, aunque deberían ser numéricas. Tenemos que identificar por qué.
3. Hay una inconsistencia en `Nomestado`: aparecen `Ciudad de México / DF` y `Ciudad de México` como valores diferentes.
4. Tenemos 453 nombres distintos de cultivos. Todavía no hemos filtrado los registros de piña.











