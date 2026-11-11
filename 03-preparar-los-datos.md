# CRISP-DM — Fase 3: Preparar los datos

**Proyecto:** AgroIA — Producción de piña en México (2003–2025)

En la fase anterior identificamos algunas inconsistencias en los datos. Ahora corregiremos estos problemas, seleccionaremos los registros correspondientes a la piña y prepararemos un nuevo dataset para el modelado.

## Objetivos de aprendizaje

Al finalizar esta actividad podrás:

1. Corregir los tipos de datos de un DataFrame.
2. Uniformar valores de variables categóricas.
3. Filtrar y seleccionar registros de interés.
4. Identificar y eliminar valores faltantes, duplicados e inválidos.
5. Guardar los datos preparados en un archivo CSV.

---

## 1. Corregir los tipos de datos

Primero cargamos el archivo CSV.

```python
import pandas as pd

df = pd.read_csv("dataset/Cierre_agricola_2003_2025.csv", low_memory=False)
df.info()
```

Identificamos que `Sembrada`, `Volumenproduccion` y `Rendimiento` aparecen como `object`, aunque contienen cantidades numéricas.

Antes de convertirlas, revisamos si existen valores que no pueden interpretarse como números.

```python
for columna in ["Sembrada", "Volumenproduccion", "Rendimiento"]:
    numeros = pd.to_numeric(df[columna], errors="coerce")
    errores = numeros.isna() & df[columna].notna()

    print(columna, df.loc[errores, columna].unique())
```

Para identificar los registros que contienen el error `#¡NUM!`:

```python
print(df[df["Rendimiento"] == "#¡NUM!"].to_string())
```

### Convertir las variables a tipos numéricos

Para las columnas "Sembrada" y "Volumenproduccion" se eliminan las comas y convertimos las columnas a valores numéricos. 

```python
df["Sembrada"] = pd.to_numeric(df["Sembrada"].astype(str).str.replace(",", ""))
df["Volumenproduccion"] = pd.to_numeric(df["Volumenproduccion"].astype(str).str.replace(",", ""))
```
Para la columna "Rendimiento", el parámetro `errors="coerce"` convierte los valores no válidos, como `#¡NUM!`, en `NaN`.

```python
df["Rendimiento"] = pd.to_numeric(df["Rendimiento"], errors="coerce")
```

Finalmente, verificamos los tipos de datos:

```python
df.info()
```

---

## 2. Uniformar los nombres de las entidades

En la fase anterior encontramos dos nombres para una misma entidad federativa:

- `Ciudad de México / DF`
- `Ciudad de México`

Utilizaremos `replace()` para unificar los nombres.

```python
df["Nomestado"] = df["Nomestado"].replace("Ciudad de México / DF", "Ciudad de México")
```

Verificamos el resultado:

```python
print(df["Nomestado"].unique())
```

---

## 3. Seleccionar los registros de piña

El dataset contiene información de diferentes cultivos. Para nuestro proyecto solamente necesitamos los registros correspondientes a la piña.

Primero eliminamos posibles espacios al inicio y al final del nombre del cultivo.

```python
df["Nomcultivo"] = df["Nomcultivo"].str.strip()
```

Después filtramos los registro y cremos un nuevo DataFrame con los registros seleccionados:

```python
df_pinna = df[df["Nomcultivo"] == "Piña"].copy()
df_pinna.info()
```

---

## 4. Limpieza de datos

### 4.1. Eliminar espacios innecesarios

Para columnas de texto, el método `str.strip()` elimina espacios al inicio y al final de los textos.

```python
df_pinna["Nomestado"] = df_pinna["Nomestado"].str.strip()
df_pinna["Nommunicipio"] = df_pinna["Nommunicipio"].str.strip()
df_pinna["Nommodalidad"] = df_pinna["Nommodalidad"].str.strip()
```
Para eliminar espacios repetidos dentro del texto:
```python
df_pinna["Nommunicipio"] = df_pinna["Nommunicipio"].str.replace(r"\s+", " ", regex=True)
```

### 4.2. Identificar valores faltantes

El método `isna()` identifica los valores faltantes y `sum()` permite contarlos.

```python
print(df_pinna.isna().sum())
```

Podemos eliminar registros que no tengan información en columnas esenciales mediante `dropna()`.

Por ejemplo:

```python
df_pinna = df_pinna.dropna(subset=["Rendimiento", "Preciomediorural"])
```

### 4.3. Eliminar registros duplicados

Primero contamos las filas exactamente duplicadas:

```python
print("Registros duplicados:", df_pinna.duplicated().sum())
```

Después las eliminamos:

```python
df_pinna = df_pinna.drop_duplicates()
```

El método `drop_duplicates()` conserva la primera aparición de cada registro idéntico.


### 4.4. Identificar valores numéricos inválidos

Algunas variables no deberían contener cantidades negativas. Por ejemplo, para revisar los registros inválidos de la variable `Rendimiento`:

```python
indices = df_pinna.index[df_pinna["Rendimiento"] < 0]
print(df_pinna.loc[indices].to_string())
```

Para eliminar esos registros:

```python
df_pinna = df_pinna.drop(index=indices)
```

---

## 5. Guardar el dataset preparado

Finalmente, guardamos los registros de piña en un nuevo archivo CSV utilizando `to_csv()`.

```python
df_pinna.to_csv("dataset/pinna_2003_2025.csv", index=False)
```

El parámetro `index=False` evita agregar el índice del DataFrame como una columna adicional.

---

## 6. Conclusiones

En la fase de preparación de los datos se realizan las siguientes actividades:

1. **Definición del conjunto de datos:** determinar las variables y el periodo de estudio.
2. **Selección de datos:** filtrar los registros relevantes para el proyecto.
3. **Limpieza de datos:** corregir inconsistencias, valores faltantes, duplicados e inválidos.
4. **Generación de datos:** crear nuevas variables a partir de las existentes.
5. **Integración de datos:** combinar información de diferentes fuentes.
6. **Formateo de datos:** convertir tipos de datos, codificar variables categóricas y normalizar valores numéricos.

Estas actividades permiten obtener un conjunto de datos adecuado para la siguiente fase de CRISP-DM: **Modelado**.