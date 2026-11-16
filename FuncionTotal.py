import tempfile
import os
import numpy as np
from amplpy import AMPL
import pandas as pd

def exportar_a_ampl(matriz, nombre_parametro="A"):
    """
    Convierte una matriz de coeficientes (lista de listas o DataFrame)
    al formato de datos de AMPL (formato bidimensional).
    """
    if isinstance(matriz, pd.DataFrame):
        # Si es un DataFrame, usamos sus índices y columnas
        filas = matriz.index.tolist()
        columnas = matriz.columns.tolist()
        valores = matriz.values
    else:
        # Si es una lista de listas estándar
        filas = [i+1 for i in range(len(matriz))]
        columnas = [j+1 for j in range(len(matriz[0]))]
        valores = matriz

    # Construcción de la cadena en formato AMPL
    ampl_str = f"param {nombre_parametro} :\n"
    ampl_str += "  " + " ".join(map(str, columnas)) + " :=\n"

    for i, fila_id in enumerate(filas):
        valores_fila = " ".join(map(str, valores[i]))
        ampl_str += f"  {fila_id} {valores_fila}\n"

    ampl_str += ";"
    return ampl_str

# Ejemplo de uso:
coeficientes = [
    [2, 3, -1, 0],
    [1, 0,  2, 5],
    [0, 4, -3, 1]
]

# Generar y mostrar el formato listo para copiar a tu archivo .dat de AMPL
ampl_data = exportar_a_ampl(coeficientes, nombre_parametro="A")
print(ampl_data)




def exportar_vectorCostes_ampl(vector, nombre_parametro="c"):
    """
    Convierte un vector de costos (lista de valores)
    al formato de parámetros unidimensionales de AMPL.
    """
    ampl_str = f"param {nombre_parametro} :=\n"
    for i, valor in enumerate(vector):
        ampl_str += f"  {i+1} {valor}\n"
    ampl_str += ";"
    return ampl_str

# Ejemplo de uso:
costos = [10, 15, -5, 22]

# Generar y mostrar el formato para AMPL con el nombre de función correcto
ampl_costos = exportar_vectorCostes_ampl(costos, nombre_parametro="c")
print(ampl_costos)


def exportar_vectorDisponibilidades_ampl(vector, nombre_parametro="b"):
    """
    Convierte un vector de disponibilidades (lista de valores)
    al formato de parámetros unidimensionales de AMPL.
    """
    ampl_str = f"param {nombre_parametro} :=\n"
    for i, valor in enumerate(vector):
        ampl_str += f"  {i+1} {valor}\n"
    ampl_str += ";"
    return ampl_str

# Ejemplo de uso:
disponibilidades = [100, 250, 180]

ampl_b = exportar_vectorDisponibilidades_ampl(disponibilidades, nombre_parametro="b")
print(ampl_b)

def resolver_programa_lineal(A, b, c, sentido="min"):

    # 1. Validación de entradas
    assert sentido in ("min", "max"), "El sentido debe ser 'min' o 'max'"

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    assert A.ndim == 2, "A debe ser una matriz bidimensional"
    assert b.ndim == 1 and c.ndim == 1
    assert A.shape == (len(b), len(c)), "Dimensiones incompatibles"

    # 2. Conversión a formato AMPL
    formato_A = exportar_a_ampl(A, nombre_parametro="A")
    formato_b = exportar_vectorDisponibilidades_ampl(b, nombre_parametro="b")
    formato_c = exportar_vectorCostes_ampl(c, nombre_parametro="c")

    sentido_ampl = "minimize" if sentido == "min" else "maximize"

    # 3. Instancia independiente de AMPL
    ampl = AMPL()

    # 4. Definición del modelo
    ampl.eval(f"""
        set I;
        set J;

        param c {{J}};
        param A {{I, J}};
        param b {{I}};

        var x {{J}} >= 0;

        {sentido_ampl} costo_total:
            sum {{j in J}} c[j] * x[j];

        subject to restriccion {{i in I}}:
            sum {{j in J}} A[i,j] * x[j] <= b[i];
    """)

    ruta_datos = None

    try:
        # 5. Generación del archivo de datos
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".dat",
            delete=False
        ) as f:

            f.write(f"set I := {' '.join(map(str, range(1, len(b)+1)))};\n")
            f.write(f"set J := {' '.join(map(str, range(1, len(c)+1)))};\n\n")

            f.write(formato_c + "\n\n")
            f.write(formato_b + "\n\n")
            f.write(formato_A + "\n")

            ruta_datos = f.name

        # 6. Carga y resolución
        ampl.read_data(ruta_datos)

        ampl.option["solver"] = "highs"
        ampl.solve()

        estado = ampl.solve_result

        if estado != "solved":
            raise RuntimeError(
                f"No se ha encontrado una solución óptima. Estado: {estado}"
            )

        # 7. Recuperación de resultados
        resultado = ampl.get_objective("costo_total").value()

        x_var = ampl.get_variable("x")

        variables = np.array([
            x_var[j].value()
            for j in range(1, len(c)+1)
        ])

        restricciones = ampl.get_constraint("restriccion")

        duales = np.array([
            restricciones[i].dual()
            for i in range(1, len(b)+1)
        ])

        # 8. Cálculo de holguras
        holguras = b - A @ variables

        if not np.all(holguras >= -1e-7):
            raise RuntimeError("La solución obtenida no es factible.")

        # 9. Diccionario de resultados
        resultado_final = {
            "estado": estado,
            "resultado": resultado,
            "variables": variables,
            "duales": duales,
            "holguras": holguras,
            "tipo": sentido
        }

        return resultado_final

    except Exception as e:
        raise RuntimeError("Error al resolver el programa lineal") from e

    finally:
        if ruta_datos is not None and os.path.exists(ruta_datos):
            os.remove(ruta_datos)