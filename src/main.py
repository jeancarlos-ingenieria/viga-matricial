"""
viga-matricial: Análisis matricial de vigas en 2D

MIT License

Copyright (c) 2025 Jean C.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import sys

#############################################
######## DATOS DE INGRESO DE LA VIGA ########
#############################################


def cargar_datos():
    """
    Devuelve los datos de entrada de la viga:
    - longitudes: array de longitudes de cada elemento
    - rigideces: array de EI de cada elemento
    - cargas_puntuales: lista de [nudo, fuerza, momento]
    - cargas_distribuidas: lista de [elemento, q1, q2] (q1 y q2 en extremos)
    - apoyos: lista de [nudo, restricción vertical, restricción momento]
    """
    longitudes = np.array([1, 0.8, 2, 2, 3], dtype=float)
    rigideces = np.array([6000, 6000, 9000, 9000, 4000], dtype=float)
    cargas_puntuales = [[1, -20, 0], [2, -40, 0]]  # [nudo, fuerza, momento]
    cargas_distribuidas = [[3, 0, -30], [4, -30, -30], [5, -30, 0]]
    apoyos = [[3, 1, 0], [5, 1, 0], [6, 1, 1]]
    return longitudes, rigideces, cargas_puntuales, cargas_distribuidas, apoyos

#######################################
######## FUNCIONES AUXILIARES #########
#######################################


def obtener_gdl_elementos(nro_elem):
    """
    Devuelve la matriz de grados de libertad para cada elemento.
    Cada elemento tiene 4 GDL: [nudo_i_v, nudo_i_m, nudo_j_v, nudo_j_m]
    """
    gdl = np.zeros([nro_elem, 4], dtype=int)
    for i in range(nro_elem):
        gdl[i][0] = (i + 1) * 2 - 1
        gdl[i][1] = (i + 1) * 2
        gdl[i][2] = (i + 1) * 2 + 1
        gdl[i][3] = (i + 1) * 2 + 2
    return gdl


def procesar_cargas_puntuales(cargas_puntuales):
    """
    Convierte las cargas puntuales en pares [gdl, valor] para el vector global.
    """
    gdl_cargados = []
    for nudo, fuerza, momento in cargas_puntuales:
        if fuerza != 0:
            gdl_cargados.append([int(nudo * 2 - 2), fuerza]
                                )   # Fuerza vertical
        if momento != 0:
            gdl_cargados.append([int(nudo * 2 - 1), momento])  # Momento
    return np.array(gdl_cargados, dtype=float)


def procesar_cargas_distribuidas(cargas_distribuidas, longitudes):
    """
    Calcula las fuerzas equivalentes nodales para cargas distribuidas lineales (trapezoidales).
    Retorna una lista de [elemento, fizq, mizq, fder, mder].
    Fórmulas estándar para viga de 2 apoyos:
        F_izq = l*(7*q1 + 3*q2)/20
        F_der = l*(3*q1 + 7*q2)/20
        M_izq = l^2*(3*q1 + 2*q2)/60
        M_der = -l^2*(2*q1 + 3*q2)/60
    """
    resultados = []
    for elemento, q1, q2 in cargas_distribuidas:
        l = longitudes[elemento - 1]
        fizq = l * (7 * q1 + 3 * q2) / 20
        fder = l * (3 * q1 + 7 * q2) / 20
        mizq = l**2 * (3 * q1 + 2 * q2) / 60
        mder = -l**2 * (2 * q1 + 3 * q2) / 60
        resultados.append([elemento, fizq, mizq, fder, mder])
    return resultados


def sumar_fuerzas_distribuidas_al_vector(f, cargas_distribuidas_eq, gdl):
    """
    Suma las fuerzas equivalentes de cargas distribuidas al vector de fuerzas nodales global.
    """
    for eq in cargas_distribuidas_eq:
        elemento = int(eq[0]) - 1  # El índice de elemento inicia en 0
        gdl_elem = gdl[elemento]
        f[gdl_elem[0] - 1] += eq[1]  # Fuerza izquierda
        f[gdl_elem[1] - 1] += eq[2]  # Momento izquierda
        f[gdl_elem[2] - 1] += eq[3]  # Fuerza derecha
        f[gdl_elem[3] - 1] += eq[4]  # Momento derecha
    return f


def ensamblar_rigidez(nro_elem, gdl, longitudes, rigideces, nro_gdl):
    """
    Ensambla la matriz de rigidez global de la estructura.
    """
    k_estruc = np.zeros([nro_gdl, nro_gdl])
    for i in range(nro_elem):
        l = longitudes[i]
        ei = rigideces[i]
        k_elem = matriz_rigidez_elemento(l, ei)
        gdl_elem = gdl[i]
        for j in range(4):
            for k in range(4):
                fila_dest = gdl_elem[j] - 1
                col_dest = gdl_elem[k] - 1
                k_estruc[fila_dest][col_dest] += k_elem[j][k]
    return k_estruc


def matriz_rigidez_elemento(l, ei):
    """
    Matriz de rigidez para elemento viga con 4 gdl (2 por nodo).
    """
    if l <= 0:
        raise ValueError("La longitud del elemento debe ser positiva")
    if ei <= 0:
        raise ValueError("La rigidez del elemento debe ser positiva")
    a = 12 * ei / l ** 3
    b = 6 * ei / l ** 2
    c = 4 * ei / l
    d = 2 * ei / l
    return np.array([[a, b, -a, b],
                     [b, c, -b, d],
                     [-a, -b, a, -b],
                     [b, d, -b, c]], dtype=float)


def construir_vector_fuerzas(gdl_cargados, nro_gdl):
    """
    Construye el vector de fuerzas nodales a partir de las cargas puntuales.
    """
    f = np.zeros(nro_gdl)
    for gdl, valor in gdl_cargados:
        gdl_int = int(gdl)
        if 0 <= gdl_int < nro_gdl:
            f[gdl_int] += valor
        else:
            print(
                f"Advertencia: GDL {gdl_int} fuera de rango [0, {nro_gdl-1}]")
    return f


def procesar_apoyos(apoyos):
    """
    Procesa los apoyos y devuelve los grados de libertad restringidos.
    """
    gdl_restr = []
    for nudo, rest_v, rest_m in apoyos:
        if rest_v == 1:  # Restricción vertical
            gdl_restr.append(int(nudo * 2 - 2))
        if rest_m == 1:  # Restricción momento
            gdl_restr.append(int(nudo * 2 - 1))
    return np.array(gdl_restr, dtype=int)

#######################################
######## FUNCIONES DE SALIDA ##########
#######################################


def mostrar_matriz_rigidez(k_estruc):
    print("=" * 50)
    print("MATRIZ DE RIGIDEZ GLOBAL")
    print("=" * 50)
    print(k_estruc)


def mostrar_fuerzas_nodales(f):
    print("=" * 50)
    print("VECTOR GLOBAL DE FUERZAS NODALES")
    print("=" * 50)
    print(f)


def mostrar_todo(gdl, gdl_cargados, gdl_restr, k_estruc, f, cargas_distribuidas_eq):
    print("=" * 50)
    print("RESULTADOS DEL ANÁLISIS")
    print("=" * 50)
    print("Grados de libertad por elemento:")
    print(gdl)
    print("\nCargas puntuales procesadas [gdl, valor]:")
    print(gdl_cargados)
    print(f"\nGrados de libertad restringidos: {gdl_restr}")
    print(
        f"\nMatriz de rigidez global ({k_estruc.shape[0]}x{k_estruc.shape[1]}):")
    print(k_estruc)
    print("\nFuerzas equivalentes nodales por cargas distribuidas:")
    for eq in cargas_distribuidas_eq:
        print(
            f"Elemento {eq[0]}: F_izq={eq[1]:.2f}, M_izq={eq[2]:.2f}, F_der={eq[3]:.2f}, M_der={eq[4]:.2f}")
    print("\nVector global de fuerzas nodales (puntuales + distribuidas):")
    print(f)

#######################################
######## RESOLUCION DE LA VIGA ########
#######################################


def main():
    try:
        longitudes, rigideces, cargas_puntuales, cargas_distribuidas, apoyos = cargar_datos()
        nro_elem = len(longitudes)
        nro_gdl = (nro_elem + 1) * 2

        gdl = obtener_gdl_elementos(nro_elem)
        gdl_cargados = procesar_cargas_puntuales(cargas_puntuales)
        gdl_restr = procesar_apoyos(apoyos)
        cargas_distribuidas_eq = procesar_cargas_distribuidas(
            cargas_distribuidas, longitudes)

        k_estruc = ensamblar_rigidez(
            nro_elem, gdl, longitudes, rigideces, nro_gdl)
        f = construir_vector_fuerzas(gdl_cargados, nro_gdl)
        f = sumar_fuerzas_distribuidas_al_vector(
            f, cargas_distribuidas_eq, gdl)

        # CLI: elige qué mostrar
        if len(sys.argv) > 1:
            opcion = sys.argv[1].lower()
        else:
            opcion = "todo"

        if opcion == "matriz":
            mostrar_matriz_rigidez(k_estruc)
        elif opcion == "fuerzas":
            mostrar_fuerzas_nodales(f)
        else:
            mostrar_todo(gdl, gdl_cargados, gdl_restr,
                         k_estruc, f, cargas_distribuidas_eq)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
