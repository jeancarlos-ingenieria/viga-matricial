import numpy as np

#############################################
######## DATOS DE INGRESO DE LA VIGA ########
#############################################

l = [1, 0.8, 2, 2, 3]
e_i = [6000, 6000, 9000, 9000, 4000]
p = [[1, 0, -20], [2, 0, -40]]
q = [[3, 0, -30], [4, -30, -30], [5, -30, 0]]
apoyos = [[3, 1, 0], [5, 1, 0], [6, 1, 1]]

###################################################

# TODO: hacer la logica
gdl_punt = np.array([[1, -20], [3, -40]], dtype=float)
gdl_restr = np.array([5, 9, 11, 12])

# -----------------------------

l = np.array(l, dtype=float)
e_i = np.array(e_i, dtype=float)

#######################################
######## RESOLUCION DE LA VIGA ########
#######################################


## K_ELEMENTO

nro_elem = len(l)
nro_gdl = (nro_elem + 1) * 2

gdl = np.zeros([nro_elem, 4])

for i in range(nro_elem):
    gdl[i][0] = (i + 1) * 2 - 1
    gdl[i][1] = (i + 1) * 2
    gdl[i][2] = (i + 1) * 2 + 1
    gdl[i][3] = (i + 1) * 2 + 2


k_estruc = np.zeros([nro_gdl, nro_gdl])

for i in range(nro_elem):
    a = 12 * e_i[i] / l[i] ** 3
    b = 6 * e_i[i] / l[i] ** 2
    c = 4 * e_i[i] / l[i]
    d = 2 * e_i[i] / l[i]
    k_elem = [[a, b, -a, b], [b, c, -b, d], [-a, -b, a, -b], [b, d, -b, c]]

    # Ensamblaje de la matriz de rigidez de cada elemento
    # dentro de la matriz K de la estructura

    gdl_elem = gdl[i]

    for j in range(4):
        for k in range(4):
            fila_dest = int(gdl_elem[j] - 1)
            col_dest = int(gdl_elem[k] - 1)
            k_estruc[fila_dest][col_dest] = k_elem[j][k] + k_estruc[fila_dest][col_dest]
