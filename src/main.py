import numpy as np
import argparse
import sys
from typing import List, Tuple, Dict, Optional


class VigaContinua:
    def __init__(self, longitudes: List[float], rigideces: List[float]):
        """Crear viga continua con longitudes y rigideces por elemento"""
        if len(longitudes) != len(rigideces):
            raise ValueError(
                "Longitudes y rigideces deben tener el mismo tamaño")

        self.longitudes = np.array(longitudes)
        self.rigideces = np.array(rigideces)
        self.num_elementos = len(longitudes)
        self.num_nudos = self.num_elementos + 1
        self.num_gdl = self.num_nudos * 2  # 2 GDL por nudo

        # Datos de cargas y apoyos
        self.cargas_puntuales = []  # (nudo, fuerza, momento)
        self.cargas_distribuidas = []  # (elemento, q1, q2)
        self.apoyos = []  # (nudo, rest_v, rest_m)

    # -------------------------------
    # Métodos para agregar condiciones
    # -------------------------------
    def agregar_carga_puntual(self, nudo: int, fuerza: float = 0, momento: float = 0):
        self.cargas_puntuales.append((nudo, fuerza, momento))

    def agregar_carga_distribuida(self, elemento: int, q1: float, q2: Optional[float] = None):
        if q2 is None:
            q2 = q1
        self.cargas_distribuidas.append((elemento, q1, q2))

    def agregar_apoyo_simple(self, nudo: int):
        self.apoyos.append((nudo, True, False))

    def agregar_empotramiento(self, nudo: int):
        self.apoyos.append((nudo, True, True))

    # -------------------------------
    # Métodos internos
    # -------------------------------
    def _matriz_elemento(self, L: float, EI: float) -> np.ndarray:
        """Matriz de rigidez de un elemento de viga"""
        a = 12 * EI / (L**3)
        b = 6 * EI / (L**2)
        c = 4 * EI / L
        d = 2 * EI / L
        return np.array([[a, b, -a, b],
                         [b, c, -b, d],
                         [-a, -b, a, -b],
                         [b, d, -b, c]])

    def _gdl_elemento(self, elem: int) -> List[int]:
        return [elem * 2, elem * 2 + 1, (elem + 1) * 2, (elem + 1) * 2 + 1]

    def _matriz_global(self) -> np.ndarray:
        K = np.zeros((self.num_gdl, self.num_gdl))
        for i in range(self.num_elementos):
            K_elem = self._matriz_elemento(
                self.longitudes[i], self.rigideces[i])
            gdl = self._gdl_elemento(i)
            for p in range(4):
                for q in range(4):
                    K[gdl[p], gdl[q]] += K_elem[p, q]
        return K

    def _fuerzas_equivalentes_distribuidas(self, L: float, q1: float, q2: float):
        f_izq = L * (7 * q1 + 3 * q2) / 20
        f_der = L * (3 * q1 + 7 * q2) / 20
        m_izq = L**2 * (3 * q1 + 2 * q2) / 60
        m_der = -(L**2) * (2 * q1 + 3 * q2) / 60
        return f_izq, m_izq, f_der, m_der

    def _vector_fuerzas(self) -> np.ndarray:
        F = np.zeros(self.num_gdl)

        # Puntuales
        for nudo, fuerza, momento in self.cargas_puntuales:
            gdl_v = (nudo - 1) * 2
            gdl_m = (nudo - 1) * 2 + 1
            F[gdl_v] += fuerza
            F[gdl_m] += momento

        # Distribuidas
        for elemento, q1, q2 in self.cargas_distribuidas:
            L = self.longitudes[elemento - 1]
            f_izq, m_izq, f_der, m_der = self._fuerzas_equivalentes_distribuidas(
                L, q1, q2)
            gdl = self._gdl_elemento(elemento - 1)
            F[gdl[0]] += f_izq
            F[gdl[1]] += m_izq
            F[gdl[2]] += f_der
            F[gdl[3]] += m_der

        return F

    def _gdl_restringidos(self) -> List[int]:
        restringidos = []
        for nudo, rest_v, rest_m in self.apoyos:
            base = (nudo - 1) * 2
            if rest_v:
                restringidos.append(base)
            if rest_m:
                restringidos.append(base + 1)
        return restringidos

    # -------------------------------
    # Resolución
    # -------------------------------
    def resolver(self) -> Dict[str, np.ndarray]:
        K = self._matriz_global()
        F = self._vector_fuerzas()
        gdl_rest = self._gdl_restringidos()

        todos_gdl = set(range(self.num_gdl))
        gdl_libres = sorted(todos_gdl - set(gdl_rest))

        if len(gdl_libres) > 0:
            K_lib = K[np.ix_(gdl_libres, gdl_libres)]
            F_lib = F[gdl_libres]
            U_lib = np.linalg.solve(K_lib, F_lib)

            U = np.zeros(self.num_gdl)
            U[gdl_libres] = U_lib
        else:
            U = np.zeros(self.num_gdl)

        R = np.zeros(len(gdl_rest))
        if len(gdl_rest) > 0:
            K_rest = K[gdl_rest, :]
            R = K_rest @ U - F[gdl_rest]

        return {"desplazamientos": U, "reacciones": R}


# -------------------------------
# Programa principal
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Análisis Matricial de Vigas")
    parser.add_argument("-ej", "--ejemplo",
                        action="store_true", help="Ejecutar un ejemplo")
    args = parser.parse_args()

    if args.ejemplo:
        # Ejemplo: viga de 1 tramo con carga puntual
        viga = VigaContinua([5.0, 6.0], [200, 500])
        viga.agregar_apoyo_simple(1)
        viga.agregar_empotramiento(3)
        viga.agregar_carga_distribuida(1, -200, -400)
        viga.agregar_carga_puntual(2, -10.0)

        resultados = viga.resolver()
        print("Desplazamientos:", resultados["desplazamientos"])
        print("Reacciones:", resultados["reacciones"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
