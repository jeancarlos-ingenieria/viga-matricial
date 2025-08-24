import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import sys
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class VigaContinua:
    def __init__(self, longitudes: List[float], rigideces: List[float]):
        """Crear viga continua con longitudes y rigideces por elemento"""
        if len(longitudes) != len(rigideces):
            raise ValueError("Longitudes y rigideces deben tener el mismo tamaño")

        self.longitudes = np.array(longitudes)
        self.rigideces = np.array(rigideces)
        self.num_elementos = len(longitudes)
        self.num_nudos = self.num_elementos + 1
        self.num_gdl = self.num_nudos * 2  # 2 GDL por nudo (vertical y rotación)

        # Almacenar cargas y apoyos
        self.cargas_puntuales = []  # (nudo, fuerza, momento)
        self.cargas_distribuidas = []  # (elemento, q_inicial, q_final)
        self.apoyos = []  # (nudo, vertical_restringido, momento_restringido)

    def agregar_carga_puntual(self, nudo: int, fuerza: float = 0, momento: float = 0):
        """Agregar carga puntual en un nudo"""
        self.cargas_puntuales.append((nudo, fuerza, momento))

    def agregar_carga_distribuida(
        self, elemento: int, q1: float, q2: Optional[float] = None
    ):
        """Agregar carga distribuida en un elemento (rectangular si q2=None)"""
        if q2 is None:
            q2 = q1
        self.cargas_distribuidas.append((elemento, q1, q2))

    def agregar_apoyo_simple(self, nudo: int):
        """Agregar apoyo simple (solo restringe vertical)"""
        self.apoyos.append((nudo, True, False))

    def agregar_empotramiento(self, nudo: int):
        """Agregar empotramiento (restringe vertical y momento)"""
        self.apoyos.append((nudo, True, True))

    def _matriz_elemento(self, L: float, EI: float) -> np.ndarray:
        """Matriz de rigidez de un elemento de viga"""
        a = 12 * EI / (L**3)
        b = 6 * EI / (L**2)
        c = 4 * EI / L
        d = 2 * EI / L

        return np.array([[a, b, -a, b], [b, c, -b, d], [-a, -b, a, -b], [b, d, -b, c]])

    def _gdl_elemento(self, elem: int) -> List[int]:
        """GDL globales de un elemento (base 0)"""
        return [elem * 2, elem * 2 + 1, (elem + 1) * 2, (elem + 1) * 2 + 1]

    def _matriz_global(self) -> np.ndarray:
        """Ensamblar matriz de rigidez global"""
        K = np.zeros((self.num_gdl, self.num_gdl))

        for i in range(self.num_elementos):
            K_elem = self._matriz_elemento(self.longitudes[i], self.rigideces[i])
            gdl = self._gdl_elemento(i)

            for p in range(4):
                for q in range(4):
                    K[gdl[p], gdl[q]] += K_elem[p, q]
        return K

    def _fuerzas_equivalentes_distribuidas(
        self, L: float, q1: float, q2: float
    ) -> Tuple[float, float, float, float]:
        """Calcular fuerzas equivalentes para carga trapezoidal"""
        # Fórmulas para carga trapezoidal
        f_izq = L * (7 * q1 + 3 * q2) / 20
        f_der = L * (3 * q1 + 7 * q2) / 20
        m_izq = L**2 * (3 * q1 + 2 * q2) / 60
        m_der = -(L**2) * (2 * q1 + 3 * q2) / 60
        return f_izq, m_izq, f_der, m_der

    def _vector_fuerzas(self) -> np.ndarray:
        """Crear vector global de fuerzas"""
        F = np.zeros(self.num_gdl)

        # Cargas puntuales
        for nudo, fuerza, momento in self.cargas_puntuales:
            gdl_v = (nudo - 1) * 2  # GDL vertical
            gdl_m = (nudo - 1) * 2 + 1  # GDL momento
            F[gdl_v] += fuerza
            F[gdl_m] += momento

        # Cargas distribuidas (convertir a fuerzas equivalentes)
        for elemento, q1, q2 in self.cargas_distribuidas:
            L = self.longitudes[elemento - 1]
            f_izq, m_izq, f_der, m_der = self._fuerzas_equivalentes_distribuidas(
                L, q1, q2 # type: ignore
            )

            gdl = self._gdl_elemento(elemento - 1)
            F[gdl[0]] += f_izq  # Fuerza nudo izquierdo
            F[gdl[1]] += m_izq  # Momento nudo izquierdo
            F[gdl[2]] += f_der  # Fuerza nudo derecho
            F[gdl[3]] += m_der  # Momento nudo derecho

        return F

    def _gdl_restringidos(self) -> List[int]:
        """Identificar GDL restringidos por apoyos"""
        restringidos = []
        for nudo, rest_v, rest_m in self.apoyos:
            base = (nudo - 1) * 2
            if rest_v:
                restringidos.append(base)  # GDL vertical
            if rest_m:
                restringidos.append(base + 1)  # GDL rotacional
        return restringidos

    def resolver(self) -> Dict[str, np.ndarray]:
        """Resolver el sistema y calcular desplazamientos y reacciones"""
        print("Resolviendo estructura...")

        # Matrices del sistema
        K = self._matriz_global()
        F = self._vector_fuerzas()
        gdl_rest = self._gdl_restringidos()

        # GDL libres
        todos_gdl = set(range(self.num_gdl))
        gdl_libres = sorted(todos_gdl - set(gdl_rest))

        # Resolver sistema reducido
        if len(gdl_libres) > 0:
            K_lib = K[np.ix_(gdl_libres, gdl_libres)]
            F_lib = F[gdl_libres]
            U_lib = np.linalg.solve(K_lib, F_lib)

            # Vector completo de desplazamientos
            U = np.zeros(self.num_gdl)
            U[gdl_libres] = U_lib
        else:
            U = np.zeros(self.num_gdl)

        # Calcular reacciones
        R = np.zeros(len(gdl_rest))
        if len(gdl_rest) > 0:
            K_rest = K[gdl_rest, :]
            R = K_rest @ U - F[gdl_rest]

        return {
            "desplazamientos": U,
            "reacciones": R,
            "gdl_restringidos": gdl_rest, # type: ignore
            "matriz_K": K,
            "vector_F": F,
        }

    def mostrar_resultados(
        self, resultados: Dict[str, np.ndarray], detallado: bool = False
    ):
        """Mostrar resultados principales"""
        U = resultados["desplazamientos"]
        R = resultados["reacciones"]
        gdl_rest = resultados["gdl_restringidos"]

        print("\n" + "=" * 50)
        print("           RESULTADOS DEL ANÁLISIS")
        print("=" * 50)

        print("\n📏 DESPLAZAMIENTOS NODALES:")
        print("-" * 40)
        for i in range(self.num_nudos):
            nudo = i + 1
            desp_v = U[i * 2] * 1000  # mm
            rot = U[i * 2 + 1] * 1000  # mrad
            print(f"Nudo {nudo:2d}: Δv = {desp_v:8.2f} mm, θ = {rot:8.2f} mrad")

        print("\n⚡ REACCIONES EN APOYOS:")
        print("-" * 40)
        if len(R) > 0:
            for i, gdl in enumerate(gdl_rest):
                nudo = gdl // 2 + 1
                tipo = "Vertical" if gdl % 2 == 0 else "Momento "
                valor = R[i]
                unidad = "kN  " if gdl % 2 == 0 else "kN·m"
                print(f"Nudo {nudo:2d} ({tipo}): {valor:10.2f} {unidad}")
        else:
            print("No hay reacciones (estructura no restringida)")

        if detallado:
            print("\n🔢 MATRIZ DE RIGIDEZ GLOBAL:")
            print("-" * 40)
            K = resultados["matriz_K"]
            print(f"Dimensión: {K.shape[0]}×{K.shape[1]}")
            print(K)

            print("\n📊 VECTOR DE FUERZAS:")
            print("-" * 40)
            F = resultados["vector_F"]
            for i, f in enumerate(F):
                if abs(f) > 1e-10:
                    print(f"F[{i+1}] = {f:.3f}")

    def graficar_deformada(self, resultados: Dict[str, np.ndarray]):
        """Graficar la deformada de la viga"""
        U = resultados["desplazamientos"]

        # Coordenadas de los nudos
        coords = np.zeros(self.num_nudos)
        for i in range(1, self.num_nudos):
            coords[i] = coords[i - 1] + self.longitudes[i - 1]

        # Gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Estructura original
        ax1.plot(coords, np.zeros(len(coords)), "k-", linewidth=2, label="Original")
        ax1.scatter(coords, np.zeros(len(coords)), c="red", s=50, zorder=5)

        # Apoyos
        for nudo, rest_v, rest_m in self.apoyos:
            x = coords[nudo - 1]
            if rest_v and rest_m:  # Empotramiento
                ax1.plot([x, x], [-0.1, 0.1], "k-", linewidth=4)
                ax1.plot([x - 0.05, x + 0.05], [-0.1, -0.1], "k-", linewidth=4)
            elif rest_v:  # Apoyo simple
                ax1.plot([x - 0.05, x, x + 0.05], [-0.05, 0, -0.05], "k-", linewidth=3)

        # Deformada (amplificada para visualización)
        desplazamientos_verticales = U[::2]  # Solo desplazamientos verticales
        max_desp = (
            max(abs(desplazamientos_verticales))
            if max(abs(desplazamientos_verticales)) > 0
            else 1e-6
        )
        factor_amp = max(coords) / (100 * max_desp)  # type: ignore # Factor de amplificación
        desp_amp = desplazamientos_verticales * factor_amp

        ax1.plot(
            coords, desp_amp, "b--", linewidth=2, label=f"Deformada (×{factor_amp:.0f})"
        )
        ax1.scatter(coords, desp_amp, c="blue", s=50, zorder=5)

        # Mostrar valores de desplazamiento
        for i, (x, d) in enumerate(zip(coords, desplazamientos_verticales)):
            ax1.annotate(
                f"{d*1000:.1f}mm",
                (x, desp_amp[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        ax1.set_xlabel("Posición (m)")
        ax1.set_ylabel("Desplazamiento")
        ax1.set_title("Deformada de la Viga")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Diagrama de cargas
        max_carga = 1
        for elemento, q1, q2 in self.cargas_distribuidas:
            max_carga = max(max_carga, abs(q1), abs(q2))
            x1 = coords[elemento - 1]
            x2 = coords[elemento]
            x = np.linspace(x1, x2, 20)
            q = np.linspace(-q1, -q2, 20)  # Negativo para mostrar hacia abajo
            ax2.fill_between(x, 0, q, alpha=0.3, color="red")
            ax2.plot(x, q, "r-", linewidth=2)

        # Cargas puntuales
        for nudo, fuerza, momento in self.cargas_puntuales:
            if fuerza != 0:
                x = coords[nudo - 1]
                escala = max_carga * 0.8
                flecha_altura = (
                    fuerza
                    / max([abs(f) for _, f, _ in self.cargas_puntuales if f != 0])
                ) * escala
                ax2.arrow(
                    x,
                    0,
                    0,
                    -flecha_altura,
                    head_width=max(coords) / 50, # type: ignore
                    head_length=escala / 20,
                    fc="blue",
                    ec="blue",
                    linewidth=2,
                )
                ax2.text(
                    x,
                    -flecha_altura - escala / 10,
                    f"{fuerza}kN",
                    ha="center",
                    va="top",
                    fontweight="bold",
                )

        ax2.set_xlabel("Posición (m)")
        ax2.set_ylabel("Carga (kN/m)")
        ax2.set_title("Cargas Aplicadas")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="k", linewidth=1)

        plt.tight_layout()
        plt.show()

    def exportar_resultados(self, resultados: Dict[str, np.ndarray], archivo: str):
        """Exportar resultados a archivo JSON"""
        datos = {
            "geometria": {
                "longitudes": self.longitudes.tolist(),
                "rigideces": self.rigideces.tolist(),
            },
            "cargas": {
                "puntuales": self.cargas_puntuales,
                "distribuidas": self.cargas_distribuidas,
            },
            "apoyos": self.apoyos,
            "resultados": {
                "desplazamientos": resultados["desplazamientos"].tolist(),
                "reacciones": resultados["reacciones"].tolist(),
                "gdl_restringidos": resultados["gdl_restringidos"],
            },
        }

        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        print(f"✅ Resultados exportados a: {archivo}")


def cargar_viga_json(archivo: str) -> VigaContinua:
    """Cargar viga desde archivo JSON"""
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            datos = json.load(f)

        # Crear viga
        viga = VigaContinua(
            datos["geometria"]["longitudes"], datos["geometria"]["rigideces"]
        )

        # Agregar cargas puntuales
        for nudo, fuerza, momento in datos.get("cargas", {}).get("puntuales", []):
            viga.agregar_carga_puntual(nudo, fuerza, momento)

        # Agregar cargas distribuidas
        for elemento, q1, q2 in datos.get("cargas", {}).get("distribuidas", []):
            viga.agregar_carga_distribuida(elemento, q1, q2)

        # Agregar apoyos
        for nudo, rest_v, rest_m in datos.get("apoyos", []):
            if rest_v and rest_m:
                viga.agregar_empotramiento(nudo)
            elif rest_v:
                viga.agregar_apoyo_simple(nudo)

        print(f"✅ Viga cargada desde: {archivo}")
        return viga

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {archivo}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Formato JSON inválido en {archivo}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Error: Campo requerido {e} no encontrado en {archivo}")
        sys.exit(1)


def crear_viga_interactiva() -> VigaContinua:
    """Crear viga mediante entrada interactiva"""
    print("\n🏗️  CREACIÓN DE VIGA INTERACTIVA")
    print("=" * 40)

    # Geometría
    while True:
        try:
            num_elem = int(input("Número de elementos: "))
            if num_elem > 0:
                break
            print("Debe ser mayor que 0")
        except ValueError:
            print("Por favor ingrese un número entero")

    longitudes = []
    rigideces = []

    print(f"\nIngrese datos para {num_elem} elementos:")
    for i in range(num_elem):
        while True:
            try:
                L = float(input(f"Elemento {i+1} - Longitud (m): "))
                EI = float(input(f"Elemento {i+1} - Rigidez EI (kN·m²): "))
                if L > 0 and EI > 0:
                    longitudes.append(L)
                    rigideces.append(EI)
                    break
                print("Valores deben ser positivos")
            except ValueError:
                print("Por favor ingrese números válidos")

    viga = VigaContinua(longitudes, rigideces)

    # Apoyos
    print(f"\n⚓ APOYOS (Nudos disponibles: 1 a {viga.num_nudos})")
    while True:
        try:
            apoyo_str = input("Nudo para apoyo (Enter para terminar): ").strip()
            if not apoyo_str:
                break

            nudo = int(apoyo_str)
            if 1 <= nudo <= viga.num_nudos:
                tipo = input("Tipo (s=simple, e=empotrado): ").lower()
                if tipo == "s":
                    viga.agregar_apoyo_simple(nudo)
                elif tipo == "e":
                    viga.agregar_empotramiento(nudo)
                else:
                    print("Tipo inválido, use 's' o 'e'")
                    continue
            else:
                print(f"Nudo debe estar entre 1 y {viga.num_nudos}")
        except ValueError:
            print("Ingrese un número válido")

    # Cargas puntuales
    print(f"\n📍 CARGAS PUNTUALES")
    while True:
        try:
            nudo_str = input("Nudo para carga (Enter para terminar): ").strip()
            if not nudo_str:
                break

            nudo = int(nudo_str)
            if 1 <= nudo <= viga.num_nudos:
                fuerza = float(input("Fuerza vertical (kN, - hacia abajo): ") or "0")
                momento = float(input("Momento (kN·m): ") or "0")
                if fuerza != 0 or momento != 0:
                    viga.agregar_carga_puntual(nudo, fuerza, momento)
            else:
                print(f"Nudo debe estar entre 1 y {viga.num_nudos}")
        except ValueError:
            print("Ingrese números válidos")

    # Cargas distribuidas
    print(f"\n📊 CARGAS DISTRIBUIDAS")
    while True:
        try:
            elem_str = input("Elemento para carga (Enter para terminar): ").strip()
            if not elem_str:
                break

            elem = int(elem_str)
            if 1 <= elem <= viga.num_elementos:
                q1 = float(input("Carga inicial (kN/m, - hacia abajo): "))
                q2_str = input("Carga final (Enter = igual a inicial): ").strip()
                q2 = float(q2_str) if q2_str else q1
                viga.agregar_carga_distribuida(elem, q1, q2)
            else:
                print(f"Elemento debe estar entre 1 y {viga.num_elementos}")
        except ValueError:
            print("Ingrese números válidos")

    return viga


def crear_ejemplo_json():
    """Crear archivo de ejemplo en JSON"""
    ejemplo = {
        "geometria": {
            "longitudes": [1.0, 0.8, 2.0, 2.0, 3.0],
            "rigideces": [6000, 6000, 9000, 9000, 4000],
        },
        "cargas": {
            "puntuales": [[1, -20.0, 0], [2, -40.0, 0]],
            "distribuidas": [[3, 0, -30], [4, -30, -30], [5, -30, 0]],
        },
        "apoyos": [[3, True, False], [5, True, False], [6, True, True]],
    }

    with open("ejemplo_viga.json", "w", encoding="utf-8") as f:
        json.dump(ejemplo, f, indent=2, ensure_ascii=False)

    print("📄 Archivo ejemplo_viga.json creado")


def main():
    """Función principal con interfaz CLI"""
    parser = argparse.ArgumentParser(
        description="🏗️  Análisis Matricial de Vigas Continuas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python viga.py -i                     # Modo interactivo
  python viga.py -f viga.json           # Cargar desde archivo
  python viga.py -f viga.json -g        # Cargar y graficar
  python viga.py --ejemplo              # Crear archivo de ejemplo
  python viga.py -f viga.json -o result.json  # Exportar resultados
        """,
    )

    parser.add_argument(
        "-f", "--archivo", type=str, help="Archivo JSON con datos de la viga"
    )
    parser.add_argument(
        "-i",
        "--interactivo",
        action="store_true",
        help="Modo interactivo para crear viga",
    )
    parser.add_argument(
        "-g", "--grafico", action="store_true", help="Mostrar gráfico de la deformada"
    )
    parser.add_argument(
        "-d", "--detallado", action="store_true", help="Mostrar resultados detallados"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Archivo para exportar resultados"
    )
    parser.add_argument(
        "--ejemplo",
        action="store_true",
        help="Crear archivo de ejemplo (ejemplo_viga.json)",
    )

    args = parser.parse_args()

    # Crear archivo de ejemplo
    if args.ejemplo:
        crear_ejemplo_json()
        return

    # Verificar que se especifique una fuente de datos
    if not args.archivo and not args.interactivo:
        print("❌ Debe especificar -f <archivo> o -i para modo interactivo")
        print("   Use --help para ver opciones disponibles")
        sys.exit(1)

    try:
        # Cargar o crear viga
        if args.archivo:
            if not Path(args.archivo).exists():
                print(f"❌ Error: El archivo {args.archivo} no existe")
                print("   Use --ejemplo para crear un archivo de ejemplo")
                sys.exit(1)
            viga = cargar_viga_json(args.archivo)
        else:
            viga = crear_viga_interactiva()

        # Resolver
        resultados = viga.resolver()

        # Mostrar resultados
        viga.mostrar_resultados(resultados, detallado=args.detallado)

        # Exportar si se especifica
        if args.output:
            viga.exportar_resultados(resultados, args.output)

        # Graficar si se solicita
        if args.grafico:
            viga.graficar_deformada(resultados)

    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
