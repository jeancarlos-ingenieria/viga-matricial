"""
viga-matricial: Análisis matricial de vigas en 2D

MIT License

Copyright (c) 2025 Jean C.

Este programa realiza el análisis matricial de una viga en 2D, ensamblando la matriz de rigidez global,
procesando cargas puntuales y distribuidas (rectangulares, triangulares, trapezoidales), y mostrando
los resultados de fuerzas nodales, desplazamientos y reacciones.

Arquitectura MVC:
- Models: Clases para datos y cálculos estructurales
- Views: Clases para presentación de resultados
- Controllers: Lógica de control y coordinación

Uso:
    python main.py [opcion]

Opciones:
    matriz          Muestra solo la matriz de rigidez global
    fuerzas         Muestra solo el vector global de fuerzas nodales
    datos           Muestra los datos de entrada
    desplazamientos Muestra los desplazamientos nodales
    reacciones      Muestra las reacciones en los apoyos
    equivalentes    Muestra las fuerzas equivalentes por cargas distribuidas
    todo            Muestra todos los resultados (por defecto)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import numpy.typing as npt
import sys
from colorama import Fore, Style, init
from tabulate import tabulate

# Inicializar colorama
init(autoreset=True)


# =====================================================
# MODELS (Modelos de datos y lógica de negocio)
# =====================================================

@dataclass
class CargaPuntual:
    """Representa una carga puntual aplicada en un nudo."""
    nudo: int
    fuerza: float
    momento: float

    def __post_init__(self):
        if self.nudo <= 0:
            raise ValueError("El número de nudo debe ser positivo")


@dataclass
class CargaDistribuida:
    """Representa una carga distribuida aplicada en un elemento."""
    elemento: int
    q1: float  # Carga en el extremo izquierdo
    q2: float  # Carga en el extremo derecho

    def __post_init__(self):
        if self.elemento <= 0:
            raise ValueError("El número de elemento debe ser positivo")


@dataclass
class Apoyo:
    """Representa un apoyo con sus restricciones."""
    nudo: int
    restriccion_vertical: bool
    restriccion_momento: bool

    def __post_init__(self):
        if self.nudo <= 0:
            raise ValueError("El número de nudo debe ser positivo")


@dataclass
class FuerzaEquivalente:
    """Representa las fuerzas equivalentes de una carga distribuida."""
    elemento: int
    fuerza_izq: float
    momento_izq: float
    fuerza_der: float
    momento_der: float


@dataclass
class PropiedadesEstructura:
    """Encapsula las propiedades geométricas y materiales de la estructura."""
    longitudes: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rigideces: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cargas_puntuales: List[CargaPuntual] = field(default_factory=list)
    cargas_distribuidas: List[CargaDistribuida] = field(default_factory=list)
    apoyos: List[Apoyo] = field(default_factory=list)

    def __post_init__(self):
        self._validar_datos()

    def _validar_datos(self):
        """Valida la consistencia de los datos estructurales."""
        if len(self.longitudes) != len(self.rigideces):
            raise ValueError("Las longitudes y rigideces deben tener la misma cantidad de elementos")
        
        if len(self.longitudes) == 0:
            raise ValueError("Debe haber al menos un elemento en la estructura")

    @property
    def numero_elementos(self) -> int:
        """Retorna el número de elementos de la estructura."""
        return len(self.longitudes)

    @property
    def numero_gdl(self) -> int:
        """Retorna el número total de grados de libertad."""
        return (self.numero_elementos + 1) * 2


class CalculadoraMatricesElemento:
    """Calcula las matrices de rigidez de elementos individuales."""
    
    @staticmethod
    def matriz_rigidez_viga(longitud: float, rigidez: float) -> np.ndarray:
        """
        Calcula la matriz de rigidez de un elemento de viga.
        
        Args:
            longitud: Longitud del elemento
            rigidez: Rigidez a flexión EI del elemento
            
        Returns:
            Matriz de rigidez 4x4 del elemento
        """
        if longitud <= 0:
            raise ValueError("La longitud debe ser positiva")
        if rigidez <= 0:
            raise ValueError("La rigidez debe ser positiva")
            
        a = 12 * rigidez / longitud ** 3
        b = 6 * rigidez / longitud ** 2
        c = 4 * rigidez / longitud
        d = 2 * rigidez / longitud
        
        return np.array([
            [a, b, -a, b],
            [b, c, -b, d],
            [-a, -b, a, -b],
            [b, d, -b, c]
        ], dtype=float)


class ProcesadorCargas:
    """Procesa y convierte cargas a vectores de fuerzas equivalentes."""
    
    @staticmethod
    def procesar_cargas_puntuales(cargas: List[CargaPuntual]) -> List[Tuple[int, float]]:
        """
        Convierte cargas puntuales a pares (gdl, valor).
        
        Args:
            cargas: Lista de cargas puntuales
            
        Returns:
            Lista de tuplas (gdl_index, valor)
        """
        gdl_cargados = []
        
        for carga in cargas:
            # GDL de fuerza vertical: nudo * 2 - 2 (índice base 0)
            if carga.fuerza != 0:
                gdl_fuerza = carga.nudo * 2 - 2
                gdl_cargados.append((gdl_fuerza, carga.fuerza))
            
            # GDL de momento: nudo * 2 - 1 (índice base 0)
            if carga.momento != 0:
                gdl_momento = carga.nudo * 2 - 1
                gdl_cargados.append((gdl_momento, carga.momento))
                
        return gdl_cargados

    @staticmethod
    def procesar_cargas_distribuidas(cargas: List[CargaDistribuida], 
                                   longitudes: np.ndarray) -> List[FuerzaEquivalente]:
        """
        Convierte cargas distribuidas a fuerzas nodales equivalentes.
        
        Args:
            cargas: Lista de cargas distribuidas
            longitudes: Array con longitudes de elementos
            
        Returns:
            Lista de fuerzas equivalentes
        """
        fuerzas_equivalentes = []
        
        for carga in cargas:
            elemento_idx = carga.elemento - 1  # Convertir a índice base 0
            
            if elemento_idx >= len(longitudes):
                raise ValueError(f"Elemento {carga.elemento} no existe")
                
            l = longitudes[elemento_idx]
            q1, q2 = carga.q1, carga.q2
            
            # Fórmulas para carga trapezoidal
            fuerza_izq = l * (7 * q1 + 3 * q2) / 20
            fuerza_der = l * (3 * q1 + 7 * q2) / 20
            momento_izq = l**2 * (3 * q1 + 2 * q2) / 60
            momento_der = -l**2 * (2 * q1 + 3 * q2) / 60
            
            fuerzas_equivalentes.append(FuerzaEquivalente(
                elemento=carga.elemento,
                fuerza_izq=fuerza_izq,
                momento_izq=momento_izq,
                fuerza_der=fuerza_der,
                momento_der=momento_der
            ))
            
        return fuerzas_equivalentes


class ModeloEstructural:
    """Modelo principal que encapsula toda la lógica de análisis estructural."""
    
    def __init__(self, propiedades: PropiedadesEstructura) -> None:
        self.propiedades = propiedades
        self.gdl_elementos: np.ndarray
        self.gdl_cargados: np.ndarray
        self.gdl_restringidos: np.ndarray
        self.fuerzas_equivalentes: List[FuerzaEquivalente]
        self.matriz_rigidez: np.ndarray
        self.vector_fuerzas: np.ndarray
        self.desplazamientos: np.ndarray
        self.reacciones: np.ndarray
        self._inicializar_modelo()
    
    def _inicializar_modelo(self) -> None:
        """Inicializa todos los componentes del modelo."""
        self.gdl_elementos = self._calcular_gdl_elementos()
        self.gdl_cargados = self._procesar_cargas_puntuales()
        self.gdl_restringidos = self._procesar_apoyos()
        self.fuerzas_equivalentes = self._procesar_cargas_distribuidas()
        self.matriz_rigidez = self._ensamblar_matriz_rigidez()
        self.vector_fuerzas = self._construir_vector_fuerzas()
        self._agregar_fuerzas_distribuidas()
        self.desplazamientos = self._calcular_desplazamientos()
        self.reacciones = self._calcular_reacciones()
    
    def _calcular_gdl_elementos(self) -> np.ndarray:
        """Calcula la tabla de conectividad de grados de libertad."""
        n_elem = self.propiedades.numero_elementos
        gdl = np.zeros([n_elem, 4], dtype=int)
        
        for i in range(n_elem):
            base = i + 1
            gdl[i, 0] = base * 2 - 1  # GDL vertical nudo izquierdo
            gdl[i, 1] = base * 2      # GDL momento nudo izquierdo
            gdl[i, 2] = base * 2 + 1  # GDL vertical nudo derecho
            gdl[i, 3] = base * 2 + 2  # GDL momento nudo derecho
            
        return gdl
    
    def _procesar_cargas_puntuales(self) -> np.ndarray:
        """Procesa las cargas puntuales y retorna array de [gdl, valor]."""
        gdl_valores = ProcesadorCargas.procesar_cargas_puntuales(
            self.propiedades.cargas_puntuales
        )
        return np.array(gdl_valores, dtype=float) if gdl_valores else np.empty((0, 2), dtype=float)
    
    def _procesar_apoyos(self) -> np.ndarray:
        """Procesa los apoyos y retorna array de GDL restringidos."""
        gdl_restr = []
        
        for apoyo in self.propiedades.apoyos:
            if apoyo.restriccion_vertical:
                gdl_restr.append(apoyo.nudo * 2 - 2)  # GDL vertical (base 0)
            if apoyo.restriccion_momento:
                gdl_restr.append(apoyo.nudo * 2 - 1)  # GDL momento (base 0)
                
        return np.array(gdl_restr, dtype=int)
    
    def _procesar_cargas_distribuidas(self) -> List[FuerzaEquivalente]:
        """Procesa las cargas distribuidas."""
        return ProcesadorCargas.procesar_cargas_distribuidas(
            self.propiedades.cargas_distribuidas,
            self.propiedades.longitudes
        )
    
    def _ensamblar_matriz_rigidez(self) -> np.ndarray:
        """Ensambla la matriz de rigidez global."""
        n_gdl = self.propiedades.numero_gdl
        matriz_global = np.zeros([n_gdl, n_gdl], dtype=float)
        
        for i in range(self.propiedades.numero_elementos):
            # Calcular matriz de rigidez del elemento
            longitud = self.propiedades.longitudes[i]
            rigidez = self.propiedades.rigideces[i]
            matriz_elem = CalculadoraMatricesElemento.matriz_rigidez_viga(longitud, rigidez)
            
            # Obtener GDL del elemento
            gdl_elem = self.gdl_elementos[i]
            
            # Ensamblar en la matriz global
            for j in range(4):
                for k in range(4):
                    fila = gdl_elem[j] - 1  # Convertir a índice base 0
                    columna = gdl_elem[k] - 1
                    matriz_global[fila, columna] += matriz_elem[j, k]
                    
        return matriz_global
    
    def _construir_vector_fuerzas(self) -> np.ndarray:
        """Construye el vector global de fuerzas."""
        vector = np.zeros(self.propiedades.numero_gdl, dtype=float)
        
        for gdl_idx, valor in self.gdl_cargados:
            gdl_int = int(gdl_idx)
            if 0 <= gdl_int < len(vector):
                vector[gdl_int] += valor
                
        return vector
    
    def _agregar_fuerzas_distribuidas(self) -> None:
        """Agrega las fuerzas equivalentes al vector de fuerzas."""
        for fuerza_eq in self.fuerzas_equivalentes:
            elemento_idx = fuerza_eq.elemento - 1
            gdl_elem = self.gdl_elementos[elemento_idx]
            
            # Agregar fuerzas a los GDL correspondientes (convertir a base 0)
            self.vector_fuerzas[gdl_elem[0] - 1] += fuerza_eq.fuerza_izq
            self.vector_fuerzas[gdl_elem[1] - 1] += fuerza_eq.momento_izq
            self.vector_fuerzas[gdl_elem[2] - 1] += fuerza_eq.fuerza_der
            self.vector_fuerzas[gdl_elem[3] - 1] += fuerza_eq.momento_der
    
    def _obtener_gdl_libres(self) -> np.ndarray:
        """Obtiene los GDL libres (no restringidos)."""
        todos_gdl = set(range(self.propiedades.numero_gdl))
        gdl_restr_set = set(self.gdl_restringidos)
        gdl_libres = sorted(todos_gdl - gdl_restr_set)
        return np.array(gdl_libres, dtype=int)
    
    def _calcular_desplazamientos(self) -> np.ndarray:
        """Calcula los desplazamientos nodales."""
        gdl_libres = self._obtener_gdl_libres()
        
        # Extraer submatrices para GDL libres
        matriz_libre = self.matriz_rigidez[np.ix_(gdl_libres, gdl_libres)]
        vector_libre = self.vector_fuerzas[gdl_libres]
        
        # Resolver sistema de ecuaciones
        desplaz_libres = np.linalg.solve(matriz_libre, vector_libre)
        
        # Construir vector completo de desplazamientos
        desplazamientos = np.zeros(self.propiedades.numero_gdl, dtype=float)
        desplazamientos[gdl_libres] = desplaz_libres
        
        return desplazamientos
    
    def _calcular_reacciones(self) -> np.ndarray:
        """Calcula las reacciones en los apoyos."""
        if len(self.gdl_restringidos) == 0:
            return np.array([], dtype=float)
            
        # R = K_restringido * U - F_restringido
        matriz_restr = self.matriz_rigidez[np.ix_(self.gdl_restringidos, 
                                                 np.arange(len(self.desplazamientos)))]
        fuerzas_restr = self.vector_fuerzas[self.gdl_restringidos]
        
        reacciones = matriz_restr @ self.desplazamientos - fuerzas_restr
        return reacciones


# =====================================================
# VIEWS (Presentación de datos)
# =====================================================

class FormateadorResultados(ABC):
    """Interfaz base para formateadores de resultados."""
    
    @abstractmethod
    def formatear_matriz(self, matriz: np.ndarray, titulo: str) -> str:
        pass
    
    @abstractmethod
    def formatear_vector(self, vector: np.ndarray, titulo: str, 
                        etiquetas: Optional[List[str]] = None) -> str:
        pass


class FormateadorTabular(FormateadorResultados):
    """Formateador que presenta resultados en tablas."""
    
    def formatear_matriz(self, matriz: npt.NDArray[np.floating[Any]], titulo: str) -> str:
        """Formatea una matriz como tabla."""
        n = matriz.shape[0]
        headers = [f"GDL{j+1}" for j in range(n)]
        
        tabla = []
        for i in range(n):
            fila = [f"{i+1}"] + [f"{matriz[i, j]:.2f}" for j in range(n)]
            tabla.append(fila)
            
        return tabulate(tabla, headers=["GDL"] + headers, tablefmt="simple_grid")
    
    def formatear_vector(self, vector: npt.NDArray[np.floating[Any]], titulo: str, 
                        etiquetas: Optional[List[str]] = None) -> str:
        """Formatea un vector como tabla."""
        if etiquetas is None:
            etiquetas = [f"{i+1}" for i in range(len(vector))]
            
        tabla = [[etiquetas[i], f"{val:.6f}"] for i, val in enumerate(vector)]
        return tabulate(tabla, headers=["GDL", "Valor"], tablefmt="simple_grid")


class VistaConsola:
    """Vista principal para mostrar resultados en consola."""
    
    def __init__(self, formateador: Optional[FormateadorResultados] = None) -> None:
        self.formateador = formateador or FormateadorTabular()
    
    def _titulo(self, texto: str) -> None:
        """Imprime un título con formato."""
        print(Fore.CYAN + Style.BRIGHT + f"\n{texto}\n" + "-" * len(texto) + Style.RESET_ALL)
    
    def _seccion(self, texto: str) -> None:
        """Imprime una sección con formato."""
        print(Fore.YELLOW + f"\n{texto}" + Style.RESET_ALL)
    
    def mostrar_propiedades_estructura(self, propiedades: PropiedadesEstructura) -> None:
        """Muestra las propiedades básicas de la estructura."""
        self._titulo("Datos de entrada")
        
        # Tabla de propiedades básicas
        datos_basicos = [
            ["Longitudes [m]", ", ".join(f"{l:.2f}" for l in propiedades.longitudes)],
            ["Rigideces EI [kN*m²]", ", ".join(f"{ei:.0f}" for ei in propiedades.rigideces)],
        ]
        print(tabulate(datos_basicos, tablefmt="simple_grid"))
        
        # Cargas puntuales
        if propiedades.cargas_puntuales:
            tabla_cargas = [["Nudo", "Fuerza [kN]", "Momento [kN*m]"]]
            for carga in propiedades.cargas_puntuales:
                tabla_cargas.append([str(carga.nudo), str(carga.fuerza), str(carga.momento)])
            print(tabulate(tabla_cargas, headers="firstrow", tablefmt="simple_grid"))
        
        # Cargas distribuidas
        if propiedades.cargas_distribuidas:
            tabla_dist = [["Elemento", "q1 [kN/m]", "q2 [kN/m]"]]
            for carga in propiedades.cargas_distribuidas:
                tabla_dist.append([str(carga.elemento), str(carga.q1), str(carga.q2)])
            print(tabulate(tabla_dist, headers="firstrow", tablefmt="simple_grid"))
        
        # Apoyos
        if propiedades.apoyos:
            tabla_apoyos = [["Nudo", "Restricción V", "Restricción M"]]
            for apoyo in propiedades.apoyos:
                tabla_apoyos.append([
                    str(apoyo.nudo),
                    str(1 if apoyo.restriccion_vertical else 0),
                    str(1 if apoyo.restriccion_momento else 0)
                ])
            print(tabulate(tabla_apoyos, headers="firstrow", tablefmt="simple_grid"))
    
    def mostrar_matriz_rigidez(self, matriz: np.ndarray) -> None:
        """Muestra la matriz de rigidez global."""
        self._titulo("Matriz de rigidez global [K]")
        print(self.formateador.formatear_matriz(matriz, "Matriz de rigidez"))
    
    def mostrar_vector_fuerzas(self, vector: np.ndarray) -> None:
        """Muestra el vector de fuerzas nodales."""
        self._titulo("Vector global de fuerzas nodales [F]")
        tabla = [[i+1, f"{val:.2f}"] for i, val in enumerate(vector)]
        print(tabulate(tabla, headers=["GDL", "Fuerza [kN]"], tablefmt="simple_grid"))
    
    def mostrar_desplazamientos(self, desplazamientos: np.ndarray) -> None:
        """Muestra los desplazamientos nodales."""
        self._titulo("Desplazamientos nodales [U]")
        tabla = [[i+1, f"{val:.6f}"] for i, val in enumerate(desplazamientos)]
        print(tabulate(tabla, headers=["GDL", "Desplazamiento/Giro"], tablefmt="simple_grid"))
    
    def mostrar_reacciones(self, reacciones: np.ndarray, 
                          gdl_restringidos: np.ndarray) -> None:
        """Muestra las reacciones en los apoyos."""
        if len(reacciones) == 0:
            return
            
        self._titulo("Reacciones en apoyos")
        tabla = [[int(gdl)+1, f"{reaccion:.2f}"] 
                for gdl, reaccion in zip(gdl_restringidos, reacciones)]
        print(tabulate(tabla, headers=["GDL", "Reacción [kN]"], tablefmt="simple_grid"))
    
    def mostrar_fuerzas_equivalentes(self, fuerzas_eq: List[FuerzaEquivalente]) -> None:
        """Muestra las fuerzas equivalentes de cargas distribuidas."""
        if not fuerzas_eq:
            return
            
        self._titulo("Fuerzas equivalentes por cargas distribuidas")
        tabla = []
        for fuerza in fuerzas_eq:
            tabla.append([
                fuerza.elemento,
                f"{fuerza.fuerza_izq:.2f}",
                f"{fuerza.momento_izq:.2f}",
                f"{fuerza.fuerza_der:.2f}",
                f"{fuerza.momento_der:.2f}"
            ])
        print(tabulate(tabla, headers=["Elemento", "F_izq", "M_izq", "F_der", "M_der"], 
                      tablefmt="simple_grid"))
    
    def mostrar_resultados_completos(self, modelo: ModeloEstructural) -> None:
        """Muestra todos los resultados del análisis."""
        self.mostrar_propiedades_estructura(modelo.propiedades)
        self.mostrar_fuerzas_equivalentes(modelo.fuerzas_equivalentes)
        self._seccion("Resultados del análisis")
        self.mostrar_matriz_rigidez(modelo.matriz_rigidez)
        self.mostrar_vector_fuerzas(modelo.vector_fuerzas)
        self.mostrar_desplazamientos(modelo.desplazamientos)
        self.mostrar_reacciones(modelo.reacciones, modelo.gdl_restringidos)












# =====================================================
# CONTROLLERS (Controladores y coordinación)
# =====================================================




class ControladorAnalisis:
    """Controlador principal que coordina el análisis estructural."""
    
    def __init__(self) -> None:
        self.vista = VistaConsola()
        self.modelo: Optional[ModeloEstructural] = None
    
    def cargar_datos_ejemplo(self) -> PropiedadesEstructura:
        """Carga datos de ejemplo para pruebas."""
        # Crear cargas puntuales
        cargas_puntuales = [
            CargaPuntual(nudo=1, fuerza=-20, momento=0),
            CargaPuntual(nudo=2, fuerza=-40, momento=0)
        ]
        
        # Crear cargas distribuidas
        cargas_distribuidas = [
            CargaDistribuida(elemento=3, q1=0, q2=-30),
            CargaDistribuida(elemento=4, q1=-30, q2=-30),
            CargaDistribuida(elemento=5, q1=-30, q2=0)
        ]
        
        # Crear apoyos
        apoyos = [
            Apoyo(nudo=3, restriccion_vertical=True, restriccion_momento=False),
            Apoyo(nudo=5, restriccion_vertical=True, restriccion_momento=False),
            Apoyo(nudo=6, restriccion_vertical=True, restriccion_momento=True)
        ]
        
        return PropiedadesEstructura(
            longitudes=np.array([1, 0.8, 2, 2, 3], dtype=float),
            rigideces=np.array([6000, 6000, 9000, 9000, 4000], dtype=float),
            cargas_puntuales=cargas_puntuales,
            cargas_distribuidas=cargas_distribuidas,
            apoyos=apoyos
        )
    
    def ejecutar_analisis(self, propiedades: PropiedadesEstructura) -> bool:
        """Ejecuta el análisis completo."""
        try:
            self.modelo = ModeloEstructural(propiedades)
            return True
        except Exception as e:
            print(Fore.RED + f"Error en el análisis: {e}" + Style.RESET_ALL)
            return False
    
    def mostrar_resultado(self, opcion: str) -> None:
        """Muestra el resultado según la opción especificada."""
        if self.modelo is None:
            print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL)
            return
        
        opciones: Dict[str, Callable[[], None]] = {
            "matriz": lambda: self.vista.mostrar_matriz_rigidez(self.modelo.matriz_rigidez) if self.modelo else print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL),
            "fuerzas": lambda: self.vista.mostrar_vector_fuerzas(self.modelo.vector_fuerzas) if self.modelo else print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL),
            "datos": lambda: self.vista.mostrar_propiedades_estructura(self.modelo.propiedades) if self.modelo else print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL),
            "desplazamientos": lambda: self.vista.mostrar_desplazamientos(self.modelo.desplazamientos) if self.modelo else print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL),
            "reacciones": lambda: self.vista.mostrar_reacciones(self.modelo.reacciones, 
                                                               self.modelo.gdl_restringidos) if self.modelo else print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL),
            "equivalentes": lambda: self.vista.mostrar_fuerzas_equivalentes(self.modelo.fuerzas_equivalentes) if self.modelo else print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL),
            "todo": lambda: self.vista.mostrar_resultados_completos(self.modelo) if self.modelo else print(Fore.RED + "No hay modelo cargado" + Style.RESET_ALL)
        }
        
        accion = opciones.get(opcion.lower(), opciones["todo"])
        accion()


# =====================================================
# FUNCIÓN PRINCIPAL Y PUNTO DE ENTRADA
# =====================================================

def main() -> None:
    """Función principal del programa."""
    try:
        # Obtener opción de línea de comandos
        opcion = sys.argv[1].lower() if len(sys.argv) > 1 else "todo"
        
        # Crear controlador
        controlador = ControladorAnalisis()
        
        # Cargar datos y ejecutar análisis
        propiedades = controlador.cargar_datos_ejemplo()
        
        if controlador.ejecutar_analisis(propiedades):
            controlador.mostrar_resultado(opcion)
        
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nProceso interrumpido por el usuario" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error inesperado: {e}" + Style.RESET_ALL)
        sys.exit(1)


if __name__ == "__main__":
    main()
    