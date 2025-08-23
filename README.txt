----------------------------------------------------------
DOCUMENTACIÓN
----------------------------------------------------------
Este programa realiza el análisis matricial de una viga en 2D, ensamblando la matriz de rigidez global,
procesando cargas puntuales y distribuidas (rectangulares, triangulares, trapezoidales), y mostrando
los resultados de fuerzas nodales y matrices. El usuario puede elegir qué resultados mostrar desde la CLI.

Estructura principal:
- Definición de datos de entrada (longitudes, rigideces, cargas, apoyos)
- Funciones auxiliares para ensamblaje y procesamiento
- Interfaz CLI para mostrar resultados

Uso:
    python src/main.py [opción]
Opciones:
    matriz      - Muestra solo la matriz de rigidez global
    fuerzas     - Muestra solo el vector global de fuerzas nodales
    todo        - Muestra todos los resultados (por defecto)

Licencia: MIT