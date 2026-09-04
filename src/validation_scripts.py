"""Validación del dataset plano que consume el dashboard (fase 06).

Comprueba que `reports/dashboard/coffee_dashboard_data.csv` está en condiciones
de importarse a Power BI o Tableau antes de abrir la herramienta, que es donde
los problemas salen tarde y sin explicación clara.

Se ejecuta solo, desde la raíz del repositorio:

    python src/validation_scripts.py

Termina con código 0 si todo está bien y 1 si alguna validación falla, así que
también sirve dentro de un script de verificación más grande.

Los valores esperados no están escritos acá: el número de lotes y el tamaño de
cada grupo se leen de `models/clustering_metadata.json`, los nombres de los
grupos y el umbral de especialidad de `config/config.yaml`, y las columnas de
catación de `config -> clustering.features`. Si mañana se reentrena el modelo con
otra configuración, este script sigue validando lo correcto sin tocarlo.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# El script se ejecuta como archivo, no como módulo, así que `src` todavía no
# está en el path: se agrega la raíz del repositorio (un nivel arriba de src/).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve_path  # noqa: E402


def encabezado(titulo):
    """Imprime un separador de sección, para que el reporte se pueda leer."""
    print()
    print(titulo)
    print("-" * len(titulo))


def marca(ok):
    """Devuelve el símbolo de resultado de una comprobación."""
    return "[ OK ]" if ok else "[FALLA]"


def validar_archivo(ruta, n_lotes_esperado):
    """1. El archivo existe, se puede leer y trae la cantidad de registros esperada."""
    encabezado("1. Archivo y cantidad de registros")

    if not ruta.exists():
        print(f"{marca(False)} el archivo no existe: {ruta}")
        print("        se genera con la fase 06 (notebook 06 o pipeline.interpret_clusters)")
        return False, None

    print(f"{marca(True)} el archivo existe: {ruta.relative_to(REPO_ROOT)}")
    print(f"        tamaño: {ruta.stat().st_size / 1024:.1f} KB")

    # Power BI lee UTF-8 sin problemas, pero un BOM le agrega caracteres raros al
    # nombre de la primera columna, así que se revisa antes de nada.
    primeros_bytes = ruta.read_bytes()[:3]
    tiene_bom = primeros_bytes == b"\xef\xbb\xbf"
    print(f"{marca(not tiene_bom)} codificación sin BOM"
          + ("" if not tiene_bom else "  <- el BOM ensucia el nombre de la primera columna"))

    df = pd.read_csv(ruta)
    ok_filas = len(df) == n_lotes_esperado
    print(f"{marca(ok_filas)} registros: {len(df)} (esperados {n_lotes_esperado})")
    print(f"        columnas: {len(df.columns)}")

    return (not tiene_bom) and ok_filas, df


def validar_columnas(df, bloques):
    """2. Están todas las columnas esperadas, agrupadas por bloque."""
    encabezado("2. Columnas esperadas")

    todo_ok = True
    for nombre_bloque, columnas in bloques.items():
        faltan = [c for c in columnas if c not in df.columns]
        ok = not faltan
        todo_ok = todo_ok and ok
        print(f"{marca(ok)} {nombre_bloque}: {len(columnas) - len(faltan)}/{len(columnas)}")
        print(f"        {', '.join(columnas)}")
        if faltan:
            print(f"        FALTAN: {', '.join(faltan)}")

    # Nombres con espacios sobrantes o repetidos: rompen las relaciones en Power BI.
    con_espacios = [c for c in df.columns if c != c.strip()]
    repetidas = [c for c in df.columns if list(df.columns).count(c) > 1]
    print(f"{marca(not con_espacios)} ningún nombre de columna con espacios sobrantes")
    print(f"{marca(not repetidas)} ningún nombre de columna repetido")

    return todo_ok and not con_espacios and not repetidas


def validar_nulos(df, criticas, esperadas_con_nulos):
    """3. No hay nulos en las columnas críticas.

    Los nulos de altitud, humedad y año de cosecha sí son legítimos: la fase 03
    decidió no imputarlos porque no faltan al azar. Se reportan aparte para que
    nadie los trate como un error ni los rellene antes de importar.
    """
    encabezado("3. Valores nulos")

    todo_ok = True
    for columna in criticas:
        if columna not in df.columns:
            continue
        n_nulos = int(df[columna].isna().sum())
        ok = n_nulos == 0
        todo_ok = todo_ok and ok
        print(f"{marca(ok)} {columna}: {n_nulos} nulos")

    print()
    print("        Nulos esperados por diseño (la fase 03 decidió no imputarlos):")
    for columna in esperadas_con_nulos:
        if columna not in df.columns:
            continue
        n_nulos = int(df[columna].isna().sum())
        print(f"        - {columna}: {n_nulos} nulos "
              f"({n_nulos / len(df) * 100:.1f} %), se perfila solo con los que reportan")

    return todo_ok


def validar_grupos(df, tam_esperado, nombres_esperados):
    """4. La distribución de los grupos es la del modelo, y cada grupo trae su nombre."""
    encabezado("4. Distribución de los grupos")

    conteos = df["grupo"].value_counts().sort_index()
    todo_ok = True

    for grupo, n_esperado in sorted(tam_esperado.items()):
        n_real = int(conteos.get(grupo, 0))
        ok = n_real == n_esperado
        todo_ok = todo_ok and ok
        print(f"{marca(ok)} grupo {grupo}: {n_real} lotes (esperados {n_esperado})")

    # Que no aparezcan grupos fuera de los del modelo.
    inesperados = sorted(set(conteos.index) - set(tam_esperado))
    ok_valores = not inesperados
    todo_ok = todo_ok and ok_valores
    print(f"{marca(ok_valores)} solo los grupos {sorted(tam_esperado)}"
          + ("" if ok_valores else f"  <- aparecieron además: {inesperados}"))

    # Cada grupo tiene que traer el nombre que dice el config, y uno solo.
    print()
    for grupo in sorted(tam_esperado):
        nombres = df.loc[df["grupo"] == grupo, "grupo_nombre"].unique()
        esperado = nombres_esperados.get(grupo)
        ok = len(nombres) == 1 and nombres[0] == esperado
        todo_ok = todo_ok and ok
        print(f"{marca(ok)} grupo {grupo} -> \"{nombres[0] if len(nombres) == 1 else list(nombres)}\"")

    return todo_ok


def validar_especialidad(df, columna_puntaje, umbral):
    """5. `especialidad` es binaria y coincide con el umbral de la industria."""
    encabezado("5. Columna `especialidad`")

    valores = sorted(df["especialidad"].dropna().unique().tolist())
    ok_binaria = set(valores).issubset({0, 1})
    print(f"{marca(ok_binaria)} valores presentes: {valores} (se esperan 0 y 1)")

    # Se recalcula desde el puntaje: si no coincide, la columna quedó desfasada
    # respecto a los datos y el tablero mostraría dos verdades distintas.
    recalculada = (df[columna_puntaje] >= umbral).astype(int)
    n_discrepancias = int((recalculada != df["especialidad"]).sum())
    ok_coherente = n_discrepancias == 0
    print(f"{marca(ok_coherente)} coincide con {columna_puntaje} >= {umbral}: "
          f"{n_discrepancias} discrepancias")

    n_si = int(df["especialidad"].sum())
    print(f"        lotes de especialidad: {n_si} de {len(df)} "
          f"({n_si / len(df) * 100:.1f} %)")

    return ok_binaria and ok_coherente


def validar_tipos(df, numericas):
    """Extra: las columnas numéricas se leyeron como números.

    Si alguna quedó como texto, casi siempre es porque el CSV se guardó con coma
    decimal; Power BI la importaría como texto y no dejaría hacer promedios.
    """
    encabezado("6. Tipos de dato (lectura de Power BI)")

    todo_ok = True
    for columna in numericas:
        if columna not in df.columns:
            continue
        ok = pd.api.types.is_numeric_dtype(df[columna])
        todo_ok = todo_ok and ok
        print(f"{marca(ok)} {columna}: {df[columna].dtype}"
              + ("" if ok else "  <- se leyó como texto, revisar el separador decimal"))
    return todo_ok


def main():
    config = load_config()

    # Todo lo esperado sale del config y de los metadatos del modelo.
    ruta = resolve_path(config["paths"]["reports"]["dashboard_dir"]) / "coffee_dashboard_data.csv"
    metadata = json.loads(
        (resolve_path(config["paths"]["models_dir"]) / "clustering_metadata.json")
        .read_text(encoding="utf-8")
    )
    tam_esperado = {int(g): n for g, n in metadata["tam_grupos"].items()}
    nombres_esperados = config["interpretation"]["group_names"]
    columna_puntaje = config["validation"]["reference_column"]
    umbral = config["validation"]["specialty_threshold"]
    sca = config["clustering"]["features"]

    bloques = {
        # Los 7 de config -> clustering.features: incluye `Cupper.Points`, que
        # también entró al agrupamiento.
        "Atributos SCA": sca,
        "Columnas de clustering": ["grupo", "grupo_nombre", "especialidad"],
        "Contexto de origen y manejo": [
            "Country.of.Origin", "Variety", "Processing.Method", "altitude_mean_meters",
        ],
        "Puntajes": [columna_puntaje, "rango_puntaje"],
    }
    criticas = ["grupo", "grupo_nombre", "especialidad", columna_puntaje] + sca
    esperadas_con_nulos = ["altitude_mean_meters", "Moisture", "Harvest.Year"]
    numericas = sca + [columna_puntaje, "grupo", "especialidad", "altitude_mean_meters"]

    print("=" * 62)
    print("VALIDACIÓN DEL DATASET DEL DASHBOARD")
    print("=" * 62)
    print(f"Archivo esperado : {ruta.relative_to(REPO_ROOT)}")
    print(f"Modelo de refer. : {metadata['representacion']} + {metadata['algoritmo']}, "
          f"k = {metadata['k']}, semilla {metadata['random_seed']}")

    ok_archivo, df = validar_archivo(ruta, metadata["n_lotes"])
    if df is None:
        print()
        print("No se puede seguir validando sin el archivo.")
        return 1

    resultados = {
        "archivo y registros": ok_archivo,
        "columnas": validar_columnas(df, bloques),
        "nulos": validar_nulos(df, criticas, esperadas_con_nulos),
        "grupos": validar_grupos(df, tam_esperado, nombres_esperados),
        "especialidad": validar_especialidad(df, columna_puntaje, umbral),
        "tipos de dato": validar_tipos(df, numericas),
    }

    encabezado("Resumen")
    for nombre, ok in resultados.items():
        print(f"{marca(ok)} {nombre}")

    print()
    print("=" * 62)
    if all(resultados.values()):
        print("✓ CSV listo para Power BI")
        print("=" * 62)
        return 0
    fallaron = [n for n, ok in resultados.items() if not ok]
    print(f"✗ El CSV NO está listo. Revisar: {', '.join(fallaron)}")
    print("  Volver a generarlo con: python -c \"from src.config import load_config; "
          "from src.pipeline import interpret_clusters; interpret_clusters(load_config())\"")
    print("=" * 62)
    return 1


if __name__ == "__main__":
    sys.exit(main())
