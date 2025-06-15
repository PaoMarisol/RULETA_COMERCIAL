# Paso A: Importar las librerías necesarias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings

warnings.filterwarnings('ignore')

print("Paso 1: Cargando los datos...")

# Paso B: Cargar los datos desde el archivo Excel
try:
    df = pd.read_excel('data_arca.xlsx..xlsx' \
    '')
except FileNotFoundError:
    print("Error: El archivo 'data_arca.xlsx..xlsx' no se encontró. Asegúrate de que esté en la misma carpeta.")
    exit()

print("Datos cargados correctamente. Filas iniciales:", len(df))

# -----------------------------------------------------------------------------
# --- CORRECCIÓN PARA EL ERROR DE MEMORIA: INICIO DEL BLOQUE DE CORRECCIÓN ---
# -----------------------------------------------------------------------------

print("\nPaso 2: Simplificando y agrupando productos en categorías...")

def agrupar_producto(nombre_producto):
    """
    Esta función toma un nombre de producto detallado y lo devuelve
    en una categoría general. Es crucial para reducir la dimensionalidad.
    """
    # Convertimos a minúsculas para que la búsqueda no distinga mayúsculas/minúsculas
    nombre_producto = str(nombre_producto).lower() 
    
    # Define tus reglas de categorización aquí. Añade más según tus productos.
    if 'coca-cola sin azucar' in nombre_producto or 'coca cola zero' in nombre_producto or 'coca cola light' in nombre_producto:
        return 'Coca-Cola Sin Azucar'
    if 'coca-cola original' in nombre_producto or 'coca cola clasica' in nombre_producto:
        return 'Coca-Cola Original'
    if 'fanta' in nombre_producto:
        return 'Fanta'
    if 'sprite' in nombre_producto:
        return 'Sprite'
    if 'ciel' in nombre_producto:
        return 'Agua Ciel'
    if 'del valle' in nombre_producto:
        return 'Jugo Del Valle'
    if 'powerade' in nombre_producto:
        return 'Powerade'
    if 'mundet' in nombre_producto:
        return 'Sidral Mundet'
    # Si no coincide con ninguna regla, lo ponemos en una categoría genérica.
    else:
        return 'Otros Productos'

# Aplicamos la función para crear una nueva columna con las categorías de productos
df['producto_categoria'] = df['Producto'].apply(agrupar_producto)

print("\nCategorías de productos creadas. Las más comunes son:")
print(df['producto_categoria'].value_counts().head(10))

# -----------------------------------------------------------------------------
# --- CORRECCIÓN PARA EL ERROR DE MEMORIA: FIN DEL BLOQUE DE CORRECCIÓN ---
# -----------------------------------------------------------------------------


# Paso C: Preparar los datos para el análisis de canasta
print("\nPaso 3: Agrupando productos por transacción (cliente-mes)...")

# --- CAMBIO IMPORTANTE: Ahora usamos la nueva columna de categorías ---
# Esto crea las "canastas" de compra usando las categorías generales, no los productos detallados.
basket = df.groupby(['ID Cliente', 'Año', 'Mes'])['producto_categoria'].apply(list).values.tolist()

print(f"Se han identificado {len(basket)} transacciones (canastas).")
# Filtramos las canastas vacías que podrían haberse creado
basket = [transaccion for transaccion in basket if transaccion]
print(f"Número de transacciones válidas: {len(basket)}")
print("Ejemplo de una transacción con categorías:", basket[0])


# Paso D: Transformar los datos al formato requerido por el algoritmo Apriori
print("\nPaso 4: Codificando las transacciones...")
# Esta parte ahora funcionará porque el número de columnas (categorías) es mucho menor.
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("Ejemplo de datos codificados:")
print(df_encoded.head())

# Paso E: Aplicar el algoritmo Apriori para encontrar "itemsets" frecuentes
print("\nPaso 5: Buscando conjuntos de productos frecuentes (Apriori)...")
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

print("Conjuntos de productos frecuentes encontrados:")
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

# Paso F: Generar las reglas de asociación a partir de los itemsets frecuentes
print("\nPaso 6: Generando reglas de asociación ('Si compra A, entonces compra B')...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
rules_sorted = rules.sort_values(by=['lift', 'confidence'], ascending=False)

print("\n--- ¡PATRONES DE COMPRA ENCONTRADOS! ---")
print("Top 10 reglas de asociación más fuertes:")
print(rules_sorted.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Paso G: Interpretación y Oportunidades de Crecimiento
print("\n--- INTERPRETACIÓN Y OPORTUNIDADES ---")

if not rules_sorted.empty:
    ejemplo_regla = rules_sorted.iloc[0]
    antecedente = list(ejemplo_regla['antecedents'])[0]
    consecuente = list(ejemplo_regla['consequents'])[0]
    confianza = round(ejemplo_regla['confidence'] * 100, 2)
    elevacion = round(ejemplo_regla['lift'], 2)

    print(f"\nEjemplo con la regla más fuerte encontrada:")
    print(f"  Regla: Si un cliente compra '{antecedente}', también tiende a comprar '{consecuente}'.")
    print(f"  Confianza: {confianza}%. De todas las veces que alguien compró '{antecedente}', el {confianza}% de las veces también compró '{consecuente}'.")
    print(f"  Lift (Elevación): {elevacion}. La compra conjunta es {elevacion} veces más probable de lo que se esperaría por pura casualidad.")

    print("\nOportunidades de Negocio basadas en esta regla:")
    print(f"1. Venta Cruzada (Cross-Selling): Crear un 'combo' o promoción que incluya '{antecedente}' y '{consecuente}'.")
    print(f"2. Diseño de Tienda/Web: Colocar '{consecuente}' cerca de '{antecedente}' o mostrarlo como 'producto recomendado'.")
    print(f"3. Marketing Dirigido: Enviar un cupón para '{consecuente}' a los clientes que solo han comprado '{antecedente}'.")
else:
    print("\nNo se encontraron reglas con los umbrales actuales. Intenta bajar el valor de `min_support` en el Paso E (ej: a 0.005) o el `min_threshold` en el Paso F.")