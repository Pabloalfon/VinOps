import os

def extraer_contexto_latex(ruta_base='.', archivo_salida='contexto_plantilla.txt'):
    # Extensiones de los archivos que nos interesa analizar
    extensiones = ('.tex', '.cls', '.sty')
    
    with open(archivo_salida, 'w', encoding='utf-8') as f_out:
        # 1. Guardar la estructura de carpetas
        f_out.write("=== ESTRUCTURA DE ARCHIVOS ===\n")
        for root, dirs, files in os.walk(ruta_base):
            for file in files:
                if file.endswith(extensiones) or file.endswith('.bib'):
                    ruta_relativa = os.path.relpath(os.path.join(root, file), ruta_base)
                    f_out.write(f"{ruta_relativa}\n")
        
        # 2. Guardar el contenido del código
        f_out.write("\n=== CONTENIDO DE LOS ARCHIVOS ===\n")
        for root, dirs, files in os.walk(ruta_base):
            for file in files:
                if file.endswith(extensiones):
                    ruta_completa = os.path.join(root, file)
                    f_out.write(f"\n--- INICIO DE {file} ---\n")
                    try:
                        with open(ruta_completa, 'r', encoding='utf-8') as f_in:
                            f_out.write(f_in.read())
                    except Exception as e:
                        f_out.write(f"[Error leyendo archivo: {e}]\n")
                    f_out.write(f"--- FIN DE {file} ---\n")
                    
    print(f"¡Listo! Se ha generado el archivo: {archivo_salida}")
    print("Por favor, sube este archivo al chat para analizar la plantilla.")

if __name__ == "__main__":
    extraer_contexto_latex()