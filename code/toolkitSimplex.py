import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import argparse
import json
import sys

def CalibEvDone(path1,path2): # Lee los archivos de calibración y evaluación
    calibration = []
    with open(path1, "r") as f:
        for i in f:
            l = i.strip()
            l = l.split()
            l = [float(x) for x in l]
            l = l[:2]
            calibration.append(l)

    evaluation = []
    with open(path2, "r") as f:
        for i in f:
            l = i.strip()
            l = l.split()
            l = [float(x) for x in l]
            l = l[:2]
            evaluation.append(l)

    return calibration, evaluation

def given_grouped(path1,path2):
    def read_grouped(path):
        data = []
        with open(path, "r") as f:
            for line_number, i in enumerate(f, start=1):
                l = i.strip()
                if not l:
                    continue
                l = [float(x) for x in l.split()]
                if len(l) < 3:
                    raise ValueError(f"El archivo agrupado {path} debe tener al menos 3 columnas en la linea {line_number}.")
                m = l[0]
                if m <= 0:
                    raise ValueError(f"El tamaño de grupo debe ser positivo en {path}, linea {line_number}.")
                if m.is_integer():
                    m = int(m)
                data.append((m, l[1], l[2]))

        if not data:
            raise ValueError(f"El archivo agrupado {path} no contiene datos.")

        return data

    return read_grouped(path1), read_grouped(path2)

def agrupar(K, calibration,f,t): # Agrupa los datos en bloques de tamaño K, calcula la media de cada bloque y guarda el resultado en un archivo. Si t es True, el resto se guarda como un bloque separado; si es False, se combina con el último bloque.
    output_file = f

    def write_grouped(vals):
        with open(output_file, "w") as out:
            for m, s1, s2 in vals:
                out.write(f"{m} {s1} {s2}\n")

    calibration.sort(key=lambda x: x[1])

    s1 = 0.0
    s2 = 0.0
    count = 0

    if K <= 0:
        raise ValueError("K debe ser un entero positivo.")
    if K > len(calibration):
        raise ValueError("K no puede ser mayor que el número de datos en la calibración.")
    
    M = len(calibration) // K
    r = len(calibration) % K

    print(f"Agrupado en bloques de tamaño M={M} con resto r={r}")

    vals = []

    limite = len(calibration)
    if r != 0:
        if t:
            limite -= r
        else:
            limite -= (M + r)

    for a, b in calibration[:limite]:
        s1 += b
        s2 += a
        count += 1

        if count % M == 0:
            vals.append((M,s1/M, s2/M))
            s1 = 0.0
            s2 = 0.0
    
    if r != 0:

        if t: 
            s1 = 0.0
            s2 = 0.0
            for a,b in calibration[-r:]:
                s1 += b
                s2 += a

            vals.append((r, s1/r, s2/r))

            write_grouped(vals)
        
        else:
            s1 = 0.0
            s2 = 0.0
            c = M + r
            for a,b in calibration[-c:]:
                s1 += b
                s2 += a
            vals.append((c, s1/c, s2/c))

            write_grouped(vals)
    
    else:
        write_grouped(vals)
        
    return vals

def train_model_simplex(x_fit, y_fit, m_fit): # Ajusta el modelo y = a * x^b a los datos utilizando el método Nelder-Mead para minimizar la suma de errores absolutos ponderados (WSSR).

    def model(p, x, eps=1e-12):
        a, b = p
        x = np.asarray(x, dtype=float)
        x_safe = np.maximum(x, eps)
        return a * np.power(x_safe, b)

    def objective(p):
        a, b = p

        if a < 0 or b < 0:
            return 1e12

        pred = model(p, x_fit)
        return np.sum(m_fit * np.abs(pred - y_fit))

    p0 = np.array([1.0, 1.0], dtype=float)

    res = minimize(
        objective,
        p0,
        method="Nelder-Mead",
        options={
            "maxiter": 5000,
            "xatol": 1e-8,
            "fatol": 1e-8,
            "disp": False
        }
    )

    a, b = res.x
    weighted_mae = res.fun / np.sum(m_fit)

    return a, b, weighted_mae

def get_mxy(data):
    m = [i[0] for i in data]
    x = [i[1] for i in data]
    y = [i[2] for i in data]

    m_fit = np.asarray(m).astype(float)
    x_fit = np.asarray(x).astype(float)
    y_fit = np.asarray(y).astype(float)

    return m_fit, x_fit, y_fit

def do_trim(lim,x,y,m,a,b): # Realiza un triming de outliers basado en los residuos del modelo ajustado. Se calculan los residuos, se determina un rango aceptable basado en la media y desviación estándar de los residuos, y se eliminan los puntos que caen fuera de ese rango.

    x_safe = np.maximum(x, 1e-12)
    y_pred = a * np.power(x_safe, b)

    residuals_vals = np.sqrt(m) * np.abs(y - y_pred)

    mean_res = np.mean(residuals_vals)
    std_res = np.std(residuals_vals)

    lower = mean_res - lim * std_res
    upper = mean_res + lim * std_res

    outlier_idx = np.where((residuals_vals < lower) | (residuals_vals > upper))[0]

    mask = np.ones_like(x, dtype=bool)
    mask[outlier_idx] = False

    m_trim = m[mask]
    x_trim = x[mask]
    y_trim = y[mask]

    return m_trim, x_trim, y_trim

def evaluate(K,a,b,evaluation,outfile): # Obtiene las predicciones del modelo para los datos de evaluación, calcula las métricas de error y guarda las predicciones en un archivo.
    res = []
    with open(outfile, "w") as out:
        for m, s1, s2 in evaluation:
            input_est = float(s1)
            s1 = max(input_est, 1e-12)
            pred = float(a * s1**b)
            res.append((m, input_est, pred, s2))
            out.write(f"{m} {s1} {pred} {s2}\n")
    
    sec = 0
    see = 0
    sem = 0
    sde = 0
    n = 0
    for i in res:
        mi = i[0]
        eei = i[1]
        epi = i[2]
        emi = i[3]

        n += mi
        see += (mi * eei)
        sec += (mi * epi)
        sem += (mi * emi)
        sde += (mi * abs(emi - epi))
    
    eee = see / n
    eca = sec / n
    eem = sem / n
    ede = sde / n

    Ee = 100*eee
    Ecal = 100*eca
    E = 100*eem
    DE = 100*abs(eca-eem)
    dE = 100*ede
    rDE = DE/eem
    rdE = dE/eem

    return Ecal, Ee, E, DE, dE, rDE, rdE

def guardar_metadata(metadata_path, metadata):
    with open(metadata_path, "w") as out:
        json.dump(metadata, out, indent=2)

def simple_plot(x,y,a,b,out_file): # Genera un gráfico de dispersión de los datos y una curva ajustada basada en el modelo y = a * x^b. El gráfico se guarda como un archivo PDF.

    titulo = None

    xlabel = "Estimated Error"
    ylabel = "Empirical Error"

    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0

    x_curve = np.linspace(xmin, xmax, 1000)
    x_safe = np.maximum(x_curve, 1e-12)
    y_curve = a * np.power(x_safe, b)

    color_puntos = "blue"
    color_linea = "red"

    tamano_puntos = 40          
    grosor_borde_puntos = 0.8   
    puntos_rellenos = True

    anchura_linea = 2           

    tamano_fig_x = 6           
    tamano_fig_y = 6

    tamano_fuente_ejes = 12
    tamano_fuente_ticks = 11
    tamano_fuente_titulo = 14

    grosor_ejes = 1.2

    mostrar_leyenda = False
    label_puntos = "data"
    label_linea = "fit"

    guardar_pdf = True

    fig, ax = plt.subplots(figsize=(tamano_fig_x, tamano_fig_y))

    ax.scatter(
        x,
        y,
        s=tamano_puntos,
        color=color_puntos,
        edgecolors="black",
        linewidths=grosor_borde_puntos,
        facecolors=color_puntos if puntos_rellenos else "none",
        label=label_puntos
    )

    ax.plot(
        x_curve,
        y_curve,
        color=color_linea,
        linewidth=anchura_linea,
        label=label_linea
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


    ax.set_xlabel(xlabel, fontsize=tamano_fuente_ejes)
    ax.set_ylabel(ylabel, fontsize=tamano_fuente_ejes)

    ax.tick_params(labelsize=tamano_fuente_ticks)

    for spine in ax.spines.values():
        spine.set_linewidth(grosor_ejes)


    if titulo is not None:
        ax.set_title(titulo, fontsize=tamano_fuente_titulo)

    if mostrar_leyenda:
        ax.legend()

    plt.tight_layout()
    plt.show(block=False)

    if guardar_pdf:
        fig.savefig(out_file, format="pdf", bbox_inches="tight")

def guardar_script_plot(ruta_archivo, a, b, nombre_script="plot_script.py", ruta_pdf="./plot.pdf"): # Genera un script de Python que, al ejecutarse, leerá los datos de un archivo, ajustará el modelo y = a * x^b a esos datos, y generará un gráfico de dispersión con la curva ajustada. El script se guarda con el nombre especificado.

    contenido = f'''import numpy as np
import matplotlib.pyplot as plt

archivo = {ruta_archivo!r}

x, y = [], []

with open(archivo, 'r') as f:
    for line in f:
        parts = line.strip().split()
        x.append(float(parts[1]))
        y.append(float(parts[2]))

x_fit = np.asarray(x).astype(float)
y_fit = np.asarray(y).astype(float)

a = {a}
b = {b}

def simple_plot(x, y, a, b):

    titulo = None

    xlabel = "x"
    ylabel = "y"

    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0

    x_curve = np.linspace(xmin, xmax, 1000)
    x_safe = np.maximum(x_curve, 1e-12)
    y_curve = a * np.power(x_safe, b)

    color_puntos = "blue"
    color_linea = "red"

    tamano_puntos = 40
    grosor_borde_puntos = 0.8
    puntos_rellenos = True

    anchura_linea = 2

    tamano_fig_x = 6
    tamano_fig_y = 6

    tamano_fuente_ejes = 12
    tamano_fuente_ticks = 11
    tamano_fuente_titulo = 14

    grosor_ejes = 1.2

    mostrar_leyenda = False
    label_puntos = "data"
    label_linea = "fit"

    guardar_pdf = True
    out_file = {ruta_pdf!r}

    fig, ax = plt.subplots(figsize=(tamano_fig_x, tamano_fig_y))

    ax.scatter(
        x,
        y,
        s=tamano_puntos,
        color=color_puntos,
        edgecolors="black",
        linewidths=grosor_borde_puntos,
        facecolors=color_puntos if puntos_rellenos else "none",
        label=label_puntos
    )

    ax.plot(
        x_curve,
        y_curve,
        color=color_linea,
        linewidth=anchura_linea,
        label=label_linea
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(xlabel, fontsize=tamano_fuente_ejes)
    ax.set_ylabel(ylabel, fontsize=tamano_fuente_ejes)

    ax.tick_params(labelsize=tamano_fuente_ticks)

    for spine in ax.spines.values():
        spine.set_linewidth(grosor_ejes)

    if titulo is not None:
        ax.set_title(titulo, fontsize=tamano_fuente_titulo)

    if mostrar_leyenda:
        ax.legend()

    plt.tight_layout()

    if guardar_pdf:
        fig.savefig(out_file, format="pdf", bbox_inches="tight")

    plt.show()


simple_plot(x_fit, y_fit, a, b)
'''

    with open(nombre_script, "w") as f:
        f.write(contenido)

def get_install_python(python_bin=None):
    if python_bin is not None:
        return os.path.abspath(python_bin)

    install_python = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".toolkit2venv", "bin", "python")
    )
    if os.path.isfile(install_python) and os.access(install_python, os.X_OK):
        return install_python

    return os.path.abspath(sys.executable)

def guardar_script_shell(nombre_script, python_bin=None):
    script_abs_path = os.path.abspath(nombre_script)
    script_name = os.path.basename(script_abs_path)
    shell_path = os.path.splitext(script_abs_path)[0] + ".sh"
    python_path = get_install_python(python_bin)

    contenido = f'''#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
exec {python_path!r} "$SCRIPT_DIR/{script_name}" "$@"
'''

    with open(shell_path, "w") as f:
        f.write(contenido)

    os.chmod(shell_path, 0o755)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Métricas de salida: Ecal=error calibrado (%), Ee=error estimado de entrada (%), E=error empírico (%), DE=|Ecal-E| (%)."
    )

    parser.add_argument("archivo1", type=str) # Archivo de calibración
    parser.add_argument("archivo2", type=str) # Archivo de evaluación
    parser.add_argument("K", type=int, nargs="?")
    parser.add_argument("--remain", action="store_true") # Si hay resto, dejarlo como grupo final separado
    parser.add_argument("--trim", type=str, default = "False") # Si hace triming o no (true o false)
    parser.add_argument("--lim", type=float, default = 0.0) # Limite para el triming
    parser.add_argument("--outdir", type=str, default = "./experiments") # Directorio raíz de salida
    parser.add_argument("--grouped", action="store_true") # Los archivos ya estan agrupados

    args = parser.parse_args()

    archivo1 = args.archivo1
    archivo2 = args.archivo2
    K = args.K
    grouped = args.grouped

    if K is None:
        parser.error("K es obligatorio. Con --grouped se usa sólo para nombrar el experimento")

    t = args.remain
    trim = (args.trim == "True")
    lim = args.lim
    outdir = os.path.abspath(args.outdir)
    t_flag = int(t)
    trim_flag = int(trim)
    suffix = f"_grouped_K{K}_t{t_flag}" if grouped else f"K{K}"

    experimento = f"simplex_grouped_K{K}_t{t_flag}_tr{trim_flag}_lim{lim}" if grouped else f"simplex_K{K}_t{t_flag}_tr{trim_flag}_lim{lim}"
    carpeta = os.path.join(outdir, experimento)
    grouped_dir = os.path.join(carpeta, "grouped")
    predictions_dir = os.path.join(carpeta, "predictions")
    plots_dir = os.path.join(carpeta, "plots")
    scripts_dir = os.path.join(carpeta, "scripts")

    if not grouped:
        os.makedirs(grouped_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    if grouped:
        calibration, evaluation = given_grouped(archivo1, archivo2)
        cal_agrup_path = os.path.abspath(archivo1)
        eval_agrup_path = os.path.abspath(archivo2)
    else:
        calibration, evaluation = CalibEvDone(archivo1, archivo2)
        cal_agrup_path = os.path.join(grouped_dir, f"calibrationK{K}")
        eval_agrup_path = os.path.join(grouped_dir, f"evaluationK{K}")
    metadata_path = os.path.join(carpeta, "metadata.json")

    print()
    print("#"*50)
    print()
    if grouped:
        print("Resultados del experimento con datos agrupados:")
    else:
        print(f"Resultados del experimento con K={K}:")
    print('-'*50)
    print()

    print('Calibración:')
    if grouped:
        print(f"Leídos {len(calibration)} grupos")
    else:
        calibration = agrupar(K, calibration, cal_agrup_path, t)
    print()

    print('Evaluación:')
    if grouped:
        print(f"Leídos {len(evaluation)} grupos")
    else:
        evaluation = agrupar(K, evaluation, eval_agrup_path, t)

    m_calib,x_calib,y_calib = get_mxy(calibration)
    m_eval,x_eval,y_eval = get_mxy(evaluation)

    a,b,ws = train_model_simplex(x_calib, y_calib, m_calib)

    print()
    print('Resultados del ajuste:')
    print('-'*50)
    print()
    print(f"Parametros del ajuste --> a={a}, b={b}")
    print()
    print(f"WSSR --> {ws}")
    print()

    predictions_file = os.path.join(predictions_dir, f"predictions{suffix}")
    Ecal, Ee, E, DE, dE, rDE, rdE = evaluate(K, a, b, evaluation, predictions_file)

    print(f"Ecal: {Ecal}")
    print(f"Ee: {Ee}")
    print(f"E: {E}")
    print(f"DE: {DE}")
    print(f"dE: {dE}")
    print(f"rDE: {rDE}")
    print(f"rdE: {rdE}")
    print()
    print("#"*50)
    print()

    def action(x_fit, y_fit,a, b, plots_dir, scripts_dir, name='Calib', path=cal_agrup_path):

        plot_pdf_path = os.path.join(plots_dir, f"SimplePlot{name}{suffix}.pdf")
        simple_plot(x_fit, y_fit, a, b, plot_pdf_path)

        data_path = os.path.abspath(path)
        plot_pdf_abs_path = os.path.abspath(plot_pdf_path)

        plot_code_path = os.path.join(scripts_dir, f"{name}_plot_script{suffix}.py")
        guardar_script_plot(data_path, a, b, plot_code_path, plot_pdf_abs_path)
        guardar_script_shell(plot_code_path)

        return plot_pdf_path, plot_code_path

    calib_plot_pdf, calib_plot_script = action(x_calib, y_calib, a, b, plots_dir, scripts_dir, name='Calib', path=cal_agrup_path)
    eval_plot_pdf, eval_plot_script = action(x_eval, y_eval, a, b, plots_dir, scripts_dir, name='Eval', path=eval_agrup_path)
    print()

    metadata = {
        "inputs": {
            "archivo1": os.path.abspath(archivo1),
            "archivo2": os.path.abspath(archivo2),
            "K": K,
            "t": t,
            "grouped": grouped,
            "trim": trim,
            "lim": lim,
            "outdir": outdir,
            "experiment_dir": os.path.abspath(carpeta)
        },
        "results": {
            "a": a,
            "b": b,
            "WSSR": ws,
            "Ecal": Ecal,
            "Ee": Ee,
            "E": E,
            "DE": DE,
            "dE": dE,
            "rDE": rDE,
            "rdE": rdE
        }
    }

    if trim:
        m_calib_trim, x_calib_trim, y_calib_trim = do_trim(lim, x_calib, y_calib, m_calib, a, b)

        trim_dir = os.path.join(carpeta, "trim")
        trim_grouped_dir = os.path.join(trim_dir, "grouped")
        trim_predictions_dir = os.path.join(trim_dir, "predictions")
        trim_plots_dir = os.path.join(trim_dir, "plots")
        trim_scripts_dir = os.path.join(trim_dir, "scripts")

        os.makedirs(trim_grouped_dir, exist_ok=True)
        os.makedirs(trim_predictions_dir, exist_ok=True)
        os.makedirs(trim_plots_dir, exist_ok=True)
        os.makedirs(trim_scripts_dir, exist_ok=True)

        cal_agrup_path_trim = os.path.join(trim_grouped_dir, f"calibration_trim{suffix}")
        
        with open(cal_agrup_path_trim, "w") as out:
            for m, s1, s2 in zip(m_calib_trim, x_calib_trim, y_calib_trim):
                out.write(f"{m} {s1} {s2}\n")
        
        a_trim, b_trim, ws_trim = train_model_simplex(x_calib_trim, y_calib_trim, m_calib_trim)
        print()
        print('Resultados del ajuste con datos trimados:')
        print('-'*50)
        print()
        print(f"Parametros del ajuste --> a={a_trim}, b={b_trim}")
        print()
        print(f"WSSR --> {ws_trim}")
        print()
        predictions_file_trim = os.path.join(trim_predictions_dir, f"predictions_trim{suffix}")
        Ecal_trim, Ee_trim, E_trim, DE_trim, dE_trim, rDE_trim, rdE_trim = evaluate(K, a_trim, b_trim, evaluation, predictions_file_trim)
        print(f"Ecal: {Ecal_trim}")
        print(f"Ee: {Ee_trim}")
        print(f"E: {E_trim}")
        print(f"DE: {DE_trim}")
        print(f"dE: {dE_trim}")
        print(f"rDE: {rDE_trim}")
        print(f"rdE: {rdE_trim}")
        print()
        trim_plot_pdf, trim_plot_script = action(x_calib_trim, y_calib_trim, a_trim, b_trim, trim_plots_dir, trim_scripts_dir, name='Calib_Trim', path=cal_agrup_path_trim)
        trim_eval_plot_pdf, trim_eval_plot_script = action(x_eval, y_eval, a_trim, b_trim, trim_plots_dir, trim_scripts_dir, name='Eval_Trim', path=eval_agrup_path)
        print()

        metadata["trim_results"] = {
            "a_trim": a_trim,
            "b_trim": b_trim,
            "WSSR_trim": ws_trim,
            "Ecal_trim": Ecal_trim,
            "Ee_trim": Ee_trim,
            "E_trim": E_trim,
            "DE_trim": DE_trim,
            "dE_trim": dE_trim,
            "rDE_trim": rDE_trim,
            "rdE_trim": rdE_trim
        }

    guardar_metadata(metadata_path, metadata)



        



        



    
