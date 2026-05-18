import numpy as np
import math
from scipy.optimize import minimize_scalar
import os
import argparse
import json

def CalibEvDone(path1,path2): # Lee los archivos de calibración y evaluación, devuelve dos listas de listas de floats (calibración y evaluación)
    calibration = []
    with open(path1, "r") as f:
        for i in f:
            l = i.strip()
            l = l.split()
            l = [float(x) for x in l]
            calibration.append(l)

    evaluation = []
    with open(path2, "r") as f:
        for i in f:
            l = i.strip()
            l = l.split()
            l = [float(x) for x in l]
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

def get_data(data):

    y_true = []
    est_true = []
    real_pos = []
    probs = []

    for row in data:

        y = int(row[0])
        est = float(row[1])
        real = float(row[2])
        p = row[4:]

        y_true.append(y)
        est_true.append(est)
        real_pos.append(real)
        probs.append(p)

    y_true = np.asarray(y_true, dtype=int)
    est_true = np.asarray(est_true, dtype=float)
    real_pos = np.asarray(real_pos, dtype=float)
    probs = np.asarray(probs, dtype=float)

    return y_true, est_true,real_pos, probs

def apply_t(probs, T, eps=1e-15): # Aplica la temperatura T a las probabilidades utilizando la fórmula: p_i' = p_i^(1/T) / sum(p_j^(1/T)), ya que se aplica sobre salidas de una softmax, no logits.

    if T <= 0:
        raise ValueError("T debe ser > 0")

    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1.0)
    powered = np.power(probs, 1.0 / T)
    scaled_probs = powered / powered.sum(axis=1, keepdims=True)
    return scaled_probs


def optimize_temperature_ce(probs,pos): # Optimiza la temperatura T para minimizar la entropía cruzada entre las probabilidades calibradas y las etiquetas verdaderas.

    def cross_entropy(pos_calib, probs, eps=1e-15):
        probs = np.clip(probs, eps, 1.0)
        p_true = []
        for i in range(len(pos_calib)):
            p_true.append(probs[i, int(pos_calib[i])])
        p_true = np.array(p_true)
        ce = -np.mean(np.log(p_true))
        return float(ce)
    
    def objective(T):
        scaled_probs = apply_t(probs, T)
        return cross_entropy(pos, scaled_probs)

    result = minimize_scalar(
        objective,
        bounds=(0.01,20),   
        method="bounded"
    )

    return float(result.x)

def optimize_temperature_dE(y_calib, probs_calib, K, t): # Optimiza la temperatura T para minimizar el error absoluto medio (dE) entre el error estimado y el error empírico, utilizando el proceso de agrupación y evaluación definido en las funciones agrupar y evaluate_ts.
    
    def objective(T):
        scaled_probs = apply_t(probs_calib, T)
        estimation_result = estimation(y_calib, scaled_probs)
        parsed = agrupar(K, estimation_result, t)
        _, _, _, dE, _, _ = evaluate_ts(parsed,None)
        return dE

    result = minimize_scalar(
        objective,
        bounds=(0.01, 20.0),   
        method="bounded"
    )

    return float(result.x)

def optimize_temperature_DE(y_calib, probs_calib, K, t): # Optimiza la temperatura T para minimizar el error absoluto medio (DE) entre el error estimado y el error empírico, utilizando el proceso de agrupación y evaluación definido en las funciones agrupar y evaluate_ts.
    
    def objective(T):
        scaled_probs = apply_t(probs_calib, T)
        estimation_result = estimation(y_calib, scaled_probs)
        parsed = agrupar(K, estimation_result, t)
        _, _, DE, _, _, _ = evaluate_ts(parsed,None)
        return DE

    result = minimize_scalar(
        objective,
        bounds=(0.01, 20.0),   
        method="bounded"
    )

    return float(result.x)
     

def estimation(real, probs): # Calcula la estimación del error como 1 - max(p_i) para cada muestra
    real = np.array(real)
    probs = np.array(probs)
    
    est = 1.0 - np.max(probs, axis=1)
    
    return list(zip(real, est))

def agrupar(K, estimated,t): # Devuelve una lista de tuplas (n, avg_real, avg_est) para cada bloque de tamaño n (M o r)G

    estimated.sort(key=lambda x: x[1])

    s1 = 0.0
    s2 = 0.0
    count = 0

    if K <= 0:
        raise ValueError("K debe ser un entero positivo.")
    if K > len(estimated):
        raise ValueError("K no puede ser mayor que el número de datos en la calibración.")
    
    M = len(estimated) // K
    r = len(estimated) % K

    vals = []

    limite = len(estimated)
    if r != 0:
        if t:
            limite -= r
        else:
            limite -= (M + r)

    for a, b in estimated[:limite]:
        s1 += b
        s2 += a
        count += 1

        if count % M == 0:
            vals.append((M,s1/M, s2/M))
            s1 = 0.0
            s2 = 0.0
    
    if r != 0:

        if t:                           # t --> True: Añade un bloque de tamaño r al final
            s1 = 0.0
            s2 = 0.0
            for a,b in estimated[-r:]:
                s1 += b
                s2 += a

            vals.append((r, s1/r, s2/r))
        
        else:                       # t --> False: Vuelve a calcular el último bloque de tamaño M incluyendo los últimos r elementos (tamaño nuevo r)
            s1 = 0.0
            s2 = 0.0
            c = M + r
            for a,b in estimated[-c:]:
                s1 += b
                s2 += a
            vals.append((c, s1/c, s2/c))

    return vals

def group_results(empirical_error, probs, T, K, t):
    scaled_probs = apply_t(probs, T)
    estimation_result = estimation(empirical_error, scaled_probs)
    parsed = agrupar(K, estimation_result, t)

    return parsed

def input_estimated_error(values):
    values = np.asarray(values, dtype=float)
    return 100 * float(np.mean(values))

def grouped_input_estimated_error(parsed):
    sec = 0.0
    n = 0.0
    for m, est_error, _ in parsed:
        n += m
        sec += m * float(est_error)
    return 100 * (sec / n)

def evaluate_ts(parsed, output_file): # Calcula las métricas de error Ecal, E, DE, dE, rDE y rdE a partir de los resultados agrupados (n, avg_real, avg_est) y guarda los resultados en un archivo de texto.
    sec = 0.0
    sem = 0.0
    sde = 0.0
    n = 0.0

    out = None
    if output_file:
        out = open(output_file, "w")

    try:
        for m, est_error, emp_error in parsed:
            epi = float(est_error)   # estimado
            emi = float(emp_error)   # empírico real

            n += m
            sec += m * epi
            sem += m * emi
            sde += m * abs(emi - epi)

            if out:
                out.write(f"{m} {est_error:.6f} {emp_error:.6f}\n")
    finally:
        if out:
            out.close()


    eca = sec / n
    eem = sem / n
    ede = sde / n

    Ecal = 100 * eca
    E = 100 * eem
    DE = 100 * abs(eca - eem)
    dE = 100 * ede
    rDE = DE / eem if eem > 0 else float("inf")
    rdE = dE / eem if eem > 0 else float("inf")

    return Ecal, E, DE, dE, rDE, rdE

def save_grouped_results(parsed, output_file):
    with open(output_file, "w") as out:
        for m, est_error, emp_error in parsed:
            out.write(f"{m} {est_error:.6f} {emp_error:.6f}\n")

def guardar_metadata(metadata_path, metadata):
    with open(metadata_path, "w") as out:
        json.dump(metadata, out, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Métricas de salida: Ecal=error calibrado (%), Ee=error estimado de entrada (%), E=error empírico (%), DE=|Ecal-E| (%)."
    )

    parser.add_argument("archivo1", type=str, help="Archivo de calibración")
    parser.add_argument("archivo2", type=str, help="Archivo de evaluación")
    parser.add_argument("K", type=int, nargs="?", help="Número de grupos")
    parser.add_argument(
        "optimization",
        type=str,
        nargs="?",
        choices=["ce", "dE", "DE"],
        help="Criterio de optimización de T"
    )
    parser.add_argument("--remain", action="store_true", help="Si hay resto, dejarlo como grupo final separado")
    parser.add_argument("--outdir", type=str, default="./experiments", help="Directorio raíz de salida")
    parser.add_argument("--grouped", action="store_true", help="Los archivos ya estan agrupados")

    args = parser.parse_args()

    try:
        archivo1 = args.archivo1
        archivo2 = args.archivo2
        K = args.K
        opt = args.optimization
        grouped = args.grouped

        if K is None:
            parser.error("K es obligatorio. Con --grouped se usa sólo para nombrar el experimento")
        if not grouped and opt is None:
            parser.error("optimization es obligatorio si no se usa --grouped")

        t = args.remain
        outdir = os.path.abspath(args.outdir)
        t_flag = int(t)
        suffix = f"_grouped_K{K}_t{t_flag}" if grouped else f"K{K}"

        if grouped:
            parsed_calib, parsed = given_grouped(archivo1, archivo2)
            Ee = grouped_input_estimated_error(parsed)
            T = None
        else:
            calibration, evaluation = CalibEvDone(archivo1, archivo2)

            y_calib, est_calib, pos_calib, probs_calib = get_data(calibration)
            y_eval, est_eval, pos_eval, probs_eval = get_data(evaluation)
            Ee = input_estimated_error(est_eval)

            if opt == "ce":
                T = optimize_temperature_ce(probs_calib, pos_calib)
            elif opt == "dE":
                T = optimize_temperature_dE(y_calib, probs_calib, K, t)
            else:  # opt == "DE"
                T = optimize_temperature_DE(y_calib, probs_calib, K, t)

        experimento = f"ts_grouped_K{K}_t{t_flag}" if grouped else f"ts_K{K}_t{t_flag}_{opt}"
        carpeta = os.path.join(outdir, experimento)
        grouped_dir = os.path.join(carpeta, "grouped")
        predictions_dir = os.path.join(carpeta, "predictions")
        plots_dir = os.path.join(carpeta, "plots")
        scripts_dir = os.path.join(carpeta, "scripts")
        metadata_path = os.path.join(carpeta, "metadata.json")

        if not grouped:
            os.makedirs(grouped_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(scripts_dir, exist_ok=True)

        if not grouped:
            parsed_calib = group_results(y_calib, probs_calib, T, K, t)
            cal_agrup_path = os.path.join(grouped_dir, f"calibrationK{K}")
            save_grouped_results(parsed_calib, cal_agrup_path)

            parsed = group_results(y_eval, probs_eval, T, K, t)
            eval_agrup_path = os.path.join(grouped_dir, f"evaluationK{K}")
            save_grouped_results(parsed, eval_agrup_path)

        predictions_file = os.path.join(predictions_dir, f"predictions{suffix}")
        Ecal, E, DE, dE, rDE, rdE = evaluate_ts(parsed, predictions_file)

        print("\n=== RESULTADOS ===")
        if grouped:
            print("T   = no optimizada (--grouped)")
        else:
            print(f"T   = {T:.6f}")
        print(f"Ecal = {Ecal:.4f}%")
        print(f"Ee   = {Ee:.4f}%")
        print(f"E   = {E:.4f}%")
        print(f"DE  = {DE:.4f}%")
        print(f"dE  = {dE:.4f}%")
        print(f"rDE = {rDE:.4f}")
        print(f"rdE = {rdE:.4f}")

        metadata = {
            "inputs": {
                "archivo1": os.path.abspath(archivo1),
                "archivo2": os.path.abspath(archivo2),
                "K": K,
                "t": t,
                "grouped": grouped,
                "optimization": opt,
                "outdir": outdir,
                "experiment_dir": os.path.abspath(carpeta)
            },
            "results": {
                "T": T,
                "Ecal": Ecal,
                "Ee": Ee,
                "E": E,
                "DE": DE,
                "dE": dE,
                "rDE": rDE,
                "rdE": rdE
            }
        }

        guardar_metadata(metadata_path, metadata)

    except FileNotFoundError as e:
        print(f"Error: no se encontró el archivo: {e.filename}")
        raise SystemExit(1)

    except ValueError as e:
        print(f"Error de valor: {e}")
        raise SystemExit(1)

    except Exception as e:
        print(f"Error inesperado: {type(e).__name__}: {e}")
        raise SystemExit(1)
