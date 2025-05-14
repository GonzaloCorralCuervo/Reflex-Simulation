import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo
F_max = 22000  # Fuerza isométrica máxima (N)
L_opt = 0.1   # Longitud óptima del músculo (m)
V_max = -12 * L_opt  # Velocidad máxima de contracción (m/s)
K = 5         # Curvatura de la relación fuerza-velocidad
N = 1.5       # Factor de aumento de fuerza excéntrica
c = np.log(0.05)  # Constante para la curva fuerza-longitud
tau = 0.01  # Constante de tiempo de activación (10 ms)
STIM0 = 0.05  # Sesgo de estimulación (mínimo estímulo presente)
G = 2.0  # Ganancia de la retroalimentación
DP = 0.015  # Retardo en la señal sensorial

# Funciones de fuerza-longitud y fuerza-velocidad
def f_l(L_CE):
    return np.exp(c * (np.abs((L_CE - L_opt) / L_opt))**3)

def f_v(V_CE):
    return np.where(V_CE < 0,
                    (V_max - V_CE) / (V_max + K * V_CE),  # Contracción concéntrica
                    (N + (N - 1) * (V_max + V_CE) / (7.56 * K * V_CE - V_max)))  # Excéntrica

# Definir la estimulación
def stimulation_signal(t):
    return min(1, max(0, STIM0 + G * max(0, np.sin(2 * np.pi * t))))

# Ecuación diferencial de activación muscular
def muscle_activation(t, ACT):
    STIM_t = stimulation_signal(t - DP) if t > DP else 0
    return (STIM_t - ACT) / tau

# Resolver activación muscular
t_span = (0, 1)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
ACT0 = [0]
sol = solve_ivp(muscle_activation, t_span, ACT0, t_eval=t_eval)
ACT_values = sol.y[0]

# Calcular F_MTC para diferentes longitudes y velocidades
L_CE_values = np.linspace(0.05, 0.15, len(ACT_values))
V_CE_values = np.gradient(L_CE_values, t_eval)
F_MTC_values = ACT_values * F_max * f_l(L_CE_values) * f_v(V_CE_values)

# Graficar resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_eval, ACT_values, label="Activación Muscular (ACT)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Nivel de Activación")
plt.title("Evolución de la Activación Muscular")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(L_CE_values, F_MTC_values, label="F_MTC (Fuerza del Músculo)", color='r')
plt.xlabel("Longitud del CE (m)")
plt.ylabel("Fuerza F_MTC (N)")
plt.title("Relación Fuerza-Longitud del Músculo")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Parámetros físicos y musculares
g = 9.81
m = 1.0
F_max = 300.0
tau_act = 0.015
tau_deact = 0.06
delay = 0.03  # segundos de retardo
y0, v0, A0 = 1.0, 0.0, 0.0
t_max = 2.0
dt = 0.001
t_eval = np.arange(0, t_max, dt)

# Creamos una lista para guardar la excitación u(t)
u_hist = []

# Función de excitación neural (control simple basado en contacto)
def control_u(y, v):
    if y <= 0.05 and v < 0:
        return min(1.0, -v / 2.0)  # activa más si cae rápido
    elif y <= 0.2 and v < -0.5:
        return 0.5  # preactivación si cae rápido antes del contacto
    else:
        return 0.0


# Sistema dinámico con retardo en excitación
def model(t, state):
    y, v, A = state

    # Evaluamos control y almacenamos en la historia
    u_now = control_u(y, v)
    u_hist.append((t, u_now))

    # Interpolamos u(t - delay)
    if t - delay <= 0:
        u_delayed = 0.0
    else:
        ts, us = zip(*u_hist)
        interp = interp1d(ts, us, kind='previous', fill_value="extrapolate")
        u_delayed = float(interp(t - delay))

    # Dinámica muscular
    tau = tau_act if u_delayed > A else tau_deact
    dA_dt = (u_delayed - A) / tau

    # Dinámica física
    force = -m * g
    if y <= 0:
        force += A * F_max

    return [v, force / m, dA_dt]

# Simulación
sol = solve_ivp(model, [0, t_max], [y0, v0, A0], t_eval=t_eval, method='RK45')

# Resultados
y = sol.y[0]
v = sol.y[1]
A = sol.y[2]
t = sol.t

# Gráficas
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, y, label="Altura", color="red")
plt.axhline(0, color="black", linestyle="dotted")
plt.ylabel("Altura (m)")
plt.title("Pierna con músculo y retardo")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, A, label="Activación muscular", color="green")
plt.ylabel("Activación A(t)")
plt.xlabel("Tiempo (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

