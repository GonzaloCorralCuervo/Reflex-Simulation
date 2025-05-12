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

