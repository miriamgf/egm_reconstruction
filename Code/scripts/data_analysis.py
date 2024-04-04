import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Datos de ejemplo
x = range(10000)
y = [i**2 for i in x]

# Crear la figura y los ejes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Dibujar los datos
line, = ax.plot(x, y)

# Agregar un slider para ajustar la posición de la ventana
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
window_slider = Slider(ax_slider, 'Ventana', 0, 10000, valinit=0)

def update(val):
    window_start = int(window_slider.val)
    window_end = window_start + 200  # Tamaño de la ventana
    ax.set_xlim(window_start, window_end)
    fig.canvas.draw_idle()

window_slider.on_changed(update)

plt.show()
