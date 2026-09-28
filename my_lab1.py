# =====================================================================
# ЛАБОРАТОРНАЯ РАБОТА № 1. Основы программного моделирования мобильного робота
# Код взят из методички. Комментарии после # — мои пояснения для понимания.
# =====================================================================

# ---------- ШАГ 1. Первый запуск программы ----------
print("Лабораторная работа № 1")
print("Моделирование движения агента")
student_name = "Дарбека Елена"
group_name = "МФБ2631"
print("Студент:", student_name)
print("Группа:", group_name)

# ---------- ШАГ 2. Арифметическая модель движения ----------
print("\n--- Шаг 2 ---")
x0 = 1.0  # начальная координата, м
vx = 2.0  # скорость, м/с
t = 3.0  # время, с
x = x0 + vx * t
print("Начальная координата:", x0, "м")
print("Скорость:", vx, "м/с")
print("Время:", t, "с")
print("Конечная координата:", x, "м")

# ---------- ШАГ 3. Самостоятельное изменение параметров ----------
print("\n--- Шаг 3 ---")
print("Опыт 2:")
x0 = 0.0
vx = 1.5
t = 4.0
x = x0 + vx * t
print("Начальная координата:", x0, "м")
print("Скорость:", vx, "м/с")
print("Время:", t, "с")
print("Конечная координата:", x, "м")

print("Опыт 3:")
x0 = 5.0
vx = -2.0
t = 3.0
x = x0 + vx * t
print("Начальная координата:", x0, "м")
print("Скорость:", vx, "м/с")
print("Время:", t, "с")
print("Конечная координата:", x, "м")

# ---------- ШАГ 4. Одномерное движение в цикле ----------
print("\n--- Шаг 4 ---")
import numpy as np
import matplotlib.pyplot as plt

x = 0.0
vx = 1.5
dt = 0.2
total_time = 4.0
num_steps = int(total_time / dt)  # 4.0 / 0.2 = 20 шагов

time_history = [0.0]  # список моментов времени, начинаем с t=0
x_history = [x]  # список координат, начинаем с x0

for k in range(num_steps):
    x = x + vx * dt  # x(k+1) = x(k) + vx*dt  — формула из задания
    current_time = (k + 1) * dt
    time_history.append(current_time)
    x_history.append(x)

print("Количество шагов:", num_steps)
print("Конечное время:", time_history[-1], "с")
print("Конечная координата:", x_history[-1], "м")

# ---------- ШАГ 5. Построение графика x(t) ----------
plt.figure(figsize=(8, 4))
plt.plot(time_history, x_history, marker="o", label="x(t)")
plt.xlabel("Время, с")
plt.ylabel("Координата x, м")
plt.title("Одномерное движение агента")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- ШАГ 6. Анализ — ответы в отчёте (см. ниже, кода нет) ----------

# ---------- ШАГ 7. Использование функции ----------
print("\n--- Шаг 7 ---")


def update_coordinate(x, vx, dt):
    new_x = x + vx * dt
    return new_x


x = 2.0
vx = -0.5
dt = 0.1
new_x = update_coordinate(x, vx, dt)
print("Старая координата:", x)
print("Новая координата:", new_x)

# ---------- ШАГ 8. Двумерное движение агента ----------
print("\n--- Шаг 8 ---")


def update_position(position, velocity, dt):
    """Вычисляет положение агента на следующем шаге."""
    return position + velocity * dt


# Параметры модели
position = np.array([1.0, 2.0], dtype=float)  # p(0) = [x, y]
velocity = np.array([0.8, 0.4], dtype=float)  # v = [vx, vy]
dt = 0.1
total_time = 5.0
num_steps = int(total_time / dt)  # 50 шагов

# История моделирования
time_history = [0.0]
position_history = [position.copy()]

# Основной цикл
for k in range(num_steps):
    position = update_position(position, velocity, dt)
    current_time = (k + 1) * dt
    time_history.append(current_time)
    position_history.append(position.copy())

# Преобразование списка в массив
time_history = np.array(time_history)
position_history = np.array(position_history)
x_history = position_history[:, 0]  # все x (первый столбец)
y_history = position_history[:, 1]  # все y (второй столбец)

# Вывод результатов
print("Количество шагов:", num_steps)
print("Начальное положение:", position_history[0])
print("Конечное положение:", position_history[-1])
print("Модуль скорости:", np.linalg.norm(velocity))

# ---------- ШАГ 9. Графики координат во времени ----------
plt.figure(figsize=(9, 4))
plt.plot(time_history, x_history, label="x(t)")
plt.plot(time_history, y_history, label="y(t)")
plt.xlabel("Время, с")
plt.ylabel("Координата, м")
plt.title("Координаты агента во времени")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- ШАГ 10. Траектория на плоскости ----------
plt.figure(figsize=(6, 6))
plt.plot(x_history, y_history, label="Траектория")
plt.scatter(x_history[0], y_history[0], color="green", s=80, label="Начало")
plt.scatter(x_history[-1], y_history[-1], color="red", s=80, marker="x", label="Конец")
plt.xlabel("x, м")
plt.ylabel("y, м")
plt.title("Траектория агента")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- ШАГ 11. Показатели движения ----------
print("\n--- Шаг 11 ---")
displacements = position_history[1:] - position_history[:-1]  # смещения за каждый шаг
segment_lengths = np.linalg.norm(displacements, axis=1)  # длина каждого шага
path_length = np.sum(segment_lengths)  # длина траектории
total_displacement = position_history[-1] - position_history[0]
displacement_norm = np.linalg.norm(total_displacement)
print("Перемещение:", total_displacement)
print("Модуль перемещения:", displacement_norm)
print("Длина траектории:", path_length)


# =====================================================================
# ШАГИ 12–14. ЭКСПЕРИМЕНТЫ. Методичка код не даёт — оборачиваем шаги 8–11
# в функцию и вызываем её с новыми параметрами.
# =====================================================================
def simulate(p0, v, dt, total_time, title):
    position = np.array(p0, dtype=float)
    velocity = np.array(v, dtype=float)
    num_steps = int(total_time / dt)
    time_history = [0.0]
    position_history = [position.copy()]
    for k in range(num_steps):
        position = update_position(position, velocity, dt)
        time_history.append((k + 1) * dt)
        position_history.append(position.copy())
    position_history = np.array(position_history)
    x_history = position_history[:, 0]
    y_history = position_history[:, 1]

    displacements = position_history[1:] - position_history[:-1]
    path_length = np.sum(np.linalg.norm(displacements, axis=1))
    total_displacement = position_history[-1] - position_history[0]

    print("\n" + title)
    print("Количество шагов:", num_steps)
    print("Начальное положение:", position_history[0])
    print("Конечное положение:", position_history[-1])
    print("Модуль скорости:", np.linalg.norm(velocity))
    print("Перемещение:", total_displacement)
    print("Модуль перемещения:", np.linalg.norm(total_displacement))
    print("Длина траектории:", path_length)

    plt.figure(figsize=(6, 6))
    plt.plot(x_history, y_history, label="Траектория")
    plt.scatter(x_history[0], y_history[0], color="green", s=80, label="Начало")
    plt.scatter(
        x_history[-1], y_history[-1], color="red", s=80, marker="x", label="Конец"
    )
    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- ШАГ 12. Эксперимент 1: другое начальное положение ----------
simulate([-2.0, 3.0], [0.8, 0.4], 0.1, 5.0, "Эксперимент 1: p(0) = [-2, 3]")

# ---------- ШАГ 13. Эксперимент 2: другая скорость ----------
simulate([1.0, 2.0], [-0.5, 1.0], 0.1, 5.0, "Эксперимент 2: v = [-0.5, 1]")

# ---------- ШАГ 14. Эксперимент 3: разный шаг моделирования ----------
for dt_exp in [0.5, 0.1, 0.02]:
    simulate([1.0, 2.0], [0.8, 0.4], dt_exp, 5.0, f"Эксперимент 3: dt = {dt_exp}")
