import numpy as np
from itertools import combinations

n = 12  # удвоенное число возможных систем скольжения
tau = 15.0  # Критическое напряжение (МПа)
tol = 1e-5  # Допустимая погрешность

# Создание данных систем скольжения (СС) из Таблицы 1
# Нормали (n) и векторы Бюргерса (b) в ненормированном виде
n_unnorm = [
    [1, 1, 1], [1, 1, 1], [1, 1, 1],
    [-1, 1, 1], [-1, 1, 1], [-1, 1, 1],
    [1, -1, 1], [1, -1, 1], [1, -1, 1],
    [-1, -1, 1], [-1, -1, 1], [-1, -1, 1],
    [1, 1, 1], [1, 1, 1], [1, 1, 1],
    [-1, 1, 1], [-1, 1, 1], [-1, 1, 1],
    [1, -1, 1], [1, -1, 1], [1, -1, 1],
    [-1, -1, 1], [-1, -1, 1], [-1, -1, 1]
]

b_unnorm = [
    [-1, 1, 0], [-1, 0, 1], [0, -1, 1],
    [1, 1, 0], [1, 0, 1], [0, -1, 1],
    [1, 1, 0], [0, 1, 1], [1, 0, -1],
    [-1, 1, 0], [0, 1, 1], [-1, 0, 1],
    [1, -1, 0], [1, 0, -1], [0, 1, -1],
    [-1, -1, 0], [-1, 0, -1], [0, 1, -1],
    [-1, -1, 0], [0, -1, -1], [-1, 0, 1],
    [1, -1, 0], [0, -1, -1], [1, 0, -1]
]

# Нормирование векторов
n_vectors = []
b_vectors = []
for i in range(24):
    n = np.array(n_unnorm[i], dtype=float)
    n_norm = n / np.linalg.norm(n)
    n_vectors.append(n_norm)

    b = np.array(b_unnorm[i], dtype=float)
    b_norm = b / np.linalg.norm(b)
    b_vectors.append(b_norm)


def main():
    # Функция вычисления коэффициентов m для СС
    global tau

    def get_coeffs(n, b):
        A = n[0] * b[0] - n[2] * b[2]
        B = n[1] * b[1] - n[2] * b[2]
        C = n[0] * b[1] + n[1] * b[0]
        D = n[0] * b[2] + n[2] * b[0]
        E = n[1] * b[2] + n[2] * b[1]
        return [A, B, C, D, E]

    # Вычисление коэффициентов m для всех СС
    coeffs_list = []
    for i in range(24):
        coeffs = get_coeffs(n_vectors[i], b_vectors[i])
        coeffs_list.append(coeffs)

    # Функция вычисления касательного напряжения
    def calc_tau(dev_components, coeffs):
        s11, s22, s12, s13, s23 = dev_components
        return coeffs[0] * s11 + coeffs[1] * s22 + coeffs[2] * s12 + coeffs[3] * s13 + coeffs[4] * s23

    # Перебор всех комбинаций из 5 СС
    valid_vertices = []
    all_combinations = list(combinations(range(24), 5))
    total_combinations = len(all_combinations)

    print(f"Всего комбинаций для перебора: {total_combinations}")

    for idx, comb in enumerate(all_combinations):
        if (idx + 1) % 5000 == 0:
            print(f"Обработано {idx + 1}/{total_combinations} комбинаций...")

        # Составление матрицы системы
        A_mat = np.zeros((5, 5))
        b_vec = np.full(5, tau)

        for i, ss_idx in enumerate(comb):
            A_mat[i, :] = coeffs_list[ss_idx]

        try:
            # Решение СЛАУ
            x = np.linalg.solve(A_mat, b_vec)

            # Проверка условий для всех 24 СС
            valid = True
            for i in range(24):
                tau = calc_tau(x, coeffs_list[i])

                # Для активных СС должно быть равенство
                if i in comb:
                    if abs(tau - tau) > tol:
                        valid = False
                        break
                # Для остальных - неравенство
                else:
                    if tau > tau + tol:
                        valid = False
                        break

            if valid:
                # Формирование полного девиатора напряжений
                s11, s22, s12, s13, s23 = x
                s33 = -s11 - s22
                deviator = np.array([
                    [s11, s12, s13],
                    [s12, s22, s23],
                    [s13, s23, s33]
                ])
                valid_vertices.append((comb, deviator))

        except np.linalg.LinAlgError:
            # Пропуск вырожденных систем
            continue

    # Вывод результатов
    print(f"\nНайдено вершин: {len(valid_vertices)}")
    for i, (active_ss, dev) in enumerate(valid_vertices):
        print(f"\nВершина {i + 1}:")
        print(f"Активные СС: {active_ss}")
        print("Девиатор напряжений (МПа):")
        print(f"[[{dev[0, 0]:.8f}, {dev[0, 1]:.8f}, {dev[0, 2]:.8f}]")
        print(f" [{dev[1, 0]:.8f}, {dev[1, 1]:.8f}, {dev[1, 2]:.8f}]")
        print(f" [{dev[2, 0]:.8f}, {dev[2, 1]:.8f}, {dev[2, 2]:.8f}]]")

    # Сохранение результатов в файл
    with open("yield_surface_vertices.txt", "w") as f:
        f.write(f"Найдено вершин: {len(valid_vertices)}\n\n")
        for i, (active_ss, dev) in enumerate(valid_vertices):
            f.write(f"Вершина {i + 1}:\n")
            f.write(f"Активные СС: {active_ss}\n")
            f.write("Девиатор напряжений (МПа):\n")
            f.write(f"[[{dev[0, 0]:.8f}, {dev[0, 1]:.8f}, {dev[0, 2]:.8f}]\n")
            f.write(f" [{dev[1, 0]:.8f}, {dev[1, 1]:.8f}, {dev[1, 2]:.8f}]\n")
            f.write(f" [{dev[2, 0]:.8f}, {dev[2, 1]:.8f}, {dev[2, 2]:.8f}]]\n\n")

    print("Результаты сохранены в файл 'yield_surface_vertices.txt'")


if __name__ == "__main__":
    main()
