import numpy as np

# входные данные (логическая операция AND)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# правильные ответы
y = np.array([0, 0, 0, 1])

# веса и смещение
weights = np.zeros(2)
bias = 0
learning_rate = 0.1

# функция активации (шаговая)
def step(x):
    return 1 if x >= 0 else 0

# обучение
for epoch in range(10):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step(linear_output)

        error = y[i] - prediction

        # обновление весов
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Обученные веса:", weights)
print("Смещение:", bias)

# проверка
for i in range(len(X)):
    output = step(np.dot(X[i], weights) + bias)
    print(f"{X[i]} -> {output}")