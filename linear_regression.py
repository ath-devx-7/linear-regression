import numpy as np
import matplotlib.pyplot as plt

def take_input_arrays():
    """ Take input feature array x and target array y from the user """
    while True:
        try:
            x = list(map(float, input("\nInput features separated by spaces: ").split()))
            y = list(map(float, input("Target values separated by spaces: ").split()))

            if len(x) != len(y):
                raise ValueError("Input features and target values must have the same length.")

            return np.array(x), np.array(y)

        except ValueError:
            print("Invalid input. Ensure numeric values and matching lengths.")

def predict(x, m, b):
    """ Predict target values using linear regression """
    return m * x + b

def mean_squared_error(y, y_pred):
    """ Compute Mean Squared Error """
    return np.mean((y - y_pred) ** 2)

def gradient_descent(x, y, y_pred):
    """ Compute gradients for m and b. """
    n = len(y)
    dm = (-2 / n) * np.sum(x * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)
    return dm, db

def linear_regression(x, y, learning_rate=0.001, epochs_num=10000, tolerance=0.05):
    """
    Train linear regression using gradient descent with feature scaling.
    Returns parameters converted back to original feature space.
    """

    mean_x = np.mean(x)
    std_x = np.std(x)

    if std_x == 0:
        raise ValueError("Standard deviation of input is zero. Cannot scale features.")

    # standardization of input feature
    x_scaled = (x - mean_x) / std_x 

    m, b = 0.0, 0.0
    mse_prev = 0.0

    for epoch in range(1, epochs_num + 1):
        y_pred = predict(x_scaled, m, b)
        mse = mean_squared_error(y, y_pred)

        dm, db = gradient_descent(x_scaled, y, y_pred)
        m -= learning_rate * dm
        b -= learning_rate * db

        if epoch % 250 == 0:
            print(f"Epoch {epoch} | MSE: {mse:.4f}")

        if mse < tolerance:
            print(f"Early stopping at epoch {epoch} with MSE: {mse:.4f}")
            break

        if mse == mse_prev:
            print(f"No improvement in MSE at epoch {epoch}. Stopping training.")
            break
        mse_prev = mse

    m_original = m / std_x
    b_original = b - (m * mean_x) / std_x

    return m_original, b_original, mse

def plot_regression(x, y, m, b):
    """ Visualize data points and regression line """
    plt.figure(figsize=(10, 7))

    plt.scatter(x, y, color="blue", label="Actual data", s=40)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")

    plt.xlabel("Input feature (x)")
    plt.ylabel("Target value (y)")
    plt.title("Linear Regression from Scratch")
    plt.legend()
    plt.grid(True)

    plt.show()

x, y = take_input_arrays()

m, b, mse = linear_regression(
    x,
    y,
    learning_rate=0.001,
    epochs_num=10000,
    tolerance=0.05
)

print("\nFinal trained model (original scale):")
print(f"m = {m:.4f}")
print(f"b = {b:.4f}")
print(f"Final MSE = {mse:.4f}")

plot_regression(x, y, m, b)