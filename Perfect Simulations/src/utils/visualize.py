import matplotlib.pyplot as plt

def plot_and_save_figure(x, y, z, xlabel, ylabel, title, **kwargs):
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    # ax = ax.flatten()  # Flatten 2D array to 1D for easier indexing

    for idx, (alpha, _) in enumerate(x.items()):
        rhos, biases = zip(*y[alpha])
        if z:
            rhos_z, values_z = zip(*z[alpha])
       # if idx < len(ax):  # Make sure we don't exceed the number of subplots
        ax.plot(rhos, biases, '-o', label=kwargs.get('label_1', f"(alpha={alpha})"))
        if z:
            ax.plot(rhos_z, values_z, '-o', label=kwargs.get('label_2', "Analytic Bound"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    return fig

