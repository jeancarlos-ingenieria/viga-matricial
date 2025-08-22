#########
def k_elemento(EI, L):
    a = 12 * EI / L**3
    b = 6 * EI / L**2
    c = 4 * EI / L
    d = 2 * EI / L
    return np.array(
        [[a, b, -a, b], [b, c, -b, d], [-a, -b, a, -b], [b, d, -b, c]], dtype=float
    )
