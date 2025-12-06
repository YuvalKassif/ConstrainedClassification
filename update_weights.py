import numpy as np
import sympy as sp


def calculate_F_and_derivative(N_K_val, N_K_p_val, b_val):
    # Define the function F
    def F(N_K, N_K_p, b):
        return (N_K - N_K_p) ** 2 * (np.tanh(b * (N_K_p - N_K)) + 1)

    # Symbolic computation of the derivative
    N_K_sym = sp.symbols('N_K')
    N_K_p_sym = sp.symbols('N_K_p')
    b_sym = sp.symbols('b')

    # Define the symbolic function
    F_sym = (N_K_sym - N_K_p_sym) ** 2 * (sp.tanh(b_sym * (N_K_p_sym - N_K_sym)) + 1)

    # Compute the symbolic derivative
    dF_dN_K_p_sym = sp.diff(F_sym, N_K_p_sym)

    # Create a numerical function from the symbolic derivative
    dF_dN_K_p_func = sp.lambdify((N_K_sym, N_K_p_sym, b_sym), dF_dN_K_p_sym, 'numpy')

    # Calculate F
    F_val = F(N_K_val, N_K_p_val, b_val)

    # Calculate dF/dN_K_p
    dF_dN_K_p_val = dF_dN_K_p_func(N_K_val, N_K_p_val, b_val)

    return F_val, dF_dN_K_p_val




# print(f"F(N_K={N_K_val}, N_K_p={N_K_p_val}, b={b_val}) = {F_val}")
# print(f"dF/dN_K_p(N_K={N_K_val}, N_K_p={N_K_p_val}, b={b_val}) = {dF_dN_K_p_val}")
