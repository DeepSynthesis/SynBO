import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SNArFlowSimulator:
    def __init__(self):
        self.R = 8.314  # Gas constant J/(mol*K)
        self.T_ref = 90 + 273.15  # Reference temperature in Kelvin
        # Kinetic Parameters (k unit: M^-1 s^-1 at T_ref, Ea unit: J/mol)
        self.params = {
            "k1": {"k_ref": 57.9e-2, "Ea": 33.3e3},  # forms ortho-3
            "k2": {"k_ref": 2.70e-2, "Ea": 35.3e3},  # forms para-4
            "k3": {"k_ref": 0.865e-2, "Ea": 38.9e3},  # 3 -> bis-5
            "k4": {"k_ref": 1.63e-2, "Ea": 44.8e3},  # 4 -> bis-5
        }

    def _get_rate_constants(self, temp_C):
        T = temp_C + 273.15
        ks = {}
        for step, p in self.params.items():
            k = p["k_ref"] * np.exp(-p["Ea"] / self.R * (1 / T - 1 / self.T_ref))
            ks[step] = k
        return ks

    def _ode_system(self, C, t, ks):
        C1, C2, C3, C4, C5 = C
        r1 = ks["k1"] * C1 * C2
        r2 = ks["k2"] * C1 * C2
        r3 = ks["k3"] * C3 * C2
        r4 = ks["k4"] * C4 * C2

        dC1dt = -r1 - r2
        dC2dt = -r1 - r2 - r3 - r4
        dC3dt = r1 - r3
        dC4dt = r2 - r4
        dC5dt = r3 + r4
        return [dC1dt, dC2dt, dC3dt, dC4dt, dC5dt]

    def get_trajectory(self, temperature, time_max_min, conc_1, conc_2):
        """
        Run simulation and return full time and concentration arrays.
        """
        ks = self._get_rate_constants(temperature)
        initial_concentrations = [conc_1, conc_2, 0, 0, 0]
        t_span_sec = np.linspace(0, time_max_min * 60, 100)

        sol = odeint(self._ode_system, initial_concentrations, t_span_sec, args=(ks,))

        return t_span_sec / 60.0, sol


# --- 2. 绘图代码 ---
def plot_figure_5_reproduction():
    sim = SNArFlowSimulator()
    # params
    c1_init = 0.20  # M
    time_max = 2.0  # min

    equivalents_list = [1.5, 4, 7]

    temps_list = [30, 60, 90, 120]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10), sharex=True, sharey=True)

    # Dict mapping species index to style
    # C1 (idx 0): Square, C3 (idx 2): Circle, C4 (idx 3): Triangle_Up, C5 (idx 4): Triangle_Down
    species_styles = {
        0: {"label": "1 (Substrate)", "marker": "s", "color": "black", "ls": "-"},
        2: {"label": "3 (Ortho)", "marker": "o", "color": "black", "ls": "-"},
        3: {"label": "4 (Para)", "marker": "^", "color": "black", "ls": "-"},
        4: {"label": "5 (Bis)", "marker": "v", "color": "black", "ls": "-"},
    }

    for row_idx, eq in enumerate(equivalents_list):
        for col_idx, temp in enumerate(temps_list):
            ax = axes[row_idx, col_idx]

            c2_init = c1_init * eq

            t, C_data = sim.get_trajectory(temp, time_max, c1_init, c2_init)

            mark_interval = 10

            ax.plot(
                t,
                C_data[:, 0],
                marker=species_styles[0]["marker"],
                markevery=mark_interval,
                color="black",
                linewidth=1,
                markersize=4,
                fillstyle="full",
            )

            ax.plot(
                t,
                C_data[:, 2],
                marker=species_styles[2]["marker"],
                markevery=mark_interval,
                color="black",
                linewidth=1,
                markersize=4,
                fillstyle="full",
            )

            ax.plot(
                t,
                C_data[:, 3],
                marker=species_styles[3]["marker"],
                markevery=mark_interval,
                color="black",
                linewidth=1,
                markersize=4,
                fillstyle="full",
            )

            ax.plot(
                t,
                C_data[:, 4],
                marker=species_styles[4]["marker"],
                markevery=mark_interval,
                color="black",
                linewidth=1,
                markersize=4,
                fillstyle="full",
            )

            ax.set_ylim(-0.01, 0.21)
            ax.set_xlim(0, 2.05)

            ax.set_xlabel(r"$\tau$ (min)")

            ax.set_ylabel("Concentration (M)")
            if col_idx == 0:
                ax.text(0., 0.1, f"{eq}Eq.", transform=ax.transData, va="center", ha="right", fontsize=12, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"{temp} °C", fontsize=12)

            plot_num = row_idx * 4 + col_idx + 1
            roman_num = ["(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)", "(vii)", "(viii)", "(ix)", "(x)", "(xi)", "(xii)"]
            ax.text(0.1, 0.20, roman_num[plot_num - 1], transform=ax.transData, va="top", ha="left")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="black", marker="s", label="2,4-DF 1", markersize=6),
        Line2D([0], [0], color="black", marker="o", label="ortho-3", markersize=6),
        Line2D([0], [0], color="black", marker="^", label="para-4", markersize=6),
        Line2D([0], [0], color="black", marker="v", label="bis-5", markersize=6),
    ]

    fig.legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False)

    plt.tight_layout()

    plt.subplots_adjust(bottom=0.08, left=0.08)

    plt.savefig("aaa.png")


if __name__ == "__main__":
    plot_figure_5_reproduction()

# # --- Example: How to integrate into a Bayesian Optimization Loop ---
# if __name__ == "__main__":
#     sim = SNArFlowSimulator()

#     # Define a sample point in the input search space
#     # Temperature: 90 °C
#     # Residence Time: 2.0 min
#     # Concentration 1: 0.19 M
#     # Concentration 2: 0.57 M (Previously 3 equivalents)

#     test_temp = 120
#     test_time = 1.85
#     test_c1 = 0.20
#     test_c2 = 0.20 * 7  # 3.0 equivalents

#     resultado = sim.simulate(
#         temperature=test_temp, residence_time=test_time, conc_1=test_c1, conc_2=test_c2, noise_level=0.0
#     )  # Set to 0.05 for 5% noise

#     print(f"--- Simulation Results ---")
#     print(f"Input Conditions: T={test_temp}C, t={test_time}min, [1]={test_c1}M, [2]={test_c2}M")
#     print("-" * 30)
#     print(f"Yield of Product 3  : {resultado['yield']:.2f} %")
#     print(f"E-factor            : {resultado['e_factor']:.2f}")
#     print(f"Substrate Conversion: {resultado['conversion']:.2f} %")
#     print(f"Product Selectivity : {resultado['selectivity']:.2f} %")
