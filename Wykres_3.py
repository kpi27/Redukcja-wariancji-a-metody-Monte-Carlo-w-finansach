import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# Przygotowanie danych
np.random.seed(50)
r = 0.05
sigma = 0.3
T = 1.0
S0 = 50.0
K = 50.0
m = 13
dt = T / m

def get_data(n_paths):
    Z = np.random.normal(0, 1, (n_paths, m))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=1)
    log_prices = np.hstack((np.full((n_paths, 1), np.log(S0)), np.log(S0) + log_prices))
    S = np.exp(log_prices)
    
    
    S_monitoring = S[:, 1:]
    
    # X 
    payoff_arith = np.maximum(np.mean(S_monitoring, axis=1) - K, 0)
    
    # Y 
    S_T = S_monitoring[:, -1]
    
    return payoff_arith, S_T

# Teoretyczne E[X]
X_huge, _ = get_data(1_000_000)
mu_x_true = np.mean(X_huge)

# Symulacja próby
N_small = 50
X, Y = get_data(N_small)

# Obliczenie statystyk
x_bar = np.mean(X)
y_bar = np.mean(Y)
cov_xy = np.cov(X, Y)[0, 1]
var_x = np.var(X, ddof=1)
b_hat = cov_xy / var_x
y_cv = y_bar - b_hat * (x_bar - mu_x_true)

# Rysowanie
fig, ax = plt.subplots(figsize=(9, 7))

# Wykres rozrzutu
ax.scatter(X, Y, facecolors='none', edgecolors='gray', alpha=0.6, s=40, label='Symulacje')

# Linia regresji
x_min_data, x_max_data = min(X), max(X)
x_vals = np.array([x_min_data, x_max_data])
y_vals = y_bar + b_hat * (x_vals - x_bar)
ax.plot(x_vals, y_vals, color='black', linewidth=1.5, label='Linia regresji')

# Kluczowe punkty
ax.plot(x_bar, y_bar, 'o', color='red', markersize=8, label='Średnia zwykła (MC)')
ax.plot(mu_x_true, y_cv, 'o', color='green', markersize=8, label='Średnia skorygowana (CV)')

# Limity
x_all = np.concatenate([X, [mu_x_true, x_bar]])
y_all = np.concatenate([Y, [y_cv, y_bar]])
x_lo, x_hi = min(x_all), max(x_all)
y_lo, y_hi = min(y_all), max(y_all)

pad_x = (x_hi - x_lo) * 0.05
pad_y = (y_hi - y_lo) * 0.05

xlim_lo = x_lo - pad_x
xlim_hi = x_hi + pad_x
ylim_lo = y_lo - pad_y * 3 
ylim_hi = y_hi + pad_y

# Linie rzutujące
ax.plot([x_bar, x_bar], [ylim_lo, y_bar], 'r:', linewidth=1)
ax.plot([xlim_lo, x_bar], [y_bar, y_bar], 'r:', linewidth=1)
ax.plot([mu_x_true, mu_x_true], [ylim_lo, y_cv], 'g--', linewidth=1)
ax.plot([xlim_lo, mu_x_true], [y_cv, y_cv], 'g--', linewidth=1)

def create_smart_ticks(ax_min, ax_max, specials, special_labels):
    locator = ticker.MaxNLocator(nbins=5)
    raw_ticks = locator.tick_values(ax_min, ax_max)
    raw_ticks = [t for t in raw_ticks if ax_min < t < ax_max]
    
    final_ticks = list(specials)
    final_labels = list(special_labels)
    threshold = (ax_max - ax_min) * 0.05
    
    for t in raw_ticks:
        if all(abs(t - s) > threshold for s in specials):
            final_ticks.append(t)
            final_labels.append(f"{t:.1f}") 
            
    return final_ticks, final_labels

# Oś X
xticks, xlabels = create_smart_ticks(
    xlim_lo, xlim_hi, 
    [mu_x_true, x_bar], 
    [r'$E[\bar{X}]$', r'$\bar{X}$']
)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=11)

# Oś Y
yticks, ylabels = create_smart_ticks(
    ylim_lo, ylim_hi, 
    [y_cv, y_bar], 
    [r'$\bar{Y}_{CV}$', r'$\bar{Y}_{MC}$']
)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=11)

ax.set_xlabel(r'Wypłata opcji arytmetycznej (Zmienna kontrolna)', fontsize=12)
ax.set_ylabel(r'Cena końcowa $S_T$ (Zmienna szacowana)', fontsize=12)

# Adnotacje
annotation_y_pos = ylim_lo + pad_y * 0.8
ax.annotate('', xy=(mu_x_true, annotation_y_pos), xytext=(x_bar, annotation_y_pos),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text((mu_x_true + x_bar)/2, annotation_y_pos + pad_y*0.5, r'błąd w $X$', ha='center', fontsize=9)

# Korekta
annotation_x_pos = xlim_lo + pad_x * 0.8
ax.annotate('', xy=(annotation_x_pos, y_cv), xytext=(annotation_x_pos, y_bar),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

text_y_korekta = min(y_cv, y_bar) - pad_y * 0.2
ax.text(annotation_x_pos, text_y_korekta, "korekta",
        ha='center', va='top', color='blue', fontsize=9)

ax.set_xlim(xlim_lo, xlim_hi)
ax.set_ylim(ylim_lo, ylim_hi)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Korelacja
rho = np.corrcoef(X, Y)[0, 1]
plt.text(xlim_hi - pad_x, ylim_lo + pad_y, f'Korelacja $\\rho \\approx {rho:.4f}$', ha='right', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.tight_layout()
plt.show()
