import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# Przygotowanie danych
np.random.seed(42)
N = 50
rho = 0.85
sigma = 1.0
mu_x_true = 2.0
mu_y_true = 3.0

# Generowanie danych
cov = [[sigma**2, rho*sigma*sigma], [rho*sigma*sigma, sigma**2]]
data = np.random.multivariate_normal([mu_x_true + 0.6, mu_y_true + 0.6], cov, N) 

X = data[:, 0]
Y = data[:, 1]

# Obliczanie statystyk
x_bar = np.mean(X)
y_bar = np.mean(Y)
cov_xy = np.cov(X, Y)[0, 1]
var_x = np.var(X, ddof=1)
b_hat = cov_xy / var_x

y_cv = y_bar - b_hat * (x_bar - mu_x_true)

# Rysowanie
fig, ax = plt.subplots(figsize=(8, 6))

# Wykres rozrzutu
ax.scatter(X, Y, facecolors='blue', edgecolors='gray', alpha=0.5, s=30)

# Linia regresji
x_vals = np.array([mu_x_true - 0.8, x_bar + 0.8])
y_vals = y_bar + b_hat * (x_vals - x_bar)
ax.plot(x_vals, y_vals, color='black', linewidth=1.5)

# Kluczowe punkty
ax.plot(x_bar, y_bar, 'o', color='black', markersize=7)
ax.plot(mu_x_true, y_cv, 'o', color='black', markersize=7)

# Linie rzutujące
ymin_plot = min(Y) - 0.5
xmin_plot = min(X) - 0.5

# Rzutowanie punktów
ax.plot([x_bar, x_bar], [ymin_plot, y_bar], 'k:', linewidth=0.8)     
ax.plot([xmin_plot, x_bar], [y_bar, y_bar], 'k:', linewidth=0.8)     

ax.plot([mu_x_true, mu_x_true], [ymin_plot, y_cv], 'k:', linewidth=0.8) 

ax.plot([xmin_plot, mu_x_true], [y_cv, y_cv], 'k:', linewidth=0.8)

# Dostosowanie osi
ax.set_xticks([mu_x_true, x_bar])
ax.set_xticklabels([r'$E[X]$', r'$\bar{X}$'], fontsize=14)

ax.set_yticks([y_cv, y_bar])
ax.set_yticklabels([r'$\bar{Y}(\hat{b}_n)$', r'$\bar{Y}$'], fontsize=14)

# Ograniczenie widoku
ax.set_xlim(xmin_plot, max(X)+0.5)
ax.set_ylim(ymin_plot, max(Y)+0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adnotacje
annotation_y = ymin_plot + 0.2
ax.annotate('', xy=(mu_x_true, annotation_y), xytext=(x_bar, annotation_y),
            arrowprops=dict(arrowstyle='->', color='black'))
ax.text((mu_x_true + x_bar)/2, annotation_y + 0.1, r'błąd w $X$', ha='center', fontsize=10)

# Opis nachylenia
mid_x = (x_bar + mu_x_true) / 2
mid_y = (y_bar + y_cv) / 2
ax.text(mid_x - 0.4, mid_y - 0.6, r'nachylenie $\hat{b}_n$', fontsize=12, rotation=25)

plt.tight_layout()
plt.show()
