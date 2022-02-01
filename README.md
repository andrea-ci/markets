# Log-Periodic Power Law Singularity (LPPLS) Model for bubble detection

In financial markets, bubbles are the result of a faster-than-exponential growth
which becomes unsustainable over time, thus forcing a significant correction in
price level.

LPPLS model comes as a combination of: mathematical and statistical physics of
phase transitions; behavorial finance, imitation and herding of traders that
creates positive feedback; economic theory of bubbles.

### Installation

```
pip install markets
```

### Quickstart

```
import markets
import pandas as pd
import matplotlib.pyplot as plt

# Load observations from an index of prices of cryptocurrencies.
time, price = markets.load_index_data()

# Init and fit the model (dates are converted to ordinal format).
model = markets.LPPLS()
time_ord = [pd.Timestamp.toordinal(t) for t in time]
model.fit(time_ord, price, persist = True)

# Compute price values estimated by model.
fitted = [model(t) for t in time_ord]

# Compare observations and fitted model.
fig, ax = plt.subplots(figsize = (14, 8))
ax.plot(time, price, color = 'black', linewidth = 1, label = 'price')
ax.plot(time, fitted, color = 'green', alpha = 0.5, label = 'LPPLS fit')
ax.grid(which = 'major', axis = 'both', linestyle='--')
ax.set_ylabel('Time (ordinal)')
ax.set_ylabel('Log-price')
ax.legend(loc = 'best')
plt.xticks(rotation = 70)
plt.savefig('fitted_model.png', dpi = 300)

```

### Estimation of confidence indicators

```
# Compute the indicators.
time_new, price_new, pos_conf, neg_conf = model.estimate_indicators(time_ord, price)
time_new_dt = [pd.Timestamp.fromordinal(t) for t in time_new.astype('int32')]

fig, (ax_1, ax_2) = plt.subplots(figsize = (14, 8), nrows = 2, ncols = 1, sharex = True)

# Plot confidence for positive bubbles.
ax_11 = ax_1.twinx()
ax_1.plot(time_new_dt, price_new, color = 'black', linewidth = 1)
ax_1.grid(which = 'major', axis = 'both', linestyle = '--')
ax_1.set_ylabel('Log-price')
ax_11.plot(time_new_dt, pos_conf, color = 'red', alpha = 0.5, label = 'Pos. confidence')
ax_11.set_ylabel('Confidence indicator')
ax_11.legend(loc = 'best')

# Plot confidence for negative bubbles.
ax_22 = ax_2.twinx()
ax_2.plot(time_new_dt, price_new, color = 'black', linewidth = 1)
ax_2.grid(which = 'major', axis = 'both', linestyle = '--')
ax_2.set_ylabel('Log-price')
ax_22.plot(time_new_dt, neg_conf, color = 'green', alpha = 0.5, label = 'Neg. confidence')
ax_22.set_ylabel('Confidence indicator')
ax_22.legend(loc = 'best')

plt.xticks(rotation = 70)
plt.savefig('confidence.png', dpi = 300)

```

### References

For details about the model see for example:
  - Sornette, Johansen & Bouchaud (1996), *"Stock market crashes, precursors
    and replicas"*, Journal de Physique I 6(1)
  - Sornette, Demos, Zhang, Cauwels, Filimonov & Zhang (2015), *"Real-time
  prediction and post-mortem analysis of the shanghai 2015 stock market
  bubble and crash"*, Swiss Finance Institute Research Paper (15-31)

For implementation details and estimation of confidence indicators see for example:
  - Jeremy (2020), *"Prediction of financial bubbles and backtesting of a
    trading strategy"*, Master Thesis at Imperial College London
