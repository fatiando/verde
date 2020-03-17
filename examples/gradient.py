"""
Gradient calculations
=====================



"""
import matplotlib.pyplot as plt
import numpy as np
import verde as vd


data = vd.datasets.CheckerBoard(region=(0, 5000, 0, 2500), w_east=5000,
                                w_north=2500).scatter(size=700, random_state=0)

spline = vd.Spline().fit((data.easting, data.northing), data.scalars)
east_deriv = vd.Gradient(spline, step=10, direction=(1, 0)).grid(spacing=50)
north_deriv = vd.Gradient(spline, step=10, direction=(0, 1)).grid(spacing=50)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

ax1.set_title("Original data")
tmp = ax1.scatter(data.easting, data.northing, c=data.scalars, cmap="RdBu_r")
plt.colorbar(tmp, ax=ax1, label="data unit")
ax1.set_ylabel("northing")

east_deriv.scalars.plot.pcolormesh(
    ax=ax2,
    cbar_kwargs=dict(label="data unit / m")
)
ax2.set_title("East derivative")
ax2.set_xlabel("")

north_deriv.scalars.plot.pcolormesh(
    ax=ax3,
    cbar_kwargs=dict(label="data unit / m")
)
ax3.set_title("North derivative")

fig.tight_layout()
plt.show()
