# Plan: clear_sky_index Erklärung verbessern

## File
`notebooks/05_linear_regression.ipynb` – cell `8d6dd9845cc198ba`

## Change
Callout-Box unterhalb der Feature-Tabelle einfügen:

```
> **Was ist `clear_sky_index`?**
> `GHI_clear` ist die Strahlung, die bei wolkenlosem Himmel ankäme — berechnet aus
> Sonnenstand + Atmosphäre, ohne Wolken.
> Das Verhältnis `GHI / GHI_clear` ist eine **reine Bewölkungsmessung**:
> Ein Wert von 0.7 bedeutet immer „70 % der maximal möglichen Strahlung kommen an" —
> egal ob es 8 Uhr oder 12 Uhr ist.
> `ghi_cloudy_sky` trägt die absolute Energie; `clear_sky_index` trägt das
> Bewölkungssignal, entkoppelt vom Sonnenstand.
```
