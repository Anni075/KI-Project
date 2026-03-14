from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: list[str]

    def extend(self, name: str, *extra: str) -> "FeatureSet":
        """Return a new FeatureSet with additional features appended."""
        return FeatureSet(name=name, features=self.features + list(extra))


_IRRADIANCE_WEATHER = [
    "ghi_cloudy_sky",
    "clear_sky_index",
    "temperature_2m",
    "cloud_cover_low",
]
_TEMPORAL = ["interval_cos_shifted", "month_cos_shifted", "doy_cos_shifted"]
_LAG = ["Solarproduktion_lag_96"]

BASE = FeatureSet(
    name="BASE",
    features=_IRRADIANCE_WEATHER,
)

BASE_WITH_TIME = BASE.extend("BASE_WITH_TIME", *_TEMPORAL)

BASE_WITH_TIME_AND_LAG = BASE_WITH_TIME.extend("BASE_WITH_TIME_AND_LAG", *_LAG)

FEATURE_SETS: dict[str, FeatureSet] = {
    fs.name: fs for fs in [BASE, BASE_WITH_TIME, BASE_WITH_TIME_AND_LAG]
}
