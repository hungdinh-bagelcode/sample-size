import json
from pathlib import Path
from typing import Any, Dict, List

from metrics import BaseMetric, BooleanMetrics, NumericMetric, RatioMetric

import numpy as np
from jsonschema import validate
from multiple_testing import MultipleTestingMixin

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_VARIANTS = 2
RANDOM_STATE = np.random.RandomState(20)
STATE = RANDOM_STATE.get_state()

schema_file_path = Path(Path(__file__).parent, "metrics_schema.json")

with open(str(schema_file_path), "r") as schema_file:
    METRICS_SCHEMA = json.load(schema_file)


class SampleSizeCalculator(MultipleTestingMixin):
    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        variants: int = DEFAULT_VARIANTS,
        power: float = DEFAULT_POWER,
    ):
        self.alpha = alpha
        self.power = power
        self.variants: int = variants
        self.metrics: List[BaseMetric] = []

    def _get_single_sample_size(self, metric: BaseMetric, alpha: float) -> float:
        effect_size = metric.mde / float(np.sqrt(metric.variance))
        power_analysis = (
            metric.power_analysis_instance
        )  # Call statistical power calculation class of metric
        # print(metric.alternative)
        sample_size = int(
            power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=self.power,
                ratio=1,
                alternative=metric.alternative,
            )
        )
        return sample_size

    def get_sample_size(self) -> float:
        if len(self.metrics) * (self.variants - 1) < 2:
            return self._get_single_sample_size(self.metrics[0], self.alpha)

        num_tests = len(self.metrics) * (self.variants - 1)
        lower = min(
            [
                self._get_single_sample_size(metric, self.alpha)
                for metric in self.metrics
            ]
        )
        upper = min(
            [
                self._get_single_sample_size(metric, self.alpha / num_tests)
                for metric in self.metrics
            ]
        )

        RANDOM_STATE.set_state(STATE)
        return self.get_multiple_sample_size(lower, upper, RANDOM_STATE)

    def register_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        METRIC_REGISTER_MAP = {
            "boolean": BooleanMetrics,
            "numeric": NumericMetric,
            "ratio": RatioMetric,
        }

        validate(instance=metrics, schema=METRICS_SCHEMA)

        for metric in metrics:
            metric_class = METRIC_REGISTER_MAP[metric["metric_type"]]
            registered_metric = metric_class(**metric["metric_metadata"])
            self.metrics.append(registered_metric)
