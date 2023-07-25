from typing import List

import numpy as np
import numpy.typing as npt
from statsmodels.stats.multitest import multipletests

from metrics import BaseMetric

DEFAULT_REPLICATION: int = 400
DEFAULT_EPSILON: float = 0.01
DEFAULT_MAX_RECURSION: int = 20


class MultipleTestingMixin:
    metrics: List[BaseMetric]
    alpha: float
    power: float
    variants: int

    def get_multiple_sample_size(
        self,
        lower: float,
        upper: float,
        random_state: np.random.RandomState,
        depth: int = 0,
        replication: int = DEFAULT_REPLICATION,
        epsilon: float = DEFAULT_EPSILON,
        max_recursion_depth: int = DEFAULT_MAX_RECURSION,
    ) -> int:
        """This method finds minimum required sample size per cohort that generates
        average power higher than required

        Args:
            lower (float): lower bound of sample size range
            upper (float): upper bound of sample size range
            random_state (np.random.RandomState):
            depth (int, optional): number of recursions. Defaults to 0.
            replication (int, optional): number of Monte Carlo simulations to calculate empirical power. Defaults to DEFAULT_REPLICATION.
            epsilon (float, optional): absolute difference between our estimate for power and desired power
                needed before we will return. Defaults to DEFAULT_EPSILON.
            max_recursion_depth (int, optional): how many recursive calls can be made before the
                search is abandoned. Defaults to DEFAULT_MAX_RECURSION.

        Returns:
            int: minimum required sample size per cohort
        """
        if depth >= max_recursion_depth:
            raise RecursionError(
                f"Couldn't find a sample size that satisfies the power you requested: {self.power}"
            )

        candidate = int(np.sqrt(lower * upper))
        expected_power = self._expected_average_power(
            candidate, random_state, replication
        )
        
        if np.isclose(self.power, expected_power, atol=epsilon):
            return candidate
        elif lower == upper:
            raise RecursionError(f"Couldn't find a sample size that satisfies the power you requested: {self.power}")

        if expected_power > self.power:
            return self.get_multiple_sample_size(lower, candidate, random_state, depth + 1)
        else:
            return self.get_multiple_sample_size(candidate, upper, random_state, depth + 1)
        
        
    def _expected_average_power(
        self,
        sample_size: int,
        random_state: np.random.RandomState,
        replication: int = DEFAULT_REPLICATION,
    ) -> float:
        """This method calculates expected average power of multiple testings. For each possible number of true null
        hypothesis, we simulate each metric/treatment variant's test statistics and calculate their p-values,
        then calculate expected average power = number of True rejection/ true alternative hypotheses

        Args:
            sample_size (int): determines the variance/ degrees of freedom of the distribution to sample test statistics from
            replication (int, optional): number of times we repeat the simulation process. Defaults to DEFAULT_REPLICATION.

        Returns:
            float: expected average power value
        """

        true_alt_count = 0.0
        true_discovery_count = 0.0
        
        # a metric for each test to conduct
        metrics = self.metrics * (self.variants - 1)
        
        def fdr_bh(a: npt.NDArray[np.float_]) -> npt.NDArray[np.bool_]:
            """False discovery rate Benjamini/Hochberg """
            rejected: npt.NDArray[np.bool_] = multipletests(a, alpha=self.alpha, method="fdr_bh")[0]
            return rejected
            
        for num_true_alt in range(1, len(metrics) + 1):
            true_alt = np.array([random_state.permutation(len(metrics)) < num_true_alt for _ in range(replication)]).T
            p_values = []
            for i, m in enumerate(metrics):
                p_values.append(m.generate_p_values(true_alt[i], sample_size, random_state))
            
            rejected = np.apply_along_axis(fdr_bh, 0, np.array(p_values)) #type: ignore[no-untyped-call]
            
            true_discoveries = rejected & true_alt
            
            true_discovery_count += true_discoveries.sum()
            true_alt_count += true_alt.sum()
            
        avg_power = true_discovery_count / true_alt_count
        
        return avg_power