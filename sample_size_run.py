def main() -> None:
    """Calculate sample size based on user inputs for
        1. metric type: Boolean, Numeric, Ratio (case sensitive)
        2. power analysis parameters: alpha, power, mde
        3. inputs specific to metric type:
            * Boolean: probability
            * Numeric: variance
            * Ratio: mean and variance of numerator and denominator and their covariance
            
    Notes:
        1. default statistical power is used in this script all the time
        2. the calculator supports single metric per calculator for now
    """
    from sample_size_calculator import SampleSizeCalculator
    from scripts.input_utils import get_alpha
    from scripts.input_utils import get_metrics
    from scripts.input_utils import get_variants
    
    try:
        # get alpha for power analysis
        alpha = get_alpha()
        variants = get_variants()
        calculator = SampleSizeCalculator(alpha=alpha, variants=variants)
        
        metrics = get_metrics()
        calculator.register_metrics(metrics=metrics)
        
        sample_size = calculator.get_sample_size()
        print("\nSample size needed in each group: {:.3f}".format(sample_size))
    except Exception as e:
        print(f"Error cant calculate sample size due to \n{e}")


if __name__ == "__main__":
    main()