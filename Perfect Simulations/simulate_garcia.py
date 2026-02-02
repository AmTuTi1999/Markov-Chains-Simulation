from src.pipelines.locally_continuous_kernel_pipeline import run_garcia_simulation
def create_simple_skeleton():
    """
    Create a simple skeleton for testing.
    """
    alphabet = [0, 1]
    
    # Simple skeleton: contexts of length 0, 1, 2
    skeleton = {
        tuple(),  # Empty context
        (0,), (1,),  # Length 1
        (0, 0), (0, 1), (1, 0), (1, 1)  # Length 2
    }
    
    return skeleton, alphabet


if __name__ == "__main__":
    # Example usage
    skeleton, alphabet = create_simple_skeleton()
    
    args = {
        'skeleton': skeleton,
        'alphabet': alphabet,
        'alpha_sequence': {},  # Will be generated in the function
        'epsilon': 0.1,
        'alpha_params': [0.1, 0.15, 0.2, 0.25, 0.3],
    }
    
    results = run_garcia_simulation(
        window=(0, 10),
        args=args,
        max_blocks=100,
        max_block_size=50,
        num_validation_samples=1000,
        validate_with_simulation=True,
        run_coalescence_analysis=True,
    )
    
    print("\nExperiment completed successfully!")
