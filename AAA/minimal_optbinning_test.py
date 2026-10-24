import numpy as np
import sys

print("\nTesting environment:")
print(f"Python version: {sys.version}")

try:
    print("\nImporting OptimalBinning...")
    from optbinning import OptimalBinning
    print("Import successful")
    
    print("\nCreating minimal test data...")
    # Create very small dataset
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0, 0, 1, 1, 1])
    
    print("\nInitializing OptimalBinning...")
    optb = OptimalBinning(
        name='test',
        dtype='numerical',
        solver='cp',
        min_n_bins=2,
        max_n_bins=3,
        # Minimal parameters
        min_bin_size=0.2,
        max_bin_size=0.8
    )
    
    print("\nFitting data...")
    optb.fit(x, y)
    
    print("\nGetting results...")
    splits = optb.splits
    if splits is not None:
        print(f"Found splits at: {splits}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("\nTrying to get package info:")
    try:
        import pkg_resources
        pkg = pkg_resources.working_set.by_key['optbinning']
        print(f"OptimalBinning version: {pkg.version}")
        print(f"Installation location: {pkg.location}")
    except:
        print("Could not get package info")
    
    print("\nTry reinstalling with:")
    print("pip uninstall optbinning")
    print("pip install --no-cache-dir optbinning")
