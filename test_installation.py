import kiteconnect
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import ta
import logging

print("✅ All packages imported successfully!")
print(f"KiteConnect version: {kiteconnect.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# Test basic functionality
try:
    # Test pandas
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print("✅ Pandas DataFrame created successfully")
    
    # Test numpy
    arr = np.array([1, 2, 3, 4, 5])
    print("✅ NumPy array created successfully")
    
    # Test ta (technical analysis)
    print("✅ Technical analysis library imported successfully")
    
    print("🎉 All tests passed! Installation is working correctly.")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
