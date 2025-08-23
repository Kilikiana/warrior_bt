#!/usr/bin/env python3
"""
Simple test to verify BullFlagDetector modification works
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add the current directory to the path to ensure we're using local files
sys.path.insert(0, '/Users/claytonsmacbookpro/Projects/warrior_bt')

from tech_analysis.patterns.bull_flag_pattern import BullFlagDetector

def test_known_flagpole():
    """Test BullFlagDetector with known flagpole"""
    print("Testing BullFlagDetector with known flagpole...")
    
    # Create sample OHLC data
    timestamps = [
        datetime(2025, 8, 13, 6, 44),  # Alert time
        datetime(2025, 8, 13, 6, 45),  # After alert
        datetime(2025, 8, 13, 6, 46),
        datetime(2025, 8, 13, 6, 47),
        datetime(2025, 8, 13, 6, 48),
    ]
    
    data = {
        'open': [2.40, 2.45, 2.44, 2.43, 2.42],
        'high': [2.47, 2.46, 2.45, 2.44, 2.43],
        'low': [2.40, 2.44, 2.43, 2.42, 2.41],
        'close': [2.47, 2.45, 2.44, 2.43, 2.42],
        'volume': [6000, 3000, 2000, 1500, 1000]
    }
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Test BullFlagDetector
    detector = BullFlagDetector()
    
    print("\n1. Testing WITHOUT known flagpole (original mode):")
    signal1 = detector.detect_bull_flag(df, "TEST")
    print(f"   Stage: {signal1.stage}, Validation: {signal1.validation}")
    
    print("\n2. Testing WITH known flagpole (new mode):")
    alert_time = datetime(2025, 8, 13, 6, 44)
    alert_price = 2.47
    
    signal2 = detector.detect_bull_flag(
        df, 
        "TEST",
        known_flagpole_high=alert_price,
        known_flagpole_time=alert_time
    )
    print(f"   Stage: {signal2.stage}, Validation: {signal2.validation}")
    print(f"   Flagpole high: {signal2.flagpole_high}")
    print(f"   Flagpole start: {signal2.flagpole_start}")
    
    if signal2.validation != "no_flagpole":
        print("✅ SUCCESS: Known flagpole mode is working!")
    else:
        print("❌ FAILED: Known flagpole mode not working")

if __name__ == "__main__":
    test_known_flagpole()