#!/usr/bin/env python3
"""
Quick test to verify optimizations are working
Run this before starting the full tuner
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")
    try:
        from pupil_tuner.config import (
            TuningParams, TuningToggles, RuntimeConfig,
            save_config, load_config, PerformanceProfile
        )
        from pupil_tuner.app import (
            PupilTrackerTuner, AdaptiveFrameSkipper, DetectionValidator
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_save_load():
    """Test config save/load functionality"""
    print("\nTesting config save/load...")
    try:
        from pupil_tuner.config import (
            TuningParams, TuningToggles, RuntimeConfig,
            save_config, load_config
        )
        
        # Create test config
        params = TuningParams()
        params.threshold_value = 123  # Unique value for testing
        toggles = TuningToggles()
        runtime = RuntimeConfig()
        
        # Save
        test_path = Path("config/test_config.json")
        save_config(params, toggles, runtime, test_path)
        
        # Load
        loaded_params, loaded_toggles, loaded_runtime = load_config(test_path)
        
        # Verify
        if loaded_params.threshold_value == 123:
            print("✓ Config save/load working")
            test_path.unlink()  # Clean up
            return True
        else:
            print("✗ Config values don't match")
            return False
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_performance_profiles():
    """Test performance profiles"""
    print("\nTesting performance profiles...")
    try:
        from pupil_tuner.config import PerformanceProfile
        
        profiles = [
            ("High Speed", PerformanceProfile.high_speed),
            ("Balanced", PerformanceProfile.balanced),
            ("High Quality", PerformanceProfile.high_quality),
            ("Glasses Mode", PerformanceProfile.glasses_mode),
        ]
        
        for name, profile_func in profiles:
            params, toggles = profile_func()
            if params is None or toggles is None:
                print(f"✗ {name} profile failed")
                return False
        
        print("✓ All profiles working")
        return True
    except Exception as e:
        print(f"✗ Profile test failed: {e}")
        return False

def test_frame_skipper():
    """Test adaptive frame skipper"""
    print("\nTesting frame skipper...")
    try:
        from pupil_tuner.app import AdaptiveFrameSkipper
        
        skipper = AdaptiveFrameSkipper(target_fps=30.0)
        
        # Test normal speed (should not skip)
        should_skip = skipper.should_skip_frame(30.0)
        if should_skip:
            print("✗ Skipping when shouldn't")
            return False
        
        # Test slow speed (should skip)
        should_skip = skipper.should_skip_frame(15.0)
        if not should_skip:
            print("✗ Not skipping when should")
            return False
        
        print("✓ Frame skipper working")
        return True
    except Exception as e:
        print(f"✗ Frame skipper test failed: {e}")
        return False

def test_validator():
    """Test detection validator"""
    print("\nTesting detection validator...")
    try:
        from pupil_tuner.app import DetectionValidator
        
        validator = DetectionValidator(max_jump_px=50.0)
        
        # Test first detection (should accept)
        ellipse1 = ((100, 100), (50, 50), 0)
        result, is_valid = validator.validate(ellipse1, 0.8)
        if not is_valid:
            print("✗ Rejected first detection")
            return False
        
        # Test small movement (should accept)
        ellipse2 = ((105, 105), (50, 50), 0)
        result, is_valid = validator.validate(ellipse2, 0.8)
        if not is_valid:
            print("✗ Rejected valid detection")
            return False
        
        # Test large jump (should reject)
        ellipse3 = ((200, 200), (50, 50), 0)
        result, is_valid = validator.validate(ellipse3, 0.8)
        if is_valid:
            print("✗ Accepted invalid jump")
            return False
        
        print("✓ Validator working")
        return True
    except Exception as e:
        print(f"✗ Validator test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("OPTIMIZATION VERIFICATION TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Config Save/Load", test_config_save_load),
        ("Performance Profiles", test_performance_profiles),
        ("Frame Skipper", test_frame_skipper),
        ("Detection Validator", test_validator),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print("=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All optimizations verified! Ready to run.")
        print("\nNext steps:")
        print("1. Run: python live_tuning.py")
        print("2. Press 's' to save settings")
        print("3. Press 'l' to load settings")
        print("4. Try Shift+1/2/3/4 for profiles")
        return 0
    else:
        print("\n⚠ Some tests failed. Check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
