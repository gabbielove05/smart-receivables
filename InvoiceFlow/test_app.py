#!/usr/bin/env python3
"""
InvoiceFlow Application Test
Simple test script to verify all components are working correctly.
"""

import sys
import os
import importlib

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing Python imports...")
    
    required_modules = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly.express',
        'sklearn',
        'openai',
        'faker',
        'requests'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_custom_modules():
    """Test that custom application modules can be imported."""
    print("\n🔧 Testing custom modules...")
    
    custom_modules = [
        'app',
        'chatbot',
        'dashboards',
        'smart_actions',
        'simple_email_system',
        'integrations',
        'utils',
        'ml_models',
        'data_quality'
    ]
    
    failed_imports = []
    
    for module in custom_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_data_files():
    """Test that required data files exist."""
    print("\n📄 Testing data files...")
    
    required_files = [
        'sample_invoices.csv',
        'sample_payments.csv'
    ]
    
    missing_files = []
    
    for filename in required_files:
        if os.path.exists(filename):
            # Check file size
            size = os.path.getsize(filename)
            print(f"  ✅ {filename} ({size:,} bytes)")
        else:
            print(f"  ❌ {filename} - File not found")
            missing_files.append(filename)
    
    return len(missing_files) == 0

def test_email_system():
    """Test that the email system can be initialized."""
    print("\n📧 Testing email system...")
    
    try:
        from simple_email_system import simple_email_system
        print("  ✅ Email system imported successfully")
        
        # Test basic functionality
        config = simple_email_system._get_smtp_config()
        print(f"  ✅ SMTP config accessible")
        
        return True
    except Exception as e:
        print(f"  ❌ Email system error: {e}")
        return False

def test_environment():
    """Test environment setup."""
    print("\n🌍 Testing environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"  📍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("  ⚠️  Python 3.8+ recommended")
    else:
        print("  ✅ Python version compatible")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"  📂 Current directory: {current_dir}")
    
    # Check for key files
    key_files = ['app.py', 'pyproject.toml', 'run_app.sh']
    for filename in key_files:
        if os.path.exists(filename):
            print(f"  ✅ {filename} found")
        else:
            print(f"  ❌ {filename} missing")
    
    return True

def run_all_tests():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("🏦 InvoiceFlow Application Test Suite")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Python Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Data Files", test_data_files),
        ("Email System", test_email_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! InvoiceFlow is ready to run.")
        print("\nTo start the application:")
        print("  ./run_app.sh")
        print("  or")
        print("  streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTry running:")
        print("  pip install -e .")
        print("  python3 generate_sample_data.py")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
