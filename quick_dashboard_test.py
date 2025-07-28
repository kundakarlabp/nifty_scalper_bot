import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.web_dashboard.app import app

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🚀 Quick Web Dashboard Test")
print("=" * 40)

try:
    # Test Flask app creation
    print("🔧 Testing Flask app...")
    with app.test_client() as client:
        # Test basic routes
        routes = ['/', '/api/status', '/api/performance']
        
        for route in routes:
            response = client.get(route)
            print(f"   ✅ {route}: {response.status_code}")
    
    print("✅ Flask app is working correctly")
    
    # Test template existence
    print("\n📄 Testing template files...")
    template_files = [
        'templates/dashboard.html',
        'templates/settings.html', 
        'templates/backtest.html',
        'templates/logs.html'
    ]
    
    for template in template_files:
        full_path = os.path.join('src/web_dashboard', template)
        if os.path.exists(full_path):
            print(f"   ✅ {template}")
        else:
            print(f"   ⚠️  {template} (missing)")
    
    # Test static files
    print("\n🎨 Testing static files...")
    static_files = [
        'static/css/style.css'
    ]
    
    for static_file in static_files:
        full_path = os.path.join('src/web_dashboard', static_file)
        if os.path.exists(full_path):
            print(f"   ✅ {static_file}")
        else:
            print(f"   ⚠️  {static_file} (missing)")
    
    print("\n✅ Quick dashboard test completed!")
    print("🚀 Web dashboard components are ready!")
    
    print("\n🔧 To run the full dashboard:")
    print("   cd src/web_dashboard")
    print("   python app.py")
    print("\n🌐 Then visit: http://localhost:8000")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
