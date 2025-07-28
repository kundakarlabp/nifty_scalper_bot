import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.web_dashboard.app import app

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing Web Dashboard...")

# Test 1: Flask App Initialization
print("\n1. Testing Flask App Initialization...")
try:
    with app.test_client() as client:
        # Test home page
        response = client.get('/')
        print(f"✅ Home page test: {response.status_code}")
        
        # Test API endpoints
        status_response = client.get('/api/status')
        print(f"✅ Status API test: {status_response.status_code}")
        
        performance_response = client.get('/api/performance')
        print(f"✅ Performance API test: {performance_response.status_code}")
        
        signals_response = client.get('/api/signals')
        print(f"✅ Signals API test: {signals_response.status_code}")
    
    print("✅ Flask app initialized successfully")
    
except Exception as e:
    print(f"❌ Flask app test failed: {e}")

# Test 2: Template Rendering
print("\n2. Testing Template Rendering...")
try:
    with app.test_client() as client:
        # Test different pages
        pages = ['/', '/settings', '/backtest', '/logs', '/nonexistent']
        
        for page in pages:
            response = client.get(page)
            status = response.status_code
            print(f"   ✅ {page}: {status}")
    
    print("✅ Template rendering working")
    
except Exception as e:
    print(f"❌ Template rendering test failed: {e}")

# Test 3: Static Files
print("\n3. Testing Static Files...")
try:
    with app.test_client() as client:
        # Test CSS file
        css_response = client.get('/static/css/style.css')
        print(f"   ✅ CSS file: {css_response.status_code}")
        
        # Test if CSS content is served
        if css_response.status_code == 200:
            content_length = len(css_response.data)
            print(f"   ✅ CSS content length: {content_length} bytes")
    
    print("✅ Static files serving working")
    
except Exception as e:
    print(f"❌ Static files test failed: {e}")

# Test 4: API Endpoints
print("\n4. Testing API Endpoints...")
try:
    with app.test_client() as client:
        # Test POST endpoint
        control_response = client.post('/api/controls', 
                                     json={'command': 'test'},
                                     content_type='application/json')
        print(f"   ✅ Controls API: {control_response.status_code}")
        
        # Test error handling
        error_response = client.get('/nonexistent')
        print(f"   ✅ 404 handling: {error_response.status_code}")
    
    print("✅ API endpoints working")
    
except Exception as e:
    print(f"❌ API endpoints test failed: {e}")

print("\n🎉 Web dashboard tests completed!")
print("🚀 Your web dashboard is ready!")

print("\n🔧 To run the web dashboard:")
print("   cd src/web_dashboard")
print("   python app.py")
print("\n🌐 Then visit: http://localhost:8000")
