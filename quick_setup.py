#!/usr/bin/env python3
"""
Replit-Specific Setup Script - Works with Nix environment
"""

import os
import sys
import secrets
import subprocess

def create_directories():
    """Create necessary directories"""
    directories = [
        'instance',
        'instance/uploads',
        'instance/images',
        'static/uploads',
        'static/images'
    ]

    print("📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}")

def create_env_file():
    """Create environment file"""
    print("📝 Creating .env file...")

    secret_key = secrets.token_urlsafe(32)

    env_content = f"""SECRET_KEY={secret_key}
DATABASE_URL=sqlite:///instance/pdf_test_tool.db
FLASK_ENV=production
FLASK_DEBUG=False
UPLOAD_FOLDER=instance/uploads
MAX_CONTENT_LENGTH=52428800
ENABLE_INSTRUCTION_FILTERING=true
DELETE_INSTRUCTION_IMAGES=true
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    print("✅ .env file created")

def check_replit_packages():
    """Check if Replit has auto-installed packages"""
    print("📦 Checking Replit package installation...")

    required_modules = [
        ('flask', 'Flask'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('cv2', 'OpenCV'),
        ('fitz', 'PyMuPDF'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy')
    ]

    missing_modules = []

    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing")
            missing_modules.append(name)

    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("💡 Replit should auto-install these when you run the app")
        return False

    print("✅ All required packages available")
    return True

def create_pyproject_toml():
    """Create pyproject.toml for Replit package management"""
    print("📝 Creating pyproject.toml for Replit...")

    pyproject_content = '''[tool.poetry]
name = "pdf-question-extraction-tool"
version = "1.0.0"
description = "PDF Question Extraction Tool for JEE Preparation"
author = "Your Name"

[tool.poetry.dependencies]
python = "^3.11"
Flask = "^2.3.3"
SQLAlchemy = "^2.0.23"
Flask-SQLAlchemy = "^3.0.5"
PyMuPDF = "^1.23.8"
opencv-python = "^4.8.1.78"
numpy = "^1.24.3"
Pillow = "^10.0.1"
pytesseract = "^0.3.10"
python-dateutil = "^2.8.2"
requests = "^2.31.0"
python-dotenv = "^1.0.0"
gunicorn = "^21.2.0"
psycopg2-binary = "^2.9.9"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
'''

    with open('pyproject.toml', 'w') as f:
        f.write(pyproject_content)

    print("✅ pyproject.toml created")

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic imports...")

    try:
        import os
        import sys
        import json
        import sqlite3
        print("✅ Standard library imports OK")
    except ImportError as e:
        print(f"❌ Standard library import failed: {e}")
        return False

    return True

def initialize_database_safe():
    """Safe database initialization for Replit"""
    print("🗄️  Initializing database...")

    # Create a minimal database initialization
    try:
        # Check if we can import basic modules
        import sqlite3

        # Create database file
        db_path = 'instance/pdf_test_tool.db'

        # Create basic tables using raw SQL (safer for Replit)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(200) NOT NULL,
                pdf_filename VARCHAR(255) NOT NULL,
                marking_scheme VARCHAR(20) NOT NULL DEFAULT 'jee_main',
                total_questions INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create questions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS question (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                question_number INTEGER NOT NULL,
                question_type VARCHAR(20) NOT NULL DEFAULT 'SCQ',
                image_path VARCHAR(500),
                page_number INTEGER,
                coordinates TEXT,
                ocr_text TEXT,
                confidence FLOAT DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES test (id)
            )
        ''')

        # Create test_session table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_session (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                student_name VARCHAR(100),
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                total_marks FLOAT DEFAULT 0.0,
                is_completed BOOLEAN DEFAULT FALSE,
                is_evaluated BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (test_id) REFERENCES test (id)
            )
        ''')

        # Create response table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                question_id INTEGER NOT NULL,
                user_answer TEXT,
                time_taken INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES test_session (id),
                FOREIGN KEY (question_id) REFERENCES question (id)
            )
        ''')

        # Create answer_key table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answer_key (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES question (id)
            )
        ''')

        conn.commit()
        conn.close()

        print("✅ Database tables created successfully")
        return True

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def create_replit_config():
    """Create .replit configuration"""
    print("📝 Creating .replit configuration...")

    replit_config = '''run = "python main.py"
modules = ["python-3.11"]

[nix]
channel = "stable-22_11"

[env]
PYTHONPATH = "$REPL_HOME"
FLASK_ENV = "production"

[gitHubImport]
requiredFiles = [".replit", "replit.nix", "main.py"]

[languages]

[languages.python3]
pattern = "**/*.py"

[languages.python3.languageServer]
start = "pylsp"

[deployment]
run = ["sh", "-c", "python main.py"]

[packager]
language = "python3"
ignoredPackages = ["unit_tests"]

[packager.features]
enabledForHosting = false
packageSearch = true
guessImports = true
'''

    with open('.replit', 'w') as f:
        f.write(replit_config)

    print("✅ .replit configuration created")

def main():
    """Main setup function for Replit"""
    print("🚀 PDF Question Extraction Tool - Replit Setup")
    print("=" * 50)

    # Step 1: Create directories
    create_directories()

    # Step 2: Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        create_env_file()
    else:
        print("✅ .env file already exists")

    # Step 3: Create Replit configuration
    create_replit_config()

    # Step 4: Create pyproject.toml for package management
    create_pyproject_toml()

    # Step 5: Test basic imports
    if not test_basic_imports():
        print("❌ Basic import test failed")
        return False

    # Step 6: Check if packages are available
    packages_available = check_replit_packages()

    # Step 7: Initialize database with safe method
    if not initialize_database_safe():
        print("❌ Database initialization failed")
        return False

    print("\n🎉 Replit setup completed!")

    if not packages_available:
        print("\n⚠️  Some packages may be missing.")
        print("📋 Next steps:")
        print("1. Click 'Run' button - Replit will auto-install packages")
        print("2. Wait for package installation to complete")
        print("3. The app should start automatically")
    else:
        print("\n📋 Ready to use:")
        print("1. Click 'Run' button or run: python main.py")
        print("2. Open web preview")
        print("3. Start uploading PDFs!")

    return True

if __name__ == "__main__":
    success = main()

    if not success:
        print("\n💡 Manual steps if setup fails:")
        print("1. Click 'Run' button in Replit")
        print("2. Replit will auto-install packages")
        print("3. App will start automatically")
        print("4. Database tables will be created on first run")

    sys.exit(0 if success else 1)