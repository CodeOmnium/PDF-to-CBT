from app import app, db
def initialize_database():
  with app.app_context():
    try:
      db.create_all()
      print("✅ Database initialized successfully!")
      print("Database location:", app.config.get('SQLALCHEMY_DATABASE_URI'))
    except Exception as e:
      print(f"❌ Database initialization failed: {e}")
      
if __name__ == "__main__":
  initialize_database()