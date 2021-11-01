import os

SECRET_KEY = 'asdfghjkl'
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')