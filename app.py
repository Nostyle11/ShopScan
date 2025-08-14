import os
import logging
from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Custom logging filter to prevent base64 image data from being logged
class Base64Filter(logging.Filter):
    def filter(self, record):
        # Filter out log messages containing long base64 strings
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if len(record.msg) > 1000 and ('/' in record.msg or '+' in record.msg):
                return False
        return True

# Set up logging - use INFO level but filter sensitive data
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Add base64 filter to root logger
logging.getLogger().addFilter(Base64Filter())

# Silence verbose third-party loggers
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1) # needed for url_for to generate with https

# configure the database, relative to the app instance folder
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///pricecompare.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# initialize the app with the extension, flask-sqlalchemy >= 3.0.x
db.init_app(app)

# Add template global for current year
@app.template_global()
def now():
    return datetime

with app.app_context():
    # Make sure to import the models here or their tables won't be created
    import models  # noqa: F401
    import routes  # noqa: F401
    
    db.create_all()
    
    # Initialize some default sources if they don't exist
    from models import Source
    if not Source.query.first():
        sources = [
            Source(name='eBay', url='https://www.ebay.com', logo='ebay', active=True),
            Source(name='AliExpress', url='https://www.aliexpress.com', logo='aliexpress', active=True),
            Source(name='Jumia', url='https://www.jumia.com.gh', logo='jumia', active=True),
            Source(name='Alibaba', url='https://www.alibaba.com', logo='alibaba', active=True),
            Source(name='Jiji', url='https://jiji.com.gh', logo='jiji', active=True),
            Source(name='Tonaton', url='https://tonaton.com', logo='tonaton', active=True),
            Source(name='Kikuu', url='https://www.kikuu.com.gh', logo='kikuu', active=True),
        ]
        
        for source in sources:
            db.session.add(source)
        
        db.session.commit()
        logging.info("Initialized default sources")
