from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv() 

DATABASE_URL = os.getenv("DATABASE_URL")

# DATABASE_URL = "mysql+pymysql://root:12345678Dd!@localhost:3306/loanai"
# DATABASE_URL = "mysql+pymysql://root:Kavyansh_123@localhost:3306/loanai"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
