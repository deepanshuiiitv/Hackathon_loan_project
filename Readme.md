install all requirements ---> "pip install -r requirements.txt"

present inside database.py --->
change DATABASE_URL = "mysql+pymysql://root:password@localhost:3306/databasename" 

run "uvicorn main:app --reload"