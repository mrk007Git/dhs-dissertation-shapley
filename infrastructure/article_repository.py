import os
import pandas as pd
import sqlalchemy
from urllib.parse import quote_plus
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for progress bar

# Load environment variables
load_dotenv()

# Database connection settings
DB_SERVER = os.getenv('DB_SERVER', 'your-server.database.windows.net')
DB_DATABASE = os.getenv('DB_DATABASE', 'coromfieldguideportal-qa')
DB_USERNAME = os.getenv('DB_USERNAME', 'your-username')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your-password')
DB_DRIVER = os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')

def get_db_connection():
    """Create and return a database connection using SQLAlchemy."""
    try:
        # URL encode the password to handle special characters
        encoded_password = quote_plus(DB_PASSWORD)
        encoded_username = quote_plus(DB_USERNAME)
        encoded_driver = quote_plus(DB_DRIVER)
        
        connection_string = f"mssql+pyodbc://{encoded_username}:{encoded_password}@{DB_SERVER}/{DB_DATABASE}?driver={encoded_driver}&Encrypt=yes&TrustServerCertificate=no"
        
        print(f"Attempting to connect to: {DB_SERVER}/{DB_DATABASE}")
        engine = sqlalchemy.create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return None

def load_article_from_db(pmc_id, limit=1):
    """Load articles from Azure SQL Database."""
    try:
        engine = get_db_connection()
        if engine is None:
            raise Exception("Failed to create database engine")
        
        query = f"""
        SELECT TOP ({limit}) 
            [Content],
            [PmcId]
        FROM [dhs].[DissertationArticles]
        WHERE [PmcId] = '{pmc_id}'
        """
        
        print(f"Loading articles from database (limit: {limit})...")
        df = pd.read_sql(query, engine)
        
        # Rename Content to content for compatibility
        df = df.rename(columns={'Content': 'content'})
        
        print(f"Loaded {len(df)} articles from database.")
        return df
        
    except Exception as e:
        print(f"Error loading articles from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


# Load and print the articles for immediate use
if __name__ == "__main__":
    print("Testing database connection...")
    
    # Try loading from database first
    try:
        df_db = load_article_from_db('PMC1726388', limit=1)  # Test with small limit first
        print(f"Database loading successful: {len(df_db)} articles.")
        print("Database columns:", df_db.columns.tolist())
        print("\nSample data from database:")
        print(df_db[['Title']].head())
    except Exception as e:
        print(f"Database loading failed: {e}")
        df_db = None
    
    # Fallback to file loading
    if df_db is None or len(df_db) == 0:
        print("\nDatabase loading failed...")
    else:
        df = df_db