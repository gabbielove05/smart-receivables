import sys
import os

# Add the InvoiceFlow directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'InvoiceFlow'))

# Import and run the main function
from app import main

if __name__ == "__main__":
    main()


