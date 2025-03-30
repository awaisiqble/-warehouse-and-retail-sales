 
# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY app.py /app/
COPY requirements.txt /app/
COPY Cleaned_Warehouse_and_Retail_Sales.csv /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
