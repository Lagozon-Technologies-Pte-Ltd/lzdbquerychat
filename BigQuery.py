#pip install --upgrade google-cloud-bigquery
import os
from google.cloud import bigquery

# Set the environment variable correctly
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Cloud_service.json'

# Initialize the BigQuery client
client = bigquery.Client()

# Define the query
sql_query = """
WITH MonthlyRetail AS (
    SELECT
        FORMAT_DATE('%b-%y', b.`Date`) AS `Month`,
        ROUND(AVG(b.`Retail Volume`), 1) AS `Average_Retail_Sales`,
        MIN(b.`Date`) AS MinDate  -- Add this to retain the earliest date for sorting
    FROM `DS_sales_data.billing_data` b
    WHERE b.`Date` BETWEEN DATE('2024-07-01') AND DATE('2024-09-30')
    GROUP BY `Month`
)
SELECT
    `Month`,
    `Average_Retail_Sales`
FROM MonthlyRetail
ORDER BY MinDate;  -- Now ordering by the stored minimum date

"""
query_job = client.query(sql_query)
results = query_job.result()

# Print the results
for row in results:
    print(row)





