# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/als-recommender. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to prepare the data for use in the ALS recommender. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC In this notebook, we will make accessible purchase history data which will be used as the basis for the construction of a matrix factorization recommender.  The dataset we will use is the [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis), downloadable from the Kaggle website. We will make the data available through a set of queryable tables and then derive implied ratings from the data before proceeding to the next notebook.

# COMMAND ----------

# MAGIC %md ## Step 1: Data Preparation
# MAGIC
# MAGIC The data in the Instacart dataset should be [downloaded](https://www.kaggle.com/c/instacart-market-basket-analysis) and uploaded to cloud storage. The cloud storage location should then be [mounted](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) to the Databricks file system as shown here:</p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=240>
# MAGIC
# MAGIC **NOTE** The name of the mount point, file locations and database used is configurable within the *00_Intro & Config* notebook.
# MAGIC
# MAGIC The individual files that make up each entity in this dataset can then be presented as a queryable table as part of a database with a high-level schema as follows:</p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>
# MAGIC
# MAGIC We have automated this data preparation step for you in the notebook below and used a `/tmp/instacart_als` storage path throughout this accelerator in place of the mount path. 

# COMMAND ----------

# MAGIC %pip install -q ydata-profiling==4.0.0

# COMMAND ----------

# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# %run "./util/data-extract"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn
from pyspark.sql import window as w

from ydata_profiling import ProfileReport
# from ydata_profiling.visualisation.plot import timeseries_heatmap

# COMMAND ----------

# DBTITLE 1,Define Table Creation Helper Functions
def read_data(file_path, schema):
  df = (spark.read.csv(
            file_path,
            header=True,
            schema=schema
        )
    )
  return df

def write_data(df, table_name):
   _ = (df.write.format('delta')
        .mode('overwrite')
        .option('overwriteSchema','true')
        .saveAsTable(table_name)
       )


# COMMAND ----------

display(dbutils.fs.ls(f'{config["mount_point"]}/bronze/aisles'))

# COMMAND ----------

# DBTITLE 1,Load the Data To Tables
# orders data
# ---------------------------------------------------------
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])

orders = read_data(config['orders_path'], orders_schema)
write_data( orders, '{0}.orders'.format(config['schema']))
# ---------------------------------------------------------

# products
# ---------------------------------------------------------
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])

products = read_data( config['products_path'], products_schema)
write_data( products, '{0}.products'.format(config['schema']))
# ---------------------------------------------------------

# order products
# ---------------------------------------------------------
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])

order_products = read_data( config['order_products_path'], order_products_schema)
write_data( order_products, '{0}.order_products'.format(config['schema']))
# ---------------------------------------------------------

# departments
# ---------------------------------------------------------
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])

departments = read_data( config['departments_path'], departments_schema)
write_data( departments, '{0}.departments'.format(config['schema']))
# ---------------------------------------------------------

# aisles
# ---------------------------------------------------------
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])

aisles = read_data( config['aisles_path'], aisles_schema)
write_data( aisles, '{0}.aisles'.format(config['schema']))
# ---------------------------------------------------------

# COMMAND ----------

# DBTITLE 1,Present Tables in Database
# MAGIC %sql 
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1a: Data Exploration
# MAGIC
# MAGIC - Let's take a look at each of these datasets. 
# MAGIC - We'll use the popular [`ydata-profiling`](https://ydata-profiling.ydata.ai/docs/master/pages/getting_started/overview.html) library to generate easy-to-understand data profiles. 
# MAGIC - Remember, the association between these tables are as follows
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>
# MAGIC
# MAGIC For an effective matrix factorisation model we require a robust and informative **user-item** matrix, where each entry represents:
# MAGIC - A user's interaction with an item (e.g. a rating, a purchase)
# MAGIC - Ideally we have **explicit** interactions where the user has knowingly and willingly scored an item. This could include leaving a star rating, a review, etc. In the context of retail, a **purchase** of an item can be considered an explicit approval of that item
# MAGIC 	- However, in the context of grocery items, this is rare. The incentive to leave a rating for a $1 snack is low compared to leaving a review for a $2000 TV
# MAGIC - In scenarios where explicit ratings are rare, we instead have to build out proxy **implicit** ratings
# MAGIC - In retail, implicit ratings could be:
# MAGIC 	- Pageviews of a product
# MAGIC 	- Dwell time on the product details page
# MAGIC 	- How many times a product is viewed across the customer's browsing history
# MAGIC 	- Whether an item was added to the cart
# MAGIC 	- How frequently a product is reordered by a customer
# MAGIC 	- The number of times a product was searched for

# COMMAND ----------

def get_profile(df, timeseries_mode=False):
    df_profile = ProfileReport(df, minimal=True, 
                               title="Profiling Report", 
                               correlations={
                                "auto": {"calculate": False},
                                "pearson": {"calculate": True},
                                "spearman": {"calculate": True}
                               },
                               tsmode=timeseries_mode
                               # progress_bar=False, 
                               # infer_dtypes=False
                               )
    profile_html = df_profile.to_html()
    return profile_html

# COMMAND ----------

# MAGIC %md
# MAGIC ## aisles
# MAGIC
# MAGIC - This data represents the grocery aisle that an ordered product belongs to 
# MAGIC - We're dealing with 134 unique aisles with no missing values
# MAGIC - This is a useful signal for suggesting items in categories that users have an affinity for

# COMMAND ----------

df_aisles = spark.sql("SELECT * FROM aisles").cache()
display(df_aisles)
displayHTML(get_profile(df_aisles))

# COMMAND ----------

# MAGIC %md
# MAGIC ## departments
# MAGIC
# MAGIC - This data represents the grocery department that an ordered product belongs to 
# MAGIC - We're dealing with 21 unique departments with no missing values
# MAGIC - This is a useful signal for suggesting items in categories that users have an affinity for

# COMMAND ----------

df_departments = spark.sql("SELECT * FROM departments").cache()
display(df_departments)
displayHTML(get_profile(df_departments))

# COMMAND ----------

# MAGIC %md
# MAGIC ## order_products
# MAGIC
# MAGIC - This data reflects: 
# MAGIC   - which products make up an individual order
# MAGIC   - what order the product was added to the cart in
# MAGIC   - whether a product was a re-order (i.e. customer purchased it before)
# MAGIC - There are roughly 3.3M orders
# MAGIC - These are useful signals for:
# MAGIC   - Identifying products that are commonly ordered together
# MAGIC   - Identifying popular products for a customer (reflected by re-ordering a product)

# COMMAND ----------

df_order_products = spark.sql("SELECT * FROM order_products").cache()
display(df_order_products)
displayHTML(get_profile(df_order_products))

# COMMAND ----------

# MAGIC %md
# MAGIC ## orders
# MAGIC
# MAGIC - This data represents 3.4M orders and their metadata, feature engineered from their raw dataset
# MAGIC - This includes:
# MAGIC   - User ID per order
# MAGIC   - Which order occurence (e.g. 3rd order, 5th order) it is for the customer
# MAGIC   - Order day of week
# MAGIC   - Order hour of day
# MAGIC   - How long it has been, in days, since the customer's previous order
# MAGIC - These are useful signals for identifying:
# MAGIC   - Customer's lifetime order history
# MAGIC   - Customer's recency & frequency
# MAGIC   - Day and time orders are typically made
# MAGIC   - When applied to a SKU-level, it can help reveal patterns around product popularity and order frequency

# COMMAND ----------

df_orders = spark.sql("SELECT * FROM orders").cache()
display(df_orders)
displayHTML(get_profile(df_orders))

# COMMAND ----------

# MAGIC %md
# MAGIC ## products
# MAGIC
# MAGIC - This data represents stocked products
# MAGIC - We're dealing with 50K unique products with no missing values

# COMMAND ----------

df_products = spark.sql("SELECT * FROM products").cache()
display(df_products)
displayHTML(get_profile(df_products))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combined datasets
# MAGIC
# MAGIC Let's join the tables together to inspect how they relate to each other

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT o.*,
# MAGIC   op.* EXCEPT (order_id, product_id),
# MAGIC   p.* EXCEPT(aisle_id, department_id),
# MAGIC   a.*,
# MAGIC   d.*
# MAGIC FROM orders o
# MAGIC INNER JOIN order_products op  ON o.order_id = op.order_id
# MAGIC INNER JOIN products p         ON op.product_id = p.product_id
# MAGIC INNER JOIN aisles a           ON p.aisle_id = a.aisle_id
# MAGIC INNER JOIN departments d      ON p.department_id = d.department_id

# COMMAND ----------

# In a Databricks Python notebook, table results from a SQL language cell are automatically made available as a Python DataFrame assigned to the variable _sqldf

df_combined = _sqldf
displayHTML(get_profile(df_combined))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Step 2: Generate Ratings
# MAGIC
# MAGIC - The records that make up the Instacart dataset represent grocery purchases. 
# MAGIC - As would be expected in a grocery scenario, there are no explicit ratings provided in this dataset. 
# MAGIC   - Explicit ratings are typically found in scenarios where users are significantly invested (either monetarily or in terms of time or social standing) in the items they are purchasing or consuming.  
# MAGIC - When we are considering apples and bananas purchased to have around the house as a snack or to be dropped in a kid's lunch, most users are just not interested in providing 1 to 5 star ratings on those items.
# MAGIC
# MAGIC - We therefore need to examine the data for implied ratings (preferences).  
# MAGIC - In a grocery scenario where items are purchased for consumption, repeat purchases may provide a strong signal of preference. 
# MAGIC - [Douglas Oard and Jinmook Kim](https://terpconnect.umd.edu/~oard/pdf/aaai98.pdf) provide a nice discussion of the various ways we might derive implicit ratings in a variety of scenarios and it is certainly worth considering alternative ways of deriving an input metric.  
# MAGIC - However, for the sake of simplicity, we'll leverage the percentage of purchases involving a particular item as our implied rating:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DROP VIEW IF EXISTS user_product_purchases;
# MAGIC
# MAGIC -- CREATE VIEW user_product_purchases
# MAGIC -- AS
# MAGIC --   SELECT
# MAGIC --     monotonically_increasing_id() as row_id,
# MAGIC --     x.user_id,
# MAGIC --     x.product_id,
# MAGIC --     x.product_purchases / y.purchase_events as rating
# MAGIC --   FROM (  -- product purchases
# MAGIC --     SELECT
# MAGIC --       a.user_id,
# MAGIC --       b.product_id,
# MAGIC --       COUNT(*) as product_purchases
# MAGIC --     FROM orders a
# MAGIC --     INNER JOIN order_products b
# MAGIC --       ON a.order_id=b.order_id
# MAGIC --     INNER JOIN products c
# MAGIC --       ON b.product_id=c.product_id
# MAGIC --     GROUP BY a.user_id, b.product_id
# MAGIC --     ) x 
# MAGIC --   INNER JOIN ( -- purchase events
# MAGIC --     SELECT 
# MAGIC --       user_id, 
# MAGIC --       COUNT(DISTINCT order_id) as purchase_events 
# MAGIC --     FROM orders 
# MAGIC --     GROUP BY user_id
# MAGIC --     ) y
# MAGIC --     ON x.user_id=y.user_id
# MAGIC --     ;
# MAGIC     
# MAGIC -- SELECT *
# MAGIC -- FROM user_product_purchases;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS user_product_purchases;
# MAGIC
# MAGIC CREATE VIEW user_product_purchases
# MAGIC AS
# MAGIC WITH product_purchases AS (
# MAGIC   SELECT
# MAGIC     o.user_id,
# MAGIC     op.product_id,
# MAGIC     -- How many times this customer ordered this product
# MAGIC     COUNT(*) as product_purchases
# MAGIC   FROM orders o
# MAGIC   INNER JOIN order_products op ON o.order_id=op.order_id
# MAGIC   INNER JOIN products p ON op.product_id=p.product_id
# MAGIC   GROUP BY o.user_id, op.product_id
# MAGIC ),
# MAGIC purchase_events AS (
# MAGIC   SELECT user_id, 
# MAGIC     COUNT(DISTINCT order_id) as purchase_events 
# MAGIC   FROM orders 
# MAGIC   GROUP BY user_id
# MAGIC )
# MAGIC SELECT
# MAGIC   monotonically_increasing_id() as row_id,
# MAGIC   pp.user_id,
# MAGIC   pp.product_id,
# MAGIC   -- Proxy rating based on how often a specific customer purchases a specific product
# MAGIC   pp.product_purchases / pe.purchase_events as rating
# MAGIC FROM product_purchases as pp
# MAGIC INNER JOIN purchase_events as pe
# MAGIC   ON pp.user_id=pe.user_id;
# MAGIC

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
