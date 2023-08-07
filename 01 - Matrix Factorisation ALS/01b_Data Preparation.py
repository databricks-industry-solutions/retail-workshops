# Databricks notebook source
# MAGIC %md 
# MAGIC # Data Exploration
# MAGIC
# MAGIC
# MAGIC In the previous Notebook we:
# MAGIC - Loaded the data into Delta tables
# MAGIC - Explored the data
# MAGIC - Had a classroom discussion on what implicit ratings we could derive from the data
# MAGIC
# MAGIC In this Notebook, we will:
# MAGIC - Perform feature engineering to generate the implicit ratings

# COMMAND ----------

# MAGIC %pip install -q ydata-profiling

# COMMAND ----------

# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# MAGIC %md
# MAGIC # Classroom Exercise
# MAGIC
# MAGIC **Question**: Given this data, what sort of implicit ratings could we derive from them?

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
# MAGIC     -- Number of unique orders per customer
# MAGIC     COUNT(DISTINCT order_id) as purchase_events 
# MAGIC   FROM orders 
# MAGIC   GROUP BY user_id
# MAGIC )
# MAGIC SELECT
# MAGIC   MONOTONICALLY_INCREASING_ID() as row_id,
# MAGIC   pp.user_id,
# MAGIC   pp.product_id,
# MAGIC   -- Proxy rating based on how often a specific customer purchases/reorders a specific product
# MAGIC   pp.product_purchases / pe.purchase_events as rating
# MAGIC FROM product_purchases as pp
# MAGIC INNER JOIN purchase_events as pe
# MAGIC   ON pp.user_id=pe.user_id;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM user_product_purchases;

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
