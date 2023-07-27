# Databricks notebook source
# MAGIC %md The purpose of this notebook is to introduce the ALS recommender solution accelerator and to provide access to configuration information for the notebooks supporting it.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC Recommender systems are becoming increasing important as companies seek better ways to select products to present to end users. In this solution accelerator, we will explore a form of collaborative filter referred to as a matrix factorization.  
# MAGIC
# MAGIC Matrix factorization works by assembling a set of ratings for various products made by a set of users.  The large user x products matrix is decomposed into smaller user and product submatrices associated with some developer-specified number of latent factors.  In many ways, a matrix factorization is a dimension reduction technique but one where missing values in the original matrix are allowed.
# MAGIC
# MAGIC When examining ratings for a large number of user and product combinations, most users will engage with a very smaller percentage of products.  This causes us to have a user x products matrix that is highly sparse. When we decompose this matrix into the submatrices, the two can be combined to *recreate* the original matrix in a manner that provides ratings estimates for all products, including those a user has not yet engaged.  This ability to fill-in the missing ratings forms the basis for recommending new products to a user.
# MAGIC
# MAGIC Matrix factorization recommenders are frequently used in scenarios where we wish to suggest new and repeat purchase items to a user.  *People like you also bought ...*, *Products we think you'll like ...*, and *Based on your purchase history ...* styled recommendations are frequently delivered through this type of recommender.
# MAGIC
# MAGIC The challenge in developing a matrix factorization recommender is the large amount of computational horsepower required to calculate the submatrices.  Alternating Least Squares (ALS) is one approach that decomposes the process into a series of incremental steps that can be implemented in a distributed manner. In this solution accelerator, we will train and deploy an ALS-based matrix factorization recommender using the ALS capabilities in Apache Spark to demonstrate how this is done.

# COMMAND ----------

username = (dbutils.notebook.entry_point.getDbutils()
            .notebook().getContext()
            .userName().get())

print(f"Your username: {username}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Configuration Settings
# MAGIC
# MAGIC - [View available catalogs](/explore/data)

# COMMAND ----------

# Catalog and schema (database) where you'll be writing your data
CATALOG = "vinny_vijeyakumaar"
CATALOG_FALLBACK = "hive_metastore"
SCHEMA = "recommendation_workshop"

# Temporary location to store the downloaded Instacart data
MOUNT_POINT = "/tmp/instacart"

# COMMAND ----------

# DBTITLE 1,Instantiate Config Variable 
if 'config' not in locals().keys():
  config = {}

config["catalog"] = CATALOG
config["schema"] = SCHEMA

# COMMAND ----------

from pyspark.sql.utils import AnalysisException

target_catalog = config["catalog"]

try:
    # Set the catalog to the desired catalog
    spark.sql(f"USE CATALOG {target_catalog}")
except AnalysisException as e:
    # If the catalog doesn't exist, fallback to the Hive metastore
    if "not found" in str(e):
        print(f"Target catalog not found. Falling back on {CATALOG_FALLBACK}")
        spark.sql(f"USE CATALOG {CATALOG_FALLBACK}")
        config["catalog"] = CATALOG_FALLBACK
    else:
        raise e

print(f'Using Catalog: {config["catalog"]}')

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config['schema']}")
spark.sql(f"USE SCHEMA {config['schema']}")

print(f'Using Schema: {config["schema"]}')

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CURRENT SCHEMA;

# COMMAND ----------

# DBTITLE 1,Identify Database
# config['database'] = 'als'

# COMMAND ----------

# DBTITLE 1,Create & Set Current Database
# create database if not exists
# _ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current database context
# _ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# MAGIC %md Here we use a temporary path in DBFS for illustration purposes to reduce external dependencies. We recommend that you use a cloud storage path or [mount point](https://docs.databricks.com/dbfs/mounts.html) to save data for production workloads. 

# COMMAND ----------

# DBTITLE 1,Define paths to data files
config['mount_point'] = MOUNT_POINT

config['products_path']         = f"{MOUNT_POINT}/bronze/products"
config['orders_path']           = f"{MOUNT_POINT}/bronze/orders"
config['order_products_path']   = f"{MOUNT_POINT}/bronze/order_products"
config['aisles_path']           = f"{MOUNT_POINT}/bronze/aisles"
config['departments_path']      = f"{MOUNT_POINT}/bronze/departments"

# COMMAND ----------

# DBTITLE 1,Model Info
config['model name'] = 'als'

# COMMAND ----------

# DBTITLE 1,Set mlflow experiment
import mlflow
mlflow.set_experiment('/Users/{}/als-recommender'.format(username))

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
