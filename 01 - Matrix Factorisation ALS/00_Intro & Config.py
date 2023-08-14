# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration Parameters
# MAGIC
# MAGIC This Notebook is used to manage your configuration parameters and keep them consistent as we run the subsequent Notebooks in these labs
# MAGIC
# MAGIC This only needs to be edited once

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
# Use hive_metastore if Unity Catalog is unavailable for this workspace
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
# MAGIC SELECT CURRENT_CATALOG() AS current_catalog, CURRENT_SCHEMA() AS current_schema;

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
config['model_name'] = 'als'

# COMMAND ----------

# DBTITLE 1,Set MLflow experiment
import mlflow

experiment_name = f"/Users/{username}/als-recommender"

# mlflow.set_experiment(experiment_name) sets the active experiment for all runs 
# within the current notebook to experiment_name. It creates the experiment if it 
# does not exist. This means that all logged runs within the notebook after 
# calling mlflow.set_experiment() will be associated with the specified experiment. 
# You can then view the results for this experiment in the MLflow Experiment UI
mlflow.set_experiment(experiment_name)

experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
experiment_url = f"/#mlflow/experiments/{experiment_id}"

print(f"MLflow experiment name set to: {experiment_name}")
displayHTML(f'<a href="{experiment_url}" target="_blank">Link to Experiment UI</a>')

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
