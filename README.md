# Workshop: Recommendation Engines on Databricks

![banner](https://raw.githubusercontent.com/vinoaj/databricks-resources/main/assets/img/banner-databricks-rec-ws.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
<!-- [![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks) -->

## Prerequisites

### Schema (database) you can write to

- Ensure you have access to a schema (database) that you can create tables on. It's preferable for a standalone schema for this workshop.

### Cluster

- You will require a cluster with the following specifications:
  - Access mode: Assigned
  - Single user: Your user ID
  - Databricks runtime version: `13.2 ML (Scala 2.12, Spark 3.4.0)`
  - Worker type: `XXXX`
    - Min workers: `1`
    - Max workers: `8`
    - Use spot/preemptible: `yes`
  - Driver type: `XXXX`

## Resources

- [Solution Accelerators: Recommendation Engines for Personalisation](https://www.databricks.com/solutions/accelerators/recommendation-engines)
- Matrix Factorisation / Alternating Least Squares: [Blog](https://www.databricks.com/blog/2023/01/06/products-we-think-you-might-generating-personalized-recommendations.html) | [Notebooks](https://notebooks.databricks.com/notebooks/RCG/als-recommender/index.html#als-recommender_1.html) | [GitHub](https://github.com/databricks-industry-solutions/als-recommender)

--

## Business Problem

<List of the business use case the solution accelerator address>

## Scope

<How we expect the user to use this content>

___
<john.doe@databricks.com>

___

IMAGE TO REFERENCE ARCHITECTURE

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | <https://github.com/yaml/pyyaml>                      |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries.

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps:

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format.
3. Execute the multi-step-job to see how the pipeline runs.
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.

## Project support

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.
