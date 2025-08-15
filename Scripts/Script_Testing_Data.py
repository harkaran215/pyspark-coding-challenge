import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from datetime import datetime, timedelta
import time
from Script_Training_Input_Pipeline import ( 
    configure_spark_session,
    process_actions,
    get_recent_actions,
    process_impressions,
    create_training_data,
    run_production_pipeline
)

# Fixture for Spark session
@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-pyspark") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield spark
    spark.stop()

# Fixture for test data
@pytest.fixture
def test_data(spark):
    # Sample clicks data
    clicks_data = [
        ("2023-01-01", 1, 101, "2023-01-01 10:00:00"),
        ("2023-01-02", 1, 102, "2023-01-02 11:00:00"),
        ("2023-01-01", 2, 201, "2023-01-01 09:00:00")
    ]
    clicks_schema = ["dt", "customer_id", "item_id", "click_time"]
    clicks_df = spark.createDataFrame(clicks_data, clicks_schema)
    
    # Sample add_to_carts data
    add_to_carts_data = [
        ("2023-01-01", 1, 101, 1, "2023-01-01 12:00:00"),
        ("2023-01-03", 2, 201, 1, "2023-01-03 10:00:00")
    ]
    add_to_carts_schema = ["dt", "customer_id", "config_id", "simple_id", "occurred_at"]
    add_to_carts_df = spark.createDataFrame(add_to_carts_data, add_to_carts_schema)
    
    # Sample orders data
    orders_data = [
        ("2023-01-05", 1, 101, 1, "2023-01-05 15:00:00"),
        ("2023-01-04", 2, 202, 1, "2023-01-04 14:00:00")
    ]
    orders_schema = ["order_date", "customer_id", "config_id", "simple_id", "occurred_at"]
    orders_df = spark.createDataFrame(orders_data, orders_schema)
    
    # Sample impressions data
    impressions_data = [
        (
            "2023-01-10", "ranking1", 1,
            [{"item_id": 101, "is_order": True}, {"item_id": 103, "is_order": False}]
        ),
        (
            "2023-01-10", "ranking2", 2,
            [{"item_id": 201, "is_order": False}]
        )
    ]
    impressions_schema = StructType([
        StructField("dt", StringType()),
        StructField("ranking_id", StringType()),
        StructField("customer_id", IntegerType()),
        StructField("impressions", ArrayType(StructType([
            StructField("item_id", IntegerType()),
            StructField("is_order", BooleanType())
        ])))
    ])
    impressions_df = spark.createDataFrame(impressions_data, impressions_schema)
    
    return {
        "clicks_df": clicks_df,
        "add_to_carts_df": add_to_carts_df,
        "orders_df": orders_df,
        "impressions_df": impressions_df
    }

# Unit tests
def test_configure_spark_session(spark):
    session = configure_spark_session()
    assert session.conf.get("spark.sql.shuffle.partitions") == "2000"
    assert session.conf.get("spark.sql.adaptive.enabled") == "true"

def test_process_actions(spark, test_data):
    all_actions = process_actions(
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    
    # Verify schema
    assert set(all_actions.columns) == {"customer_id", "item_id", "timestamp", "dt", "action_type"}
    
    # Verify counts (3 clicks + 2 add_to_carts + 2 orders = 7 total)
    assert all_actions.count() == 7
    
    # Verify action types
    action_types = all_actions.select("action_type").distinct().collect()
    assert {row.action_type for row in action_types} == {1, 2, 3}

def test_get_recent_actions(spark, test_data):
    all_actions = process_actions(
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    
    # Use a test date that includes all sample data
    recent_actions = get_recent_actions(all_actions, "2023-01-20")
    assert recent_actions.count() == 2  # 2 customers in test data
    
    # Verify customer 1 has all their actions
    cust1_actions = recent_actions.filter("customer_id = 1").first()
    assert len(cust1_actions["recent_actions"]) == 4  # 2 clicks + 1 cart + 1 order

def test_process_impressions(spark, test_data):
    processed = process_impressions(test_data["impressions_df"])
    
    # Verify explosion worked
    assert processed.count() == 3  # 2 items for customer1 + 1 for customer2
    
    # Verify schema
    assert set(processed.columns) == {"dt", "ranking_id", "customer_id", "item_id", "is_order"}
    
    # Verify customer 1 has 2 impressions
    cust1_impressions = processed.filter("customer_id = 1").collect()
    assert len(cust1_impressions) == 2
    assert {row.item_id for row in cust1_impressions} == {101, 103}

def test_create_training_data(spark, test_data):
    all_actions = process_actions(
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    recent_actions = get_recent_actions(all_actions, "2023-01-10")
    impressions_processed = process_impressions(test_data["impressions_df"])
    
    training_data = create_training_data(impressions_processed, recent_actions)
    
    # Verify schema
    expected_columns = {
        "dt", "customer_id", "ranking_id", 
        "impression_item", "is_order", 
        "action_items", "action_types"
    }
    assert set(training_data.columns) == expected_columns
    
    # Verify actual actions (before padding)
    cust1_data = training_data.filter("customer_id = 1").first()
    real_actions = [x for x in cust1_data["action_items"] if x != 0]
    assert len(real_actions) == 4  # Actual actions for customer 1

# Integration test
def test_run_production_pipeline(spark, test_data):
    training_data, metrics = run_production_pipeline(
        test_data["impressions_df"],
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    
    # Verify output structure
    assert training_data.count() == 3  # 3 impressions in test data
    assert isinstance(metrics, dict)
    assert metrics["total_records"] == 3

# Edge case tests
def test_empty_input(spark):
    # Create empty DataFrames with proper schema
    empty_clicks = spark.createDataFrame([], "dt: string, customer_id: int, item_id: int, click_time: string")
    empty_add_to_carts = spark.createDataFrame([], "dt: string, customer_id: int, config_id: int, simple_id: int, occurred_at: string")
    empty_orders = spark.createDataFrame([], "order_date: string, customer_id: int, config_id: int, simple_id: int, occurred_at: string")
    empty_impressions = spark.createDataFrame([], "dt: string, ranking_id: string, customer_id: int, impressions: array<struct<item_id:int,is_order:boolean>>")
    
    # Test pipeline with empty inputs
    training_data, metrics = run_production_pipeline(
        empty_impressions,
        empty_clicks,
        empty_add_to_carts,
        empty_orders
    )
    
    assert training_data.count() == 0
    assert metrics["total_records"] == 0

def test_duplicate_actions(spark):
    # Test data with duplicate actions
    clicks_data = [
        ("2023-01-01", 1, 101, "2023-01-01 10:00:00"),
        ("2023-01-01", 1, 101, "2023-01-01 10:05:00")  # Duplicate click
    ]
    clicks_df = spark.createDataFrame(clicks_data, ["dt", "customer_id", "item_id", "click_time"])
    
    impressions_data = [
        ("2023-01-02", "ranking1", 1, [{"item_id": 101, "is_order": False}])
    ]
    impressions_schema = StructType([
        StructField("dt", StringType()),
        StructField("ranking_id", StringType()),
        StructField("customer_id", IntegerType()),
        StructField("impressions", ArrayType(StructType([
            StructField("item_id", IntegerType()),
            StructField("is_order", BooleanType())
        ])))
    ])
    impressions_df = spark.createDataFrame(impressions_data, impressions_schema)
    
    all_actions = process_actions(clicks_df, spark.createDataFrame([], "dt: string, customer_id: int, config_id: int, simple_id: int, occurred_at: string"), spark.createDataFrame([], "order_date: string, customer_id: int, config_id: int, simple_id: int, occurred_at: string"))
    recent_actions = get_recent_actions(all_actions, "2023-01-02")
    
    # Verify duplicates are preserved
    actions = recent_actions.filter("customer_id = 1").first()["recent_actions"]
    assert len(actions) == 2
    assert actions[0]["item_id"] == 101
    assert actions[1]["item_id"] == 101

# Performance test (optional)
@pytest.mark.skip("Run manually for performance testing")
def test_performance(spark, test_data):
    start_time = time.time()
    _, metrics = run_production_pipeline(
        test_data["impressions_df"],
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    duration = time.time() - start_time
    
    print(f"\nPipeline performance metrics: {metrics}")
    print(f"Total execution time: {duration:.2f} seconds")
    
    # Add your performance assertions here
    assert duration < 10.0  # Example threshold