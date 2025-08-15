import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from datetime import datetime, timedelta
from pyspark.sql import functions as F
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
    """Fixture with the provided test data"""
    # Impressions
    impressions = [
        Row(dt="2025-08-01", ranking_id="r1", customer_id=1,
            impressions=[Row(item_id=101, is_order=True),
                         Row(item_id=102, is_order=False)])
    ]
    impressions_df = spark.createDataFrame(impressions)

    # Clicks
    clicks = [
        Row(dt="2025-07-31", customer_id=1, item_id=201,
            click_time=datetime(2025, 7, 31, 10, 0, 0))
    ]
    clicks_df = spark.createDataFrame(clicks)

    # Add-to-carts
    add_to_carts = [
        Row(dt="2025-07-30", customer_id=1, config_id=301,
            occurred_at=datetime(2025, 7, 30, 9, 0, 0))
    ]
    add_to_carts_df = spark.createDataFrame(add_to_carts)

    # Orders
    orders = [
        Row(order_date="2025-07-29", customer_id=1, config_id=401,
            occurred_at=datetime(2025, 7, 29, 8, 0, 0))
    ]
    orders_df = spark.createDataFrame(orders)
    
    return {
        "impressions_df": impressions_df,
        "clicks_df": clicks_df,
        "add_to_carts_df": add_to_carts_df,
        "orders_df": orders_df
    }
#unit testing
def test_configure_spark_session(spark):
    session = configure_spark_session()
    assert session.conf.get("spark.sql.shuffle.partitions") == "2000"
    assert session.conf.get("spark.sql.adaptive.enabled") == "true"

def test_process_actions(spark, test_data):
    """Test action processing with schema validation and counts"""
    all_actions = process_actions(
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    
    # Verify schema
    assert set(all_actions.columns) == {"customer_id", "item_id", "timestamp", "dt", "action_type"}
    
    # Verify counts (1 click + 1 ATC + 1 order)
    assert all_actions.count() == 3
    
    # Verify action types
    action_types = all_actions.select("action_type").distinct().collect()
    assert {row.action_type for row in action_types} == {1, 2, 3}

def test_get_recent_actions(spark, test_data):
    """Test recent action sequence generation"""
    all_actions = process_actions(
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    
    recent_actions = get_recent_actions(all_actions, "2025-08-01")
    
    # Should contain 1 customer with 3 actions
    assert recent_actions.count() == 1
    
    # Verify action sequence length and order (newest first)
    actions = recent_actions.first()["recent_actions"]
    assert len(actions) == 3
    assert actions[0]["item_id"] == 201  # Click (most recent)
    assert actions[1]["item_id"] == 301  # Add-to-cart
    assert actions[2]["item_id"] == 401  # Order (oldest)

def test_process_impressions(spark, test_data):
    """Test impression explosion logic"""
    processed = process_impressions(test_data["impressions_df"])
    
    # Should explode 1 impression row with 2 items into 2 rows
    assert processed.count() == 2
    
    # Verify schema
    assert set(processed.columns) == {"dt", "ranking_id", "customer_id", "item_id", "is_order"}
    
    # Verify items
    items = processed.select("item_id", "is_order").collect()
    assert {row.item_id for row in items} == {101, 102}
    assert [row.is_order for row in items] == [True, False]

def test_create_training_data(spark, test_data):
    """Test final training data assembly"""
    all_actions = process_actions(
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    recent_actions = get_recent_actions(all_actions, "2025-08-01")
    impressions_processed = process_impressions(test_data["impressions_df"])

    training_data = create_training_data(impressions_processed, recent_actions)

    # Should have 2 training examples (from 2 impression items)
    assert training_data.count() == 2

    # Verify schema
    expected_columns = {
        "dt", "customer_id", "ranking_id",
        "impression_item", "is_order",
        "action_items", "action_types"
    }
    assert set(training_data.columns) == expected_columns

    # Get the first training example
    first_example = training_data.filter(F.col("impression_item") == 101).first()
    
    # Verify the target label
    assert first_example["is_order"] == True
    
    # Verify action sequence (both padded length and real actions)
    assert len(first_example["action_items"]) == 1000  # Padded length
    real_actions = [x for x in first_example["action_items"] if x != 0]
    assert len(real_actions) == 3  # Actual number of actions
    assert real_actions == [201, 301, 401]  # Verify action items
    
    # Verify action types
    real_action_types = [t for t in first_example["action_types"] if t != 0]
    assert real_action_types == [1, 2, 3]  # Click, ATC, Order

def test_run_production_pipeline(spark, test_data):
    """Test end-to-end pipeline execution"""
    training_data, metrics = run_production_pipeline(
        test_data["impressions_df"],
        test_data["clicks_df"],
        test_data["add_to_carts_df"],
        test_data["orders_df"]
    )
    
    # Verify output counts
    assert training_data.count() == 2
    assert metrics["total_records"] == 2
    
    # Verify metrics structure
    assert "action_processing_sec" in metrics
    assert "recent_actions_sec" in metrics
    assert isinstance(metrics["training_data_creation_sec"], float)