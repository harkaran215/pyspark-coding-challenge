from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, TimestampType, ArrayType, StringType
from pyspark.sql import SparkSession
import time 

def configure_spark_session():
    """Configure Spark session for production workloads"""
    spark = SparkSession.builder \
        .config("spark.sql.shuffle.partitions", "2000") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Set log level if needed
    spark.sparkContext.setLogLevel("WARN")
    return spark

def process_actions(clicks_df, add_to_carts_df, orders_df):
    """
    Process raw action data with production optimizations:
    - Early filtering of invalid records
    - Column pruning
    - Predicate pushdown
    """
    # Schema validation
    clicks_df = clicks_df.select(
        F.col("customer_id").cast(IntegerType()).alias("customer_id"),
        F.col("item_id").cast(IntegerType()).alias("item_id"),
        F.col("click_time").cast(TimestampType()).alias("timestamp"),
        F.col("dt").cast(StringType()).alias("dt")
    ).filter("customer_id IS NOT NULL AND item_id IS NOT NULL")
    
    add_to_carts_df = add_to_carts_df.select(
        F.col("customer_id").cast(IntegerType()).alias("customer_id"),
        F.col("config_id").cast(IntegerType()).alias("item_id"),
        F.col("occurred_at").cast(TimestampType()).alias("timestamp"),
        F.col("dt").cast(StringType()).alias("dt")
    ).filter("customer_id IS NOT NULL AND config_id IS NOT NULL")
    
    orders_df = orders_df.select(
        F.col("customer_id").cast(IntegerType()).alias("customer_id"),
        F.col("config_id").cast(IntegerType()).alias("item_id"),
        F.col("occurred_at").cast(TimestampType()).alias("timestamp"),
        F.to_date(F.col("order_date")).cast(StringType()).alias("dt")
    ).filter("customer_id IS NOT NULL AND config_id IS NOT NULL")
    
    # Union with production optimizations
    return clicks_df.select(
        "customer_id", "item_id", "timestamp", "dt"
    ).withColumn("action_type", F.lit(1)) \
    .union(
        add_to_carts_df.select(
            "customer_id", "item_id", "timestamp", "dt"
        ).withColumn("action_type", F.lit(2))
    ).union(
        orders_df.select(
            "customer_id", "item_id", "timestamp", "dt"
        ).withColumn("action_type", F.lit(3))
    ).cache()  # Cache as this will be used multiple times

def get_recent_actions(all_actions, impression_dt=None, max_actions=1000):
    if impression_dt is None:
        from datetime import datetime
        impression_dt = datetime.now().strftime("%Y-%m-%d")

    # 1. Filter to last year first (minimize data early)
    one_year_ago = F.date_sub(F.lit(impression_dt).cast("date"), 365)
    recent_actions = all_actions.filter(F.col("timestamp") >= one_year_ago)
    
    # 2. Repartition by customer_id to avoid skew in window function
    recent_actions = recent_actions.repartition("customer_id")
    
    # 3. Rank and keep top 1000 per customer
    window = Window.partitionBy("customer_id").orderBy(F.col("timestamp").desc())
    ranked = recent_actions.withColumn("rank", F.row_number().over(window))
    top_actions = ranked.filter(F.col("rank") <= max_actions)
    
    # 4. Group and repartition for balanced output
    return top_actions.groupBy("customer_id").agg(
        F.collect_list(F.struct("item_id", "action_type")).alias("recent_actions")
    ).repartition("customer_id")   # Optimize for downstream joins

def process_impressions(impressions_df):
    """Production-grade impression processing with explosion"""
    # Schema enforcement
    impression_schema = ArrayType(StructType([
        StructField("item_id", IntegerType()),
        StructField("is_order", BooleanType())
    ]))
    
    processed = impressions_df.select(
        F.col("dt").cast(StringType()),
        F.col("ranking_id").cast(StringType()),
        F.col("customer_id").cast(IntegerType()),
        F.col("impressions").cast(impression_schema).alias("impressions")
    ).filter("size(impressions) > 0")  # Filter empty impressions
    
    # Explode with memory management
    return processed.select(
        "dt", "ranking_id", "customer_id",
        F.explode("impressions").alias("impression")
    ).select(
        "dt", "ranking_id", "customer_id",
        F.col("impression.item_id").alias("item_id"),
        F.col("impression.is_order").alias("is_order")
    ).repartition(200, "customer_id", "dt")  # Balanced partitioning

def create_training_data(impressions_processed, recent_actions):
    """Production-optimized training data creation"""
    # Broadcast join optimization for small action datasets
    if recent_actions.count() < 1000000:
        recent_actions = F.broadcast(recent_actions)
    
    joined = impressions_processed.join(
        recent_actions,
        on="customer_id",
        how="left"
    )
    
    # Efficient null handling and padding
    return joined.withColumn(
        "action_sequence",
        F.when(
            F.size(F.col("recent_actions")) < 1000,
            F.concat(
                F.col("recent_actions"),
                F.array_repeat(
                    F.struct(
                        F.lit(0).alias("item_id"),
                        F.lit(0).alias("action_type")
                    ),
                    1000 - F.size(F.col("recent_actions"))
                )
            )
        ).otherwise(F.col("recent_actions"))
    ).select(
        "dt", "customer_id", "ranking_id",
        F.col("item_id").alias("impression_item"),
        "is_order",
        F.col("action_sequence.item_id").alias("action_items"),
        F.col("action_sequence.action_type").alias("action_types")
    ).persist()  # Persist as this is the final output

def run_production_pipeline(impressions_df, clicks_df, add_to_carts_df, orders_df):
    """End-to-end production pipeline with monitoring"""
    spark = configure_spark_session()
    
    # Action processing with monitoring
    action_start = time.time()
    all_actions = process_actions(clicks_df, add_to_carts_df, orders_df)
    action_time = time.time() - action_start
    
    # Recent actions with resource monitoring
    recent_start = time.time()
    recent_actions = get_recent_actions(all_actions)
    recent_time = time.time() - recent_start
    
    # Impression processing
    impression_start = time.time()
    impressions_processed = process_impressions(impressions_df)
    impression_time = time.time() - impression_start
    
    # Final join and output
    training_start = time.time()
    training_data = create_training_data(impressions_processed, recent_actions)
    training_time = time.time() - training_start
    
    # Log performance metrics
    spark.sparkContext._jsc.sc().uiWebUrl().get()
    metrics = {
        "action_processing_sec": action_time,
        "recent_actions_sec": recent_time,
        "impression_processing_sec": impression_time,
        "training_data_creation_sec": training_time,
        "total_records": training_data.count()
    }
    
    return training_data, metrics