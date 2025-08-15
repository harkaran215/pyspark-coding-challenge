# PySpark Coding Challenge — Training Input Pipeline

## Overview
This project prepares training data for a transformer-based recommendation model from raw impressions and user actions (clicks, add-to-cart, previous orders).

The output contains one row per **impression item** with:
- `impression_item_id`
- `is_order`
- The last 1000 **historical actions** for that customer before the impression date (`action_items` and `action_types` arrays).

## Output Schema
| Column              | Type          | Description |
|---------------------|--------------|-------------|
| dt                  | string       | Impression day (partition key) |
| customer_id         | long         | Customer id |
| ranking_id          | string       | Carousel ranking id |
| impression_item_id  | long         | Item id in impression |
| is_order            | int          | 1 if ordered, else 0 |
| action_items             | array<int>   | Last 1000 item_ids, newest first, 0 padded |
| action_types        | array<int>   | Matching action types: 1=click, 2=add_to_cart, 3=order, 0=pad |

## Pipeline Steps
1. **Normalize actions**: unify clicks, ATC, orders into a single DataFrame.
2. **Filter by date**: for each impression day, keep actions from the last 365 days, strictly before `dt`.
3. **Rank & truncate**: assign row numbers per customer, keep top 1000 most recent.
4. **Aggregate**: collect ordered lists of item_ids and action_types, pad to fixed length.
5. **Join with impressions**: explode impressions to one row per item and attach actions.

## Performance Notes
- Filtering by 365-day window before ranking reduces shuffle size.
- Window partitioned by `customer_id` ensures per-user ordering.
- Output written in Parquet partitioned by `dt` for efficient downstream GPU training.

## Running in Databricks
```python
%run /path/to/pipeline
