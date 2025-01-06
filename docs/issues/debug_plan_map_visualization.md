# Debugging Plan: Map Visualization Issue
Date: 2024-02-13

## Current Issue
- Error: "The truth value of a Series is ambiguous" during map creation
- Occurs when checking neighbor property existence in stats_df
- Multiple properties affected

## Investigation Steps

1. Data Structure Analysis
   - [ ] Log and verify index types of both DataFrames
   - [ ] Check for any type mismatches between indices
   - [ ] Verify data integrity in both DataFrames
   - [ ] Examine sample values from both indices

2. Index Membership Check
   - [ ] Test alternative index membership methods
   - [ ] Verify behavior with sample data
   - [ ] Add debug logging for index comparison
   - [ ] Consider using explicit type conversion

3. Data Flow Analysis
   - [ ] Trace data from loading to visualization
   - [ ] Verify data transformations
   - [ ] Check for any unintended type conversions
   - [ ] Validate DataFrame operations

4. Error Handling Enhancement
   - [ ] Add more detailed error logging
   - [ ] Implement graceful fallbacks
   - [ ] Add data validation checks
   - [ ] Improve error messages

## Implementation Plan

1. Add Debug Logging
```python
# Add detailed logging
logger.info(f"Processing neighbor property {idx}")
logger.info(f"Index type: {type(idx)}")
logger.info(f"Stats df index: {stats_df.index.dtype}")
logger.info(f"Value in stats_df: {idx in stats_df.index}")
```

2. Test Index Membership
```python
# Test different methods
method1 = idx in stats_df.index
method2 = stats_df.index.isin([idx])
method3 = stats_df.index.get_loc(idx)
```

3. Data Validation
```python
# Add validation checks
if not isinstance(idx, (int, str)):
    logger.warning(f"Unexpected index type: {type(idx)}")
    continue
```

4. Error Recovery
```python
# Implement fallback behavior
try:
    # Primary method
    if not stats_df.index.isin([idx]).any():
        continue
except Exception as e:
    # Fallback method
    if str(idx) not in stats_df.index.astype(str):
        continue
```

## Success Criteria
- [ ] No ambiguous truth value errors
- [ ] All properties correctly displayed on map
- [ ] Clean error handling for edge cases
- [ ] Consistent data type handling

## Progress Tracking
- Status: 🔄 In Progress
- Current Step: Initial Investigation
- Next Action: Implement debug logging 