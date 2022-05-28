/*
    Extract data from raw table into a flat structure.
    
    In this case, we first check the latest etl and retrieve data related to that.
*/

WITH latest_etl AS (
    SELECT
        cb."ETL_ID"
    FROM {{ var('SF_SCHEMA') }}.{{ var('SF_TABLE') }} as cb
    ORDER BY cb."ETL_TIMESTAMP" DESC
    LIMIT 1
)
SELECT 
    cd."ETL_ID",
    cd."API_KEY",
    cd."EVENT_DATE",
    first_value(cd."EVENT_DATE") over (partition by cd."RAW_DATA":"session_id"::VARCHAR order by cd."RAW_DATA":"server_timestamp_epoch_ms"::INT ASC) as SESSION_DATE,
    cd."EVENT_TYPE",
    cd."RAW_DATA":"hashed_url"::VARCHAR as URL,
    cd."RAW_DATA":"product_action"::VARCHAR as PRODUCT_ACTION,
    REPLACE(LOWER(cd."RAW_DATA":"product_sku"::VARCHAR), ' ', '_') as SKU,
    cd."RAW_DATA":"server_timestamp_epoch_ms"::INT as EVENT_EPOCH_TIMESTAMP,
    cd."RAW_DATA":"session_id"::VARCHAR as SESSION_ID
FROM 
    {{ var('SF_SCHEMA') }}.{{ var('SF_TABLE') }} as cd
JOIN 
    latest_etl as le ON le.ETL_ID=cd.ETL_ID
ORDER BY SESSION_ID, EVENT_EPOCH_TIMESTAMP ASC
