/*
    Map flat rows into arrays of user-product interaction, to be ready for training 
    a next event prediction model.
*/


SELECT 
    se."SESSION_ID",
    se."API_KEY",
    se."SESSION_DATE",
    ARRAY_AGG(se."SKU") WITHIN GROUP (ORDER BY se."EVENT_EPOCH_TIMESTAMP" ASC) AS INTERACTIONS
FROM 
    {{ ref('shopping_events_exploded') }} as se
WHERE 
    se."EVENT_TYPE"='event_product' 
    AND se."PRODUCT_ACTION" IN ('detail', 'add', 'purchase')
GROUP BY
    se."SESSION_ID", se."API_KEY", se."SESSION_DATE"
ORDER BY se."SESSION_DATE" ASC