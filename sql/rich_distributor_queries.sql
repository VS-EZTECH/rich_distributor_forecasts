DROP TABLE `eztech-442521.rich.metadata_table`;


CREATE TABLE `eztech-442521.rich.metadata_table` (
  ingestion_date STRING,
  file_name STRING,
  destination_table STRING,
  status STRING,
  date_from_file_name DATETIME,
  day_from_file_name STRING,
  no_of_records INTEGER
)
PARTITION BY DATE(date_from_file_name);

DROP TABLE `eztech-442521.rich.landing_table_distributor_sales`;


CREATE TABLE `eztech-442521.rich.landing_table_distributor_sales` (
 date DATE,
 document STRING,
 unit STRING,
 town STRING,
 customer_ID STRING,
 customer_Tax_ID STRING,
 SKU STRING,
 `Sales Rep` STRING,
 Promo_discount_perc FLOAT64,
 Promo_discount_RUR FLOAT64,
 Document_for_discount STRING,
 SKU_ID STRING,
 quantity FLOAT64,
 price FLOAT64,
 amount FLOAT64,
 filename STRING,
 filename_end_date DATE,
 weekday STRING
)
PARTITION BY filename_end_date
OPTIONS (
 description = "Landing table for distributor sales data, partitioned by filename_end_date"
);

CREATE TABLE `eztech-442521.rich.landing_table_distributor_stocks` (
  date DATE,
  warehouse STRING,
  SKU_ID STRING,
  price FLOAT64,
  stock FLOAT64,
  total FLOAT64,
  filename STRING,
  filename_end_date DATE,
  weekday STRING
)
PARTITION BY filename_end_date
OPTIONS (
  description = "Landing table for distributor stock data, partitioned by filename_end_date"
);

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE `eztech-442521.rich.main_table_distributor_sales`
PARTITION BY date
AS SELECT * 
FROM `eztech-442521.rich.landing_table_distributor_sales` 
WHERE FORMAT_DATE('%Y-%m', filename_end_date) != '2025-03';



CREATE OR REPLACE PROCEDURE `eztech-442521.rich.daily_main_table_distributor_sales_update`()
BEGIN
  DECLARE max_main_date DATE;
  DECLARE max_landing_date DATE;
  DECLARE max_actual_date DATE;
  
  -- Get latest dates from both tables using filename_end_date
  SET max_main_date = (SELECT MAX(filename_end_date) FROM `eztech-442521.rich.main_table_distributor_sales`);
  SET max_landing_date = (SELECT MAX(filename_end_date) FROM `eztech-442521.rich.landing_table_distributor_sales`);
  SET max_actual_date = (SELECT MAX(date) FROM `eztech-442521.rich.landing_table_distributor_sales` WHERE filename_end_date = max_landing_date);
  
  -- Only proceed if landing has newer data
  IF max_landing_date > max_main_date THEN
    -- Delete only overlapping partitions with the latest filename_end_date
    DELETE FROM `eztech-442521.rich.main_table_distributor_sales`
    WHERE date IN (
      SELECT DISTINCT date 
      FROM `eztech-442521.rich.landing_table_distributor_sales`
      WHERE filename_end_date = max_landing_date
    );
    
    -- Insert only new data from landing table with latest filename_end_date
    -- Apply TRIM to all string columns
    INSERT INTO `eztech-442521.rich.main_table_distributor_sales`
    SELECT 
      date,
      TRIM(document) AS document,
      TRIM(unit) AS unit,
      TRIM(town) AS town,
      TRIM(customer_ID) AS customer_ID,
      TRIM(customer_Tax_ID) AS customer_Tax_ID,
      TRIM(SKU) AS SKU,
      TRIM(`Sales Rep`) AS `Sales Rep`,
      Promo_discount_perc,
      Promo_discount_RUR,
      TRIM(Document_for_discount) AS Document_for_discount,
      TRIM(SKU_ID) AS SKU_ID,
      quantity,
      price,
      amount,
      TRIM(filename) AS filename,
      filename_end_date,
      TRIM(weekday) AS weekday
    FROM `eztech-442521.rich.landing_table_distributor_sales`
    WHERE filename_end_date = max_landing_date;
    
    -- If max date in landing table is less than max_landing_date, insert a row with nulls
    IF max_actual_date < max_landing_date THEN
      INSERT INTO `eztech-442521.rich.main_table_distributor_sales` (
        date, document, unit, town, customer_ID, customer_Tax_ID, SKU, 
        `Sales Rep`, Promo_discount_perc, Promo_discount_RUR, Document_for_discount, 
        SKU_ID, quantity, price, amount, filename, filename_end_date, weekday
      )
      VALUES (
        max_landing_date, NULL, NULL, NULL, NULL, NULL, NULL, 
        NULL, NULL, NULL, NULL, 
        NULL, NULL, NULL, NULL, 'placeholder_for_missing_date', max_landing_date, FORMAT_DATE('%A', max_landing_date)
      );
    END IF;
    
    -- Update metadata
    INSERT INTO `eztech-442521.rich.metadata_table` (
      ingestion_date, file_name, destination_table, status, 
      date_from_file_name, day_from_file_name, no_of_records
    )
    SELECT
      CAST(FORMAT_DATE('%Y-%m-%d', CURRENT_DATE()) AS STRING) AS ingestion_date,
      max(TRIM(filename)) AS file_name,
      'main_table_distributor_sales' AS destination_table,
      'success' AS status,
      max_landing_date,
      FORMAT_DATE('%A', max_landing_date),
      COUNT(*)
    FROM `eztech-442521.rich.landing_table_distributor_sales`
    WHERE filename_end_date = max_landing_date;
  ELSE
    SELECT 'No newer data available in landing table';
  END IF;
END;

CALL `eztech-442521.rich.daily_main_table_distributor_sales_update`();




------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE `eztech-442521.rich.main_table_distributor_stocks`
PARTITION BY date
AS SELECT * 
FROM `eztech-442521.rich.landing_table_distributor_stocks` 
WHERE filename_end_date in ('2024-12-14', '2024-11-30', '2023-12-31', '2022-12-31');


CREATE OR REPLACE PROCEDURE `eztech-442521.rich.initial_main_table_distributor_stocks_update`()
BEGIN
  DECLARE process_dates ARRAY<DATE>;
  DECLARE i INT64 DEFAULT 1;
  
  -- Get all dates to process in chronological order
  SET process_dates = ARRAY(
    SELECT DISTINCT filename_end_date
    FROM `eztech-442521.rich.landing_table_distributor_stocks` 
    WHERE filename_end_date NOT IN ('2024-12-14', '2024-11-30', '2023-12-31', '2022-12-31')
    ORDER BY filename_end_date ASC
  );
  
  -- Process dates sequentially from earliest to latest
  WHILE i <= ARRAY_LENGTH(process_dates) DO
    -- Delete existing records for this date's data
    DELETE FROM `eztech-442521.rich.main_table_distributor_stocks`
    WHERE date IN (
      SELECT DISTINCT date 
      FROM `eztech-442521.rich.landing_table_distributor_stocks`
      WHERE filename_end_date = process_dates[ORDINAL(i)]
    );
    
    -- Insert new records
    INSERT INTO `eztech-442521.rich.main_table_distributor_stocks`
    SELECT * 
    FROM `eztech-442521.rich.landing_table_distributor_stocks`
    WHERE filename_end_date = process_dates[ORDINAL(i)];
    
    -- Update metadata
    INSERT INTO `eztech-442521.rich.metadata_table` (
      ingestion_date, file_name, destination_table, status, 
      date_from_file_name, day_from_file_name, no_of_records
    )
    SELECT
      CAST(FORMAT_DATE('%Y-%m-%d', CURRENT_DATE()) AS STRING) AS ingestion_date,
      max(filename) AS file_name,
      'main_table_distributor_stocks' AS destination_table,
      'success' AS status,
      process_dates[ORDINAL(i)],
      FORMAT_DATE('%A', process_dates[ORDINAL(i)]),
      COUNT(*)
    FROM `eztech-442521.rich.landing_table_distributor_stocks`
    WHERE filename_end_date = process_dates[ORDINAL(i)];
    
    SET i = i + 1;
  END WHILE;
END;

-- Call initialization procedure instead of daily one for initial load
CALL `eztech-442521.rich.initial_main_table_distributor_stocks_update`();

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE `eztech-442521.rich.daily_main_table_distributor_stocks_update`()
BEGIN
  DECLARE max_main_date DATE;
  DECLARE max_landing_date DATE;
  
  -- Get latest dates from both tables using filename_end_date
  SET max_main_date = (SELECT MAX(filename_end_date) FROM `eztech-442521.rich.main_table_distributor_stocks`);
  SET max_landing_date = (SELECT MAX(filename_end_date) FROM `eztech-442521.rich.landing_table_distributor_stocks`);
  
  -- Only proceed if landing has newer data
  IF max_landing_date > max_main_date THEN
    -- Delete only overlapping partitions with the latest filename_end_date
    DELETE FROM `eztech-442521.rich.main_table_distributor_stocks`
    WHERE date IN (
      SELECT DISTINCT date 
      FROM `eztech-442521.rich.landing_table_distributor_stocks`
      WHERE filename_end_date = max_landing_date
    );
    
    -- Insert only new data from landing table with latest filename_end_date
    INSERT INTO `eztech-442521.rich.main_table_distributor_stocks`
    SELECT * 
    FROM `eztech-442521.rich.landing_table_distributor_stocks`
    WHERE filename_end_date = max_landing_date;
    
    -- Update metadata
    INSERT INTO `eztech-442521.rich.metadata_table` (
      ingestion_date, file_name, destination_table, status, 
      date_from_file_name, day_from_file_name, no_of_records
    )
    SELECT
      CAST(FORMAT_DATE('%Y-%m-%d', CURRENT_DATE()) AS STRING) AS ingestion_date,
      max(filename) AS file_name,
      'main_table_distributor_stocks' AS destination_table,
      'success' AS status,
      max_landing_date,
      FORMAT_DATE('%A', max_landing_date),
      COUNT(*)
    FROM `eztech-442521.rich.landing_table_distributor_stocks`
    WHERE filename_end_date = max_landing_date;
  ELSE
    SELECT 'No newer data available in landing table';
  END IF;
END;

CALL `eztech-442521.rich.daily_main_table_distributor_stocks_update`();

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- Create the frozen table with selected columns
/*CREATE OR REPLACE TABLE `eztech-442521.rich.distributor_sales_frozen`
PARTITION BY date
AS SELECT 
  date,
  unit,
  customer_ID,
  CAST(NULL AS STRING) AS contract,
  CAST(NULL AS STRING) AS channel,
  SKU,
  Promo_discount_perc,  -- This matches "Promo discount %"
  SKU_ID,
  quantity,
  price,
  amount
FROM `eztech-442521.rich.main_table_distributor_sales` 
WHERE 1=0;*/

-- Create procedure to refresh the frozen table weekly
CREATE OR REPLACE PROCEDURE `eztech-442521.rich.weekly_distributor_sales_frozen_refresh`()
BEGIN
  DECLARE latest_sunday DATE;
  DECLARE start_date DATE;
  
  -- Find the most recent Sunday (0 = Sunday in FORMAT_DATE)
  SET latest_sunday = DATE_SUB(CURRENT_DATE(), INTERVAL MOD(EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 6, 7) DAY);
  
  -- Calculate start date (4 weeks before the latest Sunday)
  SET start_date = DATE_SUB(latest_sunday, INTERVAL 27 DAY);
  
  -- Log the calculated dates for debugging
  SELECT FORMAT('Latest Sunday: %t, Start date: %t', latest_sunday, start_date) AS debug_info;
  
  -- Clear and repopulate the frozen table
  TRUNCATE TABLE `eztech-442521.rich.distributor_sales_frozen`;
  
  -- Insert data from the main table for the 4-week period with only selected columns
  INSERT INTO `eztech-442521.rich.distributor_sales_frozen`
  SELECT 
    sales.date,
    sales.unit,
    sales.customer_ID,
    catalog.parent_1_document,
    customers.parent_3_name,
    sales.SKU,
    sales.Promo_discount_perc,  -- This matches "Promo discount %"
    sales.SKU_ID,
    sales.quantity,
    sales.price,
    sales.amount
  FROM `eztech-442521.rich.main_table_distributor_sales` sales
  left join `eztech-442521.rich.customers` customers on sales.customer_ID = customers.child_code
  left join `eztech-442521.rich.catalog` catalog on catalog.SKU_ID = sales.SKU_ID
  WHERE sales.date BETWEEN start_date AND latest_sunday;
  
  -- Log the refresh in metadata table
  INSERT INTO `eztech-442521.rich.metadata_table` (
    ingestion_date, file_name, destination_table, status, 
    date_from_file_name, day_from_file_name, no_of_records
  )
  SELECT
    CAST(FORMAT_DATE('%Y-%m-%d', CURRENT_DATE()) AS STRING) AS ingestion_date,
    'weekly_refresh' AS file_name,
    'distributor_sales_frozen' AS destination_table,
    'success' AS status,
    latest_sunday,
    'Sunday',
    COUNT(*)
  FROM `eztech-442521.rich.distributor_sales_frozen`;
END;

-- This procedure would be scheduled to run every Wednesday
-- CALL `eztech-442521.rich.weekly_distributor_sales_frozen_refresh`();

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- Create the frozen table for distributor stocks with selected columns
CREATE OR REPLACE TABLE `eztech-442521.rich.distributor_stocks_frozen`
PARTITION BY date
AS SELECT 
  date,
  warehouse,
  SKU_ID,
  price,
  stock,
  total
FROM `eztech-442521.rich.main_table_distributor_stocks` 
WHERE 1=0;

-- Create procedure to refresh the stocks frozen table weekly
CREATE OR REPLACE PROCEDURE `eztech-442521.rich.weekly_distributor_stocks_frozen_refresh`()
BEGIN
  DECLARE latest_sunday DATE;
  DECLARE start_date DATE;
  
  -- Find the most recent Sunday (0 = Sunday in FORMAT_DATE)
  SET latest_sunday = DATE_SUB(CURRENT_DATE(), INTERVAL MOD(EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 6, 7) DAY);
  
  -- Calculate start date (4 weeks before the latest Sunday)
  SET start_date = DATE_SUB(latest_sunday, INTERVAL 27 DAY);
  
  -- Log the calculated dates for debugging
  SELECT FORMAT('Latest Sunday: %t, Start date: %t', latest_sunday, start_date) AS debug_info;
  
  -- Clear and repopulate the frozen table
  TRUNCATE TABLE `eztech-442521.rich.distributor_stocks_frozen`;
  
  -- Insert data from the main table for the 4-week period with only selected columns
  INSERT INTO `eztech-442521.rich.distributor_stocks_frozen`
  SELECT 
    date,
    warehouse,
    SKU_ID,
    price,
    stock,
    total
  FROM `eztech-442521.rich.main_table_distributor_stocks`
  WHERE date BETWEEN start_date AND latest_sunday;
  
  -- Log the refresh in metadata table
  INSERT INTO `eztech-442521.rich.metadata_table` (
    ingestion_date, file_name, destination_table, status, 
    date_from_file_name, day_from_file_name, no_of_records
  )
  SELECT
    CAST(FORMAT_DATE('%Y-%m-%d', CURRENT_DATE()) AS STRING) AS ingestion_date,
    'weekly_refresh' AS file_name,
    'distributor_stocks_frozen' AS destination_table,
    'success' AS status,
    latest_sunday,
    'Sunday',
    COUNT(*)
  FROM `eztech-442521.rich.distributor_stocks_frozen`;
END;

-- This procedure would be scheduled to run every Wednesday
-- CALL `eztech-442521.rich.weekly_distributor_stocks_frozen_refresh`();


CREATE OR REPLACE VIEW `eztech-442521.rich.goods_and_attributes` AS
SELECT
  TRIM(Code) AS `Код`,
  TRIM(Name) AS `Наименование`,
  SAFE_CAST(REPLACE(gl, ',', '') AS FLOAT64) AS `гл`,
  SAFE_CAST(REPLACE(NetWeight, ',', '') AS FLOAT64) AS `весНетто`,
  SAFE_CAST(REPLACE(GrossWeight, ',', '') AS FLOAT64) AS `весБрутто`,
  SAFE_CAST(REPLACE(Packaging, ',', '') AS FLOAT64) AS `упаковка`,
  TRIM(GTIN) AS `GTIN`,
  TRIM(SKU) AS `SKU`,
  TRIM(SKU_Heineken) AS `SKU_Хайнекен`,
  TRIM(English) AS `Англ`,
  TRIM(Article) AS `Артикул`,
  TRIM(BEREZNiki) AS `БЕРЕЗНИКИ`,
  TRIM(Brand) AS `Бренд`,
  TRIM(VERESHCHAGINO) AS `ВЕРЕЩАГИНО`,
  TRIM(EXTERNAL_CODE_BORJOMI) AS `ВНЕШНИЙ_КОД_БОРЖОМИ`,
  TRIM(GUBAKHA) AS `ГУБАХА`,
  TRIM(CODE_LBD) AS `КОД_ЛБД`,
  TRIM(CODE_NIDAN) AS `КОД_НИДАН`,
  TRIM(Code_iSales) AS `Код_iSales`,
  TRIM(Code_iSalesPepsi1) AS `Код_iSalesПепси1`,
  TRIM(Code_iSalesPepsi2) AS `Код_iSalesПепси2`,
  TRIM(Code_iSalesPepsi3) AS `Код_iSalesПепси3`,
  TRIM(Code_iSalesPepsi4) AS `Код_iSalesПепси4`,
  TRIM(Code_iSalesPepsi5) AS `Код_iSalesПепси5`,
  TRIM(Code_RP) AS `Код_РП`,
  TRIM(Strength) AS `Крепость`,
  TRIM(KUDYMAR) AS `КУДЫМКАР`,
  TRIM(KUNGUR) AS `КУНГУР`,
  TRIM(LYSVA) AS `ЛЫСЬВА`,
  TRIM(NameJTS) AS `НаименованиеJTS`,
  TRIM(NomenclatureGroup_MPK) AS `Номенклатурная_группа_МПК`,
  TRIM(MainProperty) AS `Основное_свойство`,
  TRIM(PERM) AS `ПЕРМЬ`,
  TRIM(PET_Container) AS `ПЭТ_тара`,
  SAFE_CAST(REPLACE(Rocase, ',', '') AS FLOAT64) AS `рокейс`,
  TRIM(Segment) AS `Сегмент`,
  TRIM(ProductGroup) AS `Товарная_группа`,
  TRIM(Packaging1) AS `Упаковка1`,
  TRIM(CHERNUSHKA) AS `ЧЕРНУШКА`
FROM
  `eztech-442521.rich.trans_goods`;


  CREATE VIEW `eztech-442521.rich.customers` AS
SELECT
  child_code AS child_code,
  name AS `Наименование`,
  tax_id AS `ИНН`,
  full_name AS `Полное_наименование`,
  parent_1 AS parent_1,
  parent_1_name AS parent_1_name,
  parent_2 AS parent_2,
  parent_2_name AS parent_2_name,
  parent_3 AS parent_3,
  parent_3_name AS parent_3_name,
  parent_4 AS parent_4,
  parent_4_name AS parent_4_name,
  parent_5 AS parent_5,
  parent_5_name AS parent_5_name,
  parent_6 AS parent_6,
  parent_6_name AS parent_6_name,
  level AS level
FROM `eztech-442521.rich.trans_customers`;

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE VIEW `eztech-442521.rich.stock_model_view` AS
WITH warehouse_units AS (
  SELECT * FROM UNNEST([
    STRUCT('ЛЫСЬВА ОСНОВНОЙ СКЛАД' AS warehouse, 'Лысьва' AS unit),
    ('КУНГУР ОСНОВНОЙ СКЛАД', 'Кунгур'),
    ('ВЕРЕЩАГИНО ОСНОВНОЙ СКЛАД', 'Верещагино'),
    ('БЕРЕЗНИКИ ОСНОВНОЙ СКЛАД', 'Березники'),
    ('БЕРЕЗНИКИ НОРД', 'Березники'),
    ('ПЕРМЬ ОСНОВНОЙ СКЛАД', 'Пермь'),
    ('ЛЫСЬВА НОРД', 'Лысьва')
  ])),
  current_date AS (
  SELECT CURRENT_DATE() AS current_date
),
sales as (
  SELECT 
    date,
    TRIM(unit) AS unit,
    TRIM(customer_ID) AS customer_ID,
    TRIM(SKU) AS SKU,
    TRIM(SKU_ID) AS SKU_ID,
    quantity,
    amount
  FROM `eztech-442521.rich.main_table_distributor_sales` 
  WHERE date >= (SELECT DATE_SUB(current_date, INTERVAL 15 DAY) FROM current_date)
), 

customers as (
  SELECT 
    child_code,
    parent_3_name as channel
  FROM `eztech-442521.rich.customers`
),
stock as (
  SELECT 
    date,
    warehouse,
    sku_id,
    price,
    stock,
    total
  FROM `eztech-442521.rich.main_table_distributor_stocks`
),
max_stock_date AS (
  SELECT MAX(date) AS max_date FROM stock
),
latest_stock AS (
  SELECT 
    s.sku_id,
    s.warehouse,
    sum(s.stock) stock,
    wu.unit as mapped_unit
  FROM stock s
  JOIN max_stock_date m ON s.date = m.max_date
  LEFT JOIN warehouse_units wu ON s.warehouse = wu.warehouse
  group by s.sku_id, s.warehouse, wu.unit
),
max_sales_date AS (
  SELECT MAX(date) AS max_date FROM sales
),
sales_14_days_average as (
  SELECT 
    sku_id,
    unit,
    sum(quantity)/14 as average_quantity,
    sum(amount)/14 as average_amount
  FROM sales
  WHERE date >= (SELECT DATE_SUB(current_date, INTERVAL 15 DAY) FROM current_date)
  GROUP BY sku_id, unit
),
goods as (
  SELECT 
    `Код` as sku_id,
    `гл` as gl
  from `eztech-442521.rich.goods_and_attributes`
),
forecasts as (
  select 
    sku_id,
    channel,
    unit,
    sum(predicted_quantity) as predicted_quantity
  from `eztech-442521.rich.distributor_forecasts`
  where `Type of Forecast` = 'Accurate'
  group by sku_id, channel, unit
),
catalog as (
  select 
    sku_id,
    parent_1_document as contract
  from `eztech-442521.rich.catalog`
)
  

-- Now for the final SELECT statement connecting everything:
SELECT 
  (SELECT current_date FROM current_date) as report_date,
  s.sku_id,
  s.SKU as sku_name,
  c.channel,
  cat.contract ,
  s.unit,
  COALESCE(st.stock, 0) as current_stock_items,
  ROUND(COALESCE(st.stock, 0) * g.gl,2) as current_stock_HL,
  CASE 
    WHEN avg.average_quantity > 0 
    THEN ROUND(SAFE_DIVIDE(COALESCE(st.stock, 0), avg.average_quantity))
    ELSE NULL 
  END as days_of_stock_remaining,
  ROUND(avg.average_quantity) as hist_avg_daily_quantity_14d,
  ROUND(f.predicted_quantity/14) as predicted_avg_daily_quantity_14d,

FROM sales s
JOIN max_sales_date msd ON s.date = msd.max_date
LEFT JOIN customers c ON s.customer_ID = c.child_code
LEFT JOIN latest_stock st ON s.SKU_ID = st.sku_id AND s.unit = st.mapped_unit
LEFT JOIN sales_14_days_average avg ON s.SKU_ID = avg.sku_id AND s.unit = avg.unit
LEFT JOIN goods g ON s.SKU_ID = g.sku_id
LEFT JOIN catalog cat ON trim(s.SKU_ID) = trim(cat.sku_id)
LEFT JOIN forecasts f ON trim(s.SKU_ID) = trim(f.sku_id) AND trim(s.unit) = trim(f.unit) AND trim(f.channel) = trim(c.channel);

select * from `eztech-442521.rich.stock_model_view` where predicted_daily_quantity_14d is not null;



select count(distinct sku_id) from `eztech-442521.rich.stock_model_view` where trim(sku_id) in (select distinct trim(sku_id) from `rich.distributor_forecasts` );

select sku_id,channel,unit,predicted_quantity,b.channel as b_channel,b.unit as b_unit from 
`eztech-442521.rich.distributor_forecasts` a 
join `eztech-442521.rich.stock_model_view` b on trim(a.sku_id) = trim(b.sku_id) 
 where `Type of Forecast` = 'Accurate' 


select contract,count(distinct sku_id) from (
select a.sku_id,b.parent_1_document as contract from
(SELECT
    sku_id,
    COUNT(DISTINCT date) AS cnt
FROM
    `rich.main_table_distributor_sales`
WHERE
    date >= '2024-01-01'
GROUP BY
    sku_id
HAVING
    COUNT(DISTINCT date) > 130
    -- Ensure the SKU has records in both 2024 and 2025 periods
    AND MIN(date) <= '2024-12-31' -- Check for presence in 2024
    AND MAX(date) >= '2025-01-01' -- Check for presence in 2025
) a
join `rich.catalog` b on a.sku_id=b.sku_id)
group by contract
order by count(distinct sku_id) desc;

SELECT
    sku_id,
    COUNT(DISTINCT date) AS cnt,min(quantity) as min_quantity,max(quantity) as max_quantity
FROM
    `rich.main_table_distributor_sales`
WHERE
    date >= '2024-01-01'
GROUP BY
    sku_id
HAVING
    COUNT(DISTINCT date) > 130
    -- Ensure the SKU has records in both 2024 and 2025 periods
    AND MIN(date) <= '2024-12-31' -- Check for presence in 2024
    AND MAX(date) >= '2025-01-01' -- Check for presence in 2025
