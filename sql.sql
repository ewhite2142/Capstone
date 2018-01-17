WITH imcnts AS (
SELECT summit_id, COUNT(image_id) cnt
FROM summits
GROUP BY summit_id
HAVING COUNT(image_id) = 1
)
SELECT m.type_str, COUNT(m.summit_id)
FROM imcnts c INNER JOIN master m
ON c.summit_id = m.summit_id
GROUP BY m.type_str
;

#to count #files in a folder, from Terminal:
find . -type f | wc -l

ALTER TABLE summits
ADD PRIMARY KEY (summit_id);

ALTER TABLE images
ADD CONSTRAINT images_summit_id_fkey
FOREIGN KEY (summit_id)
REFERENCES summits (summit_id)
ON UPDATE CASCADE;

UPDATE summits
SET longitude = -longitude
WHERE longitude > 0;

SELECT * FROM summits
WHERE longitude > 0;

ALTER TABLE summits
ADD COLUMN counties varchar(50),
ADD COLUMN state varchar(50);

ALTER TABLE summits
ALTER COLUMN type type smallint using type::smallint;

ALTER TABLE summits
RENAME COLUMN state to states;

ALTER TABLE summits
RENAME COLUMN summit_state to state;

SELECT COUNT(*) "# in summits" FROM summits;
SELECT COUNT(DISTINCT summit_id) "#distinct summit_id in images" FROM images;
SELECT COUNT(*) "# in images" FROM images;

SELECT s.type_str, COUNT(s.summit_id) "#summits", COUNT(i.image_id) "#images"
FROM images i INNER JOIN summits s ON i.summit_id=s.summit_id
GROUP BY type_str;

SELECT summit_id, name, type, type_str FROM summits
WHERE
    name LIKE '%Mountain%' AND type_str <> 'mountain' OR
    name LIKE '%Mount%' AND name NOT LIKE '%Mountain%' AND type_str <> 'mount' OR
    name LIKE '%Peak#' AND type_str <> 'peak'
ORDER BY summit_id;

SELECT COUNT(*) "num wrong Mount" FROM summits
WHERE name LIKE '%Mount%' AND name NOT LIKE '%Mountain%' AND type_str <> 'mount';

SELECT COUNT(*) "num wrong Mountain" FROM summits
WHERE name LIKE '%Mountain%' AND type_str <> 'mountain';

SELECT COUNT(*) "num wrong Peak" FROM summits
WHERE name LIKE '%Peak%' AND type_str <> 'peak';

SELECT COUNT(*) "num wrong None" FROM summits
WHERE
    name NOT LIKE '%Mount%' AND name NOT LIKE '%Mountain%' AND name NOT LIKE '%Peak%' AND type_str <> 'None';

SELECT COUNT(*) FROM summits
WHERE state IS NULL OR counties IS NULL;

#UBUNTO to take ownership of a file
sudo chown ed:ed filename


#MAC: to save table to to_csv
#in psql, connected to summitsdb:
COPY summits TO '/Users/edwardwhite/dsi/Capstone/summits.csv' DELIMITER ',' CSV HEADER;
COPY summits TO '/home/ed/dsi/Capstone/summits.csv' DELIMITER ','

#UBUNTU: to save table to to_csv
#in psql, connected to summitsdb:
COPY summits TO '/tmp/summits.csv' DELIMITER ',' CSV HEADER;
in bash:
cp '/tmp/summits.csv' '/home/ed/dsi/Capstone/'

#to import csv into table in postgresql
ALTER TABLE images DROP CONSTRAINT images_summit_id_fkey;
DELETE FROM summits;
COPY summits FROM '/home/ed/dsi/Capstone/summits.csv' WITH CSV HEADER DELIMITER ',';
ALTER TABLE images ADD FOREIGN KEY(summit_id) REFERENCES summits(summit_id);

#export entire DB to file. In terminal on Ubuntu:
pg_dump -U ed -O summitsdb > summitsdb.sql;

#import entire DB from file:
In psql:
DROP DATABASE IF EXISTS summitsdb;
CREATE DATABASE summitsdb;
In terminal on mac:
psql -U edwardwhite summitsdb < summitsdb.sql;

-- \copy summits FROM 'path' DELIMITER ',' csv
COPY summits FROM 'completepath' WITH HEADER, DELIMITER ',';

#to get OWNER of current database:
SELECT u.usename
FROM pg_database d
JOIN pg_user u ON (d.datdba = u.usesysid)
WHERE d.datname = '(SELECT current_database())';

##################################################
#MOVE bad images into new tables, bad_images and bad_summits
CREATE TABLE bad_images
AS TABLE images
WITH NO DATA;

INSERT INTO bad_images
SELECT * FROM images WHERE image_id IN (1000,2000,3000,4000,5000,6000,7000,8000,8186,8374,8739,9000,10000,10255,10422,10647,10667,10670,10671,10672,11000,11065,11066,11247,11292,11293,11418,11419,11420,11421,11422,11423,11424,11425,11426,11427,11428,11430,11431,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000,41000,42000,43000,44000,45000,46000,47000,48000,49000,50000);

SELECT * FROM bad_images ORDER BY image_id;

DELETE FROM images WHERE image_id IN
(SELECT image_id FROM bad_images);

SELECT * FROM summits s WHERE NOT EXISTS
(SELECT image_id FROM images WHERE summit_id=s.summit_id);

CREATE TABLE bad_summits
AS TABLE summits
WITH NO DATA;

INSERT INTO bad_summits
SELECT * FROM summits s WHERE NOT EXISTS
(SELECT image_id FROM images WHERE summit_id=s.summit_id);

DELETE FROM summits WHERE summit_id IN
(SELECT summit_id from bad_summits);
###################################################

# #summits & images for Mount, Mountains, & Peaks by type
SELECT type_str, COUNT(DISTINCT s.summit_id) "#summits", COUNT(i.image_id) "#images"
FROM summits s RIGHT OUTER JOIN images i ON s.summit_id=i.summit_id
GROUP BY type_str, type ORDER BY type;

SELECT state, COUNT(DISTINCT s.summit_id) "#summits", COUNT(i.image_id) "#images"
FROM summits s RIGHT OUTER JOIN images i ON s.summit_id=i.summit_id
GROUP BY s.state ORDER BY "#summits" DESC, "#images" DESC;

SELECT COUNT(DISTINCT s.summit_id) "#summits-Appalachia", COUNT(image_id) AS "#images-Appalachia"
FROM summits s RIGHT OUTER JOIN images i on s.summit_id = i.summit_id
WHERE state IN ('WV', 'AL', 'GA', 'KY', 'MD', 'MS', 'NY', 'NC', 'OH', 'PA', 'SC', 'TN', 'VA');

SELECT COUNT(DISTINCT s.summit_id) "#summits-Mtn States", COUNT(image_id) AS "#images-Mtn States"
FROM summits s RIGHT OUTER JOIN images i on s.summit_id = i.summit_id
WHERE state IN ('AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY');

SELECT COUNT(*) FROM summits;
SELECT COUNT(*) FROM images;


SELECT EXISTS (SELECT summit_id from summits where summit_id<0);

SELECT * FROM summits LIMIT 1;
SELECT DISTINCT state FROM summits ORDER BY state;

SELECT * FROM images WHERE summit_id=11;
SELECT * FROM summits WHERE summit_id=11;

SELECT * FROM summits WHERE LENGTH(state)>2;
