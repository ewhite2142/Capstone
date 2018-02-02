CREATE TABLE summits (
summit_id       int PRIMARY KEY,
name            varchar(50),
elevation       int,
isolation       numeric(6,2),
prominence      int,
latitude        numeric(8,5),
longitude       numeric(8,5),
type            smallint,
type_str        varchar(8),
counties        varchar(50),
states          varchar(50),
state           char(2),
quad            varchar(50)
);


CREATE TABLE images (
summit_id   int,
image_id    int PRIMARY KEY,
url         varchar(50),
filename    varchar(20)
);
ALTER TABLE images
ADD CONSTRAINT images_summit_id_fkey
FOREIGN KEY (summit_id)
REFERENCES summits (summit_id)
ON UPDATE CASCADE;


CREATE TABLE model_fit_results (
model_num               int,
time_completed          timestamp with time zone,
cnn_test_accuracy       numeric(5, 2),
gbc_test_accuracy       numeric(5, 2),
comparison              varchar(20),
by_type                 varchar(14),
numrows_in_each_class   smallint,
num_epochs              smallint,
batch_size              int,
num_filters             int,
pool_size               int[],
kernel_size             int[],
input_shape             int[],
dense                   int,
dropout1                numeric(3,2),
dropout2                numeric(3,2),
num_classes             smallint,
model_filename          varchar(50),
PRIMARY KEY (model_num, comparison, by_type)
);

###### set summit type, type_str ################################
UPDATE summits SET type=5, type_str='None';

UPDATE summits SET type=0, type_str='mount'
WHERE name LIKE '%Mount%' AND name NOT LIKE '%Mountain%' AND name NOT LIKE '%Mountaineer'
    AND  ( name NOT LIKE '%Peak%'
        OR name LIKE '%-North Peak%'
        OR name LIKE '%-South Peak%'
        OR name LIKE '%-East Peak%'
        OR name LIKE '%-West Peak%'
        OR name LIKE '%-Middle Peak%'
        OR name LIKE '%-Southwest Peak%' );

UPDATE summits SET type=1, type_str='mountain'
WHERE name LIKE '%Mountain%' AND name NOT LIKE '%Mountaineer'
AND  ( name NOT LIKE '%Peak%'
    OR name LIKE '%-North Peak%'
    OR name LIKE '%-South Peak%'
    OR name LIKE '%-East Peak%'
    OR name LIKE '%-West Peak%'
    OR name LIKE '%-Middle Peak%'
    OR name LIKE '%-Southwest Peak%' );

UPDATE summits SET type=2, type_str='peak'
WHERE name LIKE '%Peak%' AND name NOT LIKE '%Mount%'
    AND name NOT LIKE '%-North Peak%'
    AND name NOT LIKE '%-South Peak%'
    AND name NOT LIKE '%-East Peak%'
    AND name NOT LIKE '%-West Peak%'
    AND name NOT LIKE '%-Middle Peak%'
    AND name NOT LIKE '%-Southwest Peak%';

UPDATE summits SET type=4, type_str='ambiguous' WHERE summit_id IN
(
SELECT summit_id FROM summits
WHERE type = 5 AND (
    name LIKE '%Mount%'  OR
    name LIKE '%Peak%'
                    )
    AND name NOT LIKE '%-North Peak%'
    AND name NOT LIKE '%-South Peak%'
    AND name NOT LIKE '%-East Peak%'
    AND name NOT LIKE '%-West Peak%'
    AND name NOT LIKE '%-Middle Peak%'
    AND name NOT LIKE '%-Southwest Peak%'
);

UPDATE summits SET type=2, type_str='peak'
WHERE name LIKE '%Mountaineer Peak%';

# ----- to confirm above is correct ------
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

SELECT COUNT(*) FROM summits WHERE state IS NULL;
SELECT summit_id, longitude FROM summits WHERE longitude > 0;
SELECT * FROM summits WHERE LENGTH(state)>2;
SELECT DISTINCT state FROM summits ORDER BY state;
SELECT EXISTS (SELECT summit_id from summits where summit_id<0);
####################################################


##################################################
# some image files were corrupt and could not be loaded, so move them out of summits and images tables
#MOVE bad images into new tables, bad_images and bad_summits
CREATE TABLE bad_images
AS TABLE images
WITH NO DATA;

INSERT INTO bad_images
SELECT * FROM images WHERE image_id IN (1000,2000,3000,4000,5000,6000,7000,8000,8186,8374,8739,9000,10000,10255,10422,10647,10667,10670,10671,10672,11000,11065,11066,11247,11292,11293,11418,11419,11420,11421,11422,11423,11424,11425,11426,11427,11428,11430,11431,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000,41000,42000,43000,44000,45000,46000,47000,48000,49000,50000);

DELETE FROM images WHERE image_id IN
(SELECT image_id FROM bad_images);

CREATE TABLE bad_summits
AS TABLE summits
WITH NO DATA;

INSERT INTO bad_summits
SELECT * FROM summits s WHERE NOT EXISTS
(SELECT image_id FROM images WHERE summit_id=s.summit_id);

DELETE FROM summits WHERE summit_id IN
(SELECT summit_id from bad_summits);
###################################################

# VARIOUS SCRIPTS TO GET INFO ABOUT DATA IN DB

SELECT * FROM model_fit_results WHERE model_num = (SELECT MAX(model_num) FROM model_fit_results) ORDER BY model_num DESC, time_completed;

SELECT COUNT(*) "Total #summits in summits TABLE" FROM summits;
SELECT COUNT(DISTINCT summit_id) "#Total #summits in images table" FROM images;
SELECT COUNT(*) "Total #images in images table" FROM images;

# #summits & images for Mount, Mountains, & Peaks by type
SELECT type_str, COUNT(DISTINCT s.summit_id) "#summits", COUNT(DISTINCT i.image_id) "#images"
FROM summits s RIGHT OUTER JOIN images i ON s.summit_id=i.summit_id
GROUP BY type_str, type
ORDER BY type;

SELECT state, COUNT(DISTINCT s.summit_id) "#summits", COUNT(DISTINCT i.image_id) "#images"
FROM summits s RIGHT OUTER JOIN images i ON s.summit_id=i.summit_id
GROUP BY s.state ORDER BY "#images" DESC, "#summits" DESC;

SELECT COUNT(DISTINCT s.summit_id) "#summits-Appalachia", COUNT(image_id) AS "#images-Appalachia"
FROM summits s RIGHT OUTER JOIN images i on s.summit_id = i.summit_id
WHERE state IN ('WV', 'AL', 'GA', 'KY', 'MD', 'MS', 'NY', 'NC', 'OH', 'PA', 'SC', 'TN', 'VA');

SELECT COUNT(DISTINCT s.summit_id) "#summits-Mtn States", COUNT(image_id) AS "#images-Mtn States"
FROM summits s RIGHT OUTER JOIN images i on s.summit_id = i.summit_id
WHERE state IN ('AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY');

SELECT state, AVG(elevation) "Avg Elev", MAX(elevation) "Max Elev", MIN(elevation) "Min Elev", AVG(prominence) "Avg Promin", MAX(prominence) "Max Promin", MIN(prominence) "Min Promin", AVG(isolation) "Avg Isol", MAX(isolation) "Max Isol", MIN(isolation) "Min Isol"
FROM summits
WHERE state IN ('NM', 'UT', 'WA', 'CO')
GROUP BY state
ORDER BY state;

SELECT state, COUNT(summit_id) "# Summits", AVG(elevation) "Avg Elev", AVG(prominence) "Avg Promin", AVG(isolation) "Avg Isol"
FROM summits
WHERE state IN ('NM', 'UT', 'WA', 'CO')
GROUP BY state
ORDER BY state;

SELECT type_str, COUNT(summit_id) "# Summits", AVG(elevation) "Avg Elev", AVG(prominence) "Avg Promin", AVG(isolation) "Avg Isol"
FROM summits
WHERE state IN ('NM', 'UT', 'WA', 'CO')
    AND type_str IN ('mount', 'mountain', 'peak')
GROUP BY type_str
ORDER BY type_str;

SELECT type_str, COUNT(summit_id) "# Summits", AVG(elevation) "Avg Elev", AVG(prominence) "Avg Promin", AVG(isolation) "Avg Isol"
FROM summits
WHERE state IN ('WA', 'CO')
    AND type_str IN ('mount', 'mountain', 'peak')
GROUP BY type_str
ORDER BY type_str;

SELECT state, count(summit_id) "# Summits"
FROM summits
WHERE elevation > 1000
GROUP BY state
ORDER BY "# Summits" DESC
LIMIT 6;

WITH summits_ AS (
SELECT * FROM summits WHERE elevation>1000
   AND state IN ('NM', 'UT', 'WA', 'CO') )
SELECT s.state, COUNT(DISTINCT s.summit_id) "#Summits", COUNT(DISTINCT i.image_id) "#Images" FROM summits_ s INNER JOIN images i ON s.summit_id=i.summit_id
GROUP BY s.state
ORDER BY state;

SELECT type_str, MIN(elevation) "Min Elev"
FROM summits
GROUP BY type_str
ORDER BY type_str;

SELECT COUNT(*) FROM summits WHERE elevation < 1000;
