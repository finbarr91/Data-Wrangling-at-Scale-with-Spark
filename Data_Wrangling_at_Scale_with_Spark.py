import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
spark = SparkSession.builder.appName("Sparkminiproject").getOrCreate()
import re
import pandas as pd
from pyspark.sql.functions import regexp_extract
#
# if 'sc' not in locals():
#     import pyspark
#     from pyspark.context import SparkContext
#     from pyspark.sql.context import SQLContext
#     from pyspark.sql.session import SparkSession
#
#     sc = SparkContext()
#     sqlContext = SQLContext(sc)
#     spark = SparkSession(sc)

m = re.finditer(r'.*?(spark).*?', "I'm searching for a spark in PySpark", re.I)
for match in m:
    print(match)

# make sure you have upload NASA_access_log_Aug95.txt and NASA_access_log_Jul95.txt
# onto Spark before you run the following code
base_df = spark.read.text(r'C:\Users\chukw\PycharmProjects\Data-Wrangling-at-Scale-with-Spark\access_log_Jul95')
base_df.printSchema()

print(type(base_df))

# You can also convert a dataframe to an RDD if needed
base_df_rdd = base_df.rdd
print(type(base_df_rdd))


# Viewing sample data in our dataframe
# Looks like it needs to be wrangled and parsed!
print(base_df.show(10, truncate=False))



# Getting data from an RDD is slightly different.
# You can see how the data representation is different in the following RDD

print(base_df_rdd.take(10))
# Let's take a look at our dataset dimensions
print((base_df.count(), len(base_df.columns)))

# Let's extract and take a look at some sample log messages

sample_logs = [item['value'] for item in base_df.take(15)]
print(sample_logs)

# Extracting host names
# Let's try and write some regular expressions to extract the host name from the logs

host_pattern = r'(^\S+\.[\S+\.]+\S+)\s'
hosts = [re.search(host_pattern, item).group(1)
           if re.search(host_pattern, item)
           else 'no match'
           for item in sample_logs]
print(hosts)

# Extracting timestamps
# Let's now try and use regular expressions to extract the timestamp fields from the logs
ts_pattern = r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]'
timestamps = [re.search(ts_pattern, item).group(1) for item in sample_logs]
print(timestamps)

# Extracting HTTP Request Method, URIs and Protocol
# Let's now try and use regular expressions to extract the HTTP request methods,
# URIs and Protocol patterns fields from the logs
method_uri_protocol_pattern = r'\"(\S+)\s(\S+)\s*(\S*)\"'
method_uri_protocol = [re.search(method_uri_protocol_pattern, item).groups()
               if re.search(method_uri_protocol_pattern, item)
               else 'no match'
              for item in sample_logs]
print(method_uri_protocol)

# Building an intermediate parsed dataframe
# Let's try and use our regular expressions we have implemented so far into parsing and ' \
# extracting the relevant entities in separate columns in a new dataframe
logs_df = base_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                         regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                         regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                         regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                         regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'))
logs_df.show(10, truncate=False)
print((logs_df.count(), len(logs_df.columns)))

# Extracting HTTP Status Codes
# Let's now try and use regular expressions to extract the HTTP status codes from the logs
status_pattern = r'\s(\d{3})\s'
status = [re.search(status_pattern, item).group(1) for item in sample_logs]
print(status)

# Extracting HTTP Response Content Size
# Let's now try and use regular expressions to extract the HTTP response content size from the logs
content_size_pattern = r'\s(\d+)$'
content_size = [re.search(content_size_pattern, item).group(1) for item in sample_logs]
print(content_size)
"""
Q1: Your Turn: Putting it all together
Let's now try and leverage all the regular expression patterns we previously built and use the regexp_extract(...) method to build our dataframe with all the log attributes neatly extracted in their own separate columns.

    You can reuse the code we used previously to build the intermediate dataframe
    Remember to cast the HTTP status code and content size as integers.
    You can cast data as integer type using the following: regexp_extract('value', ...., ...).cast('integer').alias(...)  
"""

# TODO: Replace <FILL IN> with appropriate code

logs_df = base_df.select(regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                         regexp_extract('value', content_size_pattern, 1).cast('integer').alias('content_size'),
                         regexp_extract('value', host_pattern, 1).alias('host'),
                         regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                         regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                         regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                         regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'))
logs_df.show(10, truncate=True)
print((logs_df.count(), len(logs_df.columns)))

# Finding Missing Values
# Missing and null values are the bane of data analysis and machine learning.
# Let's see how well our data parsing and extraction logic worked. First, ' \
# let's verify that there are no null rows in the original dataframe.

base_df.filter(base_df['value'].isNull()).count()

# If our data parsing and extraction worked properly, we should not have any rows with
# potential null values. Let's try and put that to test!

bad_rows_df = logs_df.filter(logs_df['host'].isNull()|
                             logs_df['timestamp'].isNull() |
                             logs_df['method'].isNull() |
                             logs_df['endpoint'].isNull() |
                             logs_df['status'].isNull() |
                             logs_df['content_size'].isNull()|
                             logs_df['protocol'].isNull())
print(bad_rows_df.count())

# Ouch! Looks like we have over 30K missing values in our data! Can we handle this?
# Do remember, this is not a regular pandas dataframe which you can directly query and get which columns have null. Our so-called big dataset is residing on disk which can potentially be present in multiple nodes in a spark cluster. So how do we find out which columns have potential nulls?
# Finding Null Counts
# We can typically use the following technique to find out which columns have null values.
# (Note: This approach is adapted from an excellent answer on StackOverflow.)

print(logs_df.columns)

from pyspark.sql.functions import col
from pyspark.sql.functions import sum as spark_sum

def count_null(col_name):
    return spark_sum(col(col_name).isNull().cast('integer')).alias(col_name)

# Build up a list of column expressions, one per column.
exprs = [count_null(col_name) for col_name in logs_df.columns]

# Run the aggregation. The *exprs converts the list of expressions into
# variable function arguments.
logs_df.agg(*exprs).show()



# Well, looks like we have one missing value in the status column and everything else is in the content_size column. Let's see if we can figure out what's wrong!
# Handling nulls in HTTP status
# If you had solved it correctly, our original parsing regular expression for the status column was:
# regexp_extract('value', r'\s(\d{3})\s', 1).cast('integer').alias('status')
# Could it be that there are more digits making our regular expression wrong? or is the data point itself bad? Let's try and find out!
# Note: In the expression below, ~ means "not".
null_status_df = base_df.filter(~base_df['value'].rlike(r'\s(\d{3})\s'))
print(null_status_df.count())

null_status_df.show(truncate=False)



bad_status_df = null_status_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                                      regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                                      regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                                      regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                                      regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'),
                                      regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                                      regexp_extract('value', content_size_pattern, 1).cast('integer').alias('content_size'))
bad_status_df.show(truncate=False)

# Looks like the record itself is an incomplete record with no
# useful information, the best option would be to drop this record as follows!
print(logs_df.count())

logs_df = logs_df[logs_df['status'].isNotNull()]
print(logs_df.count())

exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()

"""
Handling nulls in HTTP content size

Again based on our previous regular expression and assuming you were able to solve it correctly, our original parsing regular expression for the content_size column was:

regexp_extract('value', r'\s(\d+)$', 1).cast('integer').alias('content_size')

Could there be missing data in our original dataset itself? Let's try and find out!
Q2: Your Turn: Find out the records in our base data frame with potential missing content sizes

    Use the r'\s\d+$' regex pattern with the rlike() function like we demonstrated in the previous example
    Remember to work on base_df since we are searching on the raw records NOT the parsed logs_df
    Find the total count of the records with missing content size in base_df using the count() function

"""
# TODO: Replace <FILL IN> with appropriate code
null_content_size_df = base_df.filter(~base_df['value'].rlike(r'\s\d+$'))
print(null_content_size_df.count())

# Q3: Your Turn: Display the top ten records of your data frame having missing content sizes
# TODO: Replace <FILL IN> with appropriate code

print(null_content_size_df.take(10))

"""
Assuming you were able to get to the missing records above, it is quite evident that the bad raw data records correspond to error responses, where no content was sent back and the server emitted a "-" for the content_size field.
Since we don't want to discard those rows from our analysis, let's impute or fill them to 0.
Q4: Your Turn: Fix the rows with null content_size
The easiest solution is to replace the null values in logs_df with 0 like we discussed earlier. The Spark DataFrame API provides a set of functions and fields specifically designed for working with null values, among them:
    fillna(), which fills null values with specified non-null values.
    na, which returns a DataFrameNaFunctions object with many functions for operating on null columns.
There are several ways to invoke this function. The easiest is just to replace all null columns with known values. But, for safety, it's better to pass a Python dictionary containing (column_name, value) mappings. That's what we'll do. A sample example from the documentation is depicted below"""

# TODO: Replace <FILL IN> with appropriate code
logs_df = logs_df.na.fill({'content_size': 0})

# Now assuming you were able to fill in the missing values successfully in the previous question,
# we should have no missing values \ nulls in our dataset.
# Let's verify this!
exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()

# Look at that, no missing values!
# Handling Temporal Fields (Timestamp)
# Now that we have a clean, parsed DataFrame, we have to parse the timestamp field into an actual timestamp. The Common Log Format time is somewhat non-standard. A User-Defined Function (UDF) is the most straightforward way to parse it.
from pyspark.sql.functions import udf

month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(text):
    """ Convert Common Log time format into a Python datetime object
    Args:
        text (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring time zone here. In a production application, you'd want to handle that.
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(text[7:11]),
      month_map[text[3:6]],
      int(text[0:2]),
      int(text[12:14]),
      int(text[15:17]),
      int(text[18:20])
    )
sample_ts = [item['timestamp'] for item in logs_df.select('timestamp').take(5)]
print(sample_ts)
[parse_clf_time(item) for item in sample_ts]

udf_parse_time = udf(parse_clf_time)
logs_df = logs_df.select('*', udf_parse_time(logs_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp')
logs_df.show(10, truncate=True)

print(logs_df.printSchema())

print(logs_df.limit(5).toPandas())

"""
Let's now cache logs_df since we will be using it extensively for our data analysis section in the next part!
Section 4 - Data Analysis on our Web Logs
Now that we have a DataFrame containing the parsed log file as a data frame, we can perform some interesting exploratory data analysis (EDA)
Example: Content Size Statistics
Let's compute some statistics about the sizes of content being returned by the web server. In particular, we'd like to know what are the average, minimum, and maximum content sizes.
We can compute the statistics by calling .describe() on the content_size column of logs_df. The .describe() function returns the count, mean, stddev, min, and max of a given column.
"""
content_size_summary_df = logs_df.describe(['content_size'])
print(content_size_summary_df.toPandas())

# Alternatively, we can use SQL to directly calculate these statistics.
# You can explore many useful functions within the pyspark.sql.functions module in the documentation.
# After we apply the .agg() function, we call toPandas() to extract and convert the result into a pandas dataframe which has better formatting on Jupyter notebooks

from pyspark.sql import functions as F
(logs_df.agg(F.min(logs_df['content_size']).alias('min_content_size'),
             F.max(logs_df['content_size']).alias('max_content_size'),
             F.mean(logs_df['content_size']).alias('mean_content_size'),
             F.stddev(logs_df['content_size']).alias('std_content_size'),
             F.count(logs_df['content_size']).alias('count_content_size'))
        .toPandas())

# Example: HTTP Status Code Analysis
# Next, let's look at the status code values that appear in the log. We want to know which status code values appear in the data and how many times.
# We again start with logs_df, then group by the status column, apply the .count() aggregation function, and sort by the status column.
status_freq_df = (logs_df
                     .groupBy('status')
                     .count()
                     .sort('status')
                     .cache())
print('Total distinct HTTP Status Codes:', status_freq_df.count())

# Total distinct HTTP Status Codes: 8
status_freq_pd_df = status_freq_df.toPandas()
print(status_freq_pd_df)

import matplotlib.pyplot as plt
status_freq_pd_df.plot(x='status', y='count', kind='bar')

log_freq_df = status_freq_df.withColumn('log(count)', F.log(status_freq_df['count']))
log_freq_df.show()


#Q5: Your Turn: Convert the log_freq_df to a pandas DataFrame and plot a bar chart displaying counts of each HTTP Status Code
# TODO: Replace <FILL IN> with appropriate code
content_size_summary_df.toPandas()
log_freq_pd_df = log_freq_df.toPandas()
log_freq_pd_df.plot(x='status', y='log(count)', kind='bar')

"""
Your Turn: Q6: Analyzing Frequent Hosts
Let's look at hosts that have accessed the server frequently. Try to get the count of total accesses by each host and then sort by the counts and display only the top ten most frequent hosts.
Hints:
    Your Spark DataFrame has a host column
    Get the counts per host which would make a count column
    Sort by the counts. Please check the documentation to see how to sort in reverse
    Remember only to get the top 10 rows from the aggregated dataframe and show them
"""


# TODO: Replace <FILL IN> with appropriate code

host_sum_df = logs_df.groupby('host').count().sort('count',ascending = False).limit(10)
host_sum_df.show(truncate=False)

host_sum_pd_df = host_sum_df.toPandas()
print(host_sum_pd_df.iloc[8]['host'])

"""
Looks like we have some empty strings as one of the top host names! This teaches us a valuable lesson to not just check for nulls but also potentially empty strings when data wrangling.
Your Turn: Q7: Display the Top 20 Frequent EndPoints
Now, let's visualize the number of hits to endpoints (URIs) in the log. To perform this task, start with our logs_df and group by the endpoint column, aggregate by count, and sort in descending order like the previous question. Also remember to show only the top 20 most frequently accessed endpoints
"""
# TODO: Replace <FILL IN> with appropriate code
paths_df = (logs_df.groupby('endpoint').count().sort('count',ascending = False).limit(20))

paths_pd_df = paths_df.toPandas()
print(paths_pd_df)

"""
Your Turn: Q8: Top Ten Error Endpoints
What are the top ten endpoints requested which did not have return code 200 (HTTP Status OK)?
Create a sorted list containing the endpoints and the number of times that they were accessed with a non-200 return code and show the top ten.
Think about the steps that you need to perform to determine which endpoints did not have a 200 return code (combination of filtering, grouping, sorting and selecting the top ten aggregated records)
"""


# TODO: Replace <FILL IN> with appropriate code

not200_df = logs_df.filter(logs_df.status != 200)
error_endpoints_freq_df = (not200_df
                               .groupby('endpoint').count().sort('count',ascending = False).limit(20))
error_endpoints_freq_df.show(truncate=False)

"""
Example: Number of Unique Daily Hosts
For an advanced example, let's look at a way to determine the number of unique hosts in the entire log on a day-by-day basis. This computation will give us counts of the number of unique daily hosts.
We'd like a DataFrame sorted by increasing day of the month which includes the day of the month and the associated number of unique hosts for that day.
Think about the steps that you need to perform to count the number of different hosts that make requests each day. Since the log only covers a single month, you can ignore the month. You may want to use the dayofmonth function in the pyspark.sql.functions module (which we have already imported as F.
host_day_df
A DataFrame with two columns
column 	explanation
host 	the host name
day 	the day of the month
There will be one row in this DataFrame for each row in logs_df. Essentially, we are just transforming each row of logs_df. For example, for this row in logs_df:
unicomp6.unicomp.net - - [01/Aug/1995:00:35:41 -0400] "GET /shuttle/missions/sts-73/news HTTP/1.0" 302 -
your host_day_df should have:
unicomp6.unicomp.net 1
"""
host_day_df = logs_df.select(logs_df.host,
                             F.dayofmonth('time').alias('day'))
host_day_df.show(5, truncate=False)

# host_day_distinct_df
# This DataFrame has the same columns as host_day_distinct_df, but with duplicate (day, host) rows removed.
host_day_distinct_df = (host_day_df
                          .dropDuplicates())
host_day_distinct_df.show(5, truncate=False)

daily_hosts_df = (host_day_distinct_df
                     .groupBy('day')
                     .count()
                     .sort("day"))
daily_hosts_df = daily_hosts_df.toPandas()
print(daily_hosts_df.T)

daily_hosts_df.plot(x='day', y='count', kind='line')
plt.show()

"""
Your Turn: Q9: Counting 404 Response Codes
Create a DataFrame containing only log records with a 404 status code (Not Found).
Make sure you cache() the not_found_df dataframe as we will use it in the rest of the exercises here.
How many 404 records are in the log?
"""
# TODO: Replace <FILL IN> with appropriate code

not_found_df = logs_df.filter(logs_df['status'] == 404).cache()
print(('Total 404 responses: {}').format(not_found_df.count()))

# Your Turn: Q10: Listing the Top Twenty 404 Response Code Endpoints
# Using the DataFrame containing only log records with a 404 response code that you cached in Q9, print out a list of the top twenty endpoints that generate the most 404 errors.
# Remember, top endpoints should be in sorted order
# TODO: Replace <FILL IN> with appropriate code

hosts_404_count_df = (not_found_df
                          .groupby('endpoint').count().sort('count',ascending=False).limit(20))
hosts_404_count_df.show(truncate=False)


# Your Turn: Q11: Visualizing 404 Errors per Day
# Let's explore the 404 records temporally now. Similar to the example showing the number of unique daily hosts, break down the 404 requests by day and get the daily counts sorted by day in errors_by_date_sorted_df.
#     Display the results as a pandas dataframe
#     Also visualize the same dataframe then as a line chart

# TODO: Replace <FILL IN> with appropriate code

errors_by_date_sorted_df = (not_found_df
                                .select(not_found_df.host, F.dayofmonth('time').alias('day')))

errors_by_date_sorted_df= errors_by_date_sorted_df.groupby('day').count().sort("day")
errors_by_date_sorted_df = errors_by_date_sorted_df.toPandas()
print(errors_by_date_sorted_df.T)

errors_by_date_sorted_df.plot(x='day', y='count', kind='line')
plt.show()
"""
Your Turn: Q12: Visualizing Hourly 404 Errors
Using the DataFrame not_found_df you cached in the Q10, group and sort by hour of the day in increasing order, to create a DataFrame containing the total number of 404 responses for HTTP requests for each hour of the day (midnight starts at 0).

    Remember to check out the hour function and use it (we have already imported pyspark.sql.functions as F earlier
    Output should be a bar graph displaying the total number of 404 errors per hour
"""
# TODO: Replace <FILL IN> with appropriate code

hourly_avg_errors_sorted_df = (not_found_df
                                   .select(not_found_df.host, F.hour('time').alias('hour')))
hourly_avg_errors_sorted_df = hourly_avg_errors_sorted_df.groupby('hour').count().sort("hour")
hourly_avg_errors_sorted_df = hourly_avg_errors_sorted_df.toPandas()
print(hourly_avg_errors_sorted_df)

hourly_avg_errors_sorted_df.plot(x='hour', y='count', kind='bar')
plt.show()

# Your Turn: Q13: Check data integrity before loading
# # TODO: Review the data frame you will like to store and load. Replace <FILL IN> with appropriate code
print(base_df.count())
print(base_df.columns)

# To save your dataframe in CSV file format,
# you call simply replace the name of the dataframe and assign file name in the following:

# Your Turn: Q14: Save your data as a CSV file
# TODO: Review the data frame you will like to store and load. Replace <FILL IN> with appropriate code

# base_df.write.save("base_dataframe", format = 'csv')
# # TODO: Check to see if you have stored and loaded the CSV file successfully by checking the first 5 rows. Replace <FILL IN> with appropriate code

# pyspark\
# 	.SparkContext\
# 	.textFile(r"C:\Users\chukw\PycharmProjects\Data-Wrangling-at-Scale-with-Spark\base_dataframe")\
# 	.take(5)

# Your Turn: Q15: Save your data as a JSON file
# TODO: Review the data frame you will like to store and load. Replace <FILL IN> with appropriate code

base_df.write.save("base_dataframe_json", format = 'json')
# TODO: Similarly, check the first 5 rows in the JSON file. Replace <FILL IN> with appropriate code

pyspark\
	.SparkContext\
	.textFile(r"C:\Users\chukw\PycharmProjects\Data-Wrangling-at-Scale-with-Spark\base_dataframe_json" )\
	.take(5)