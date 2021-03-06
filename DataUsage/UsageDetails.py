import collections
import tensorflow as tf

#===============================================================================
# census_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
# census_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
# census_train_path = tf.contrib.keras.utils.get_file('census.train', census_train_url)
# census_test_path = tf.contrib.keras.utils.get_file('census.test', census_test_url)
#===============================================================================

# Provide default values for each of the CSV columns
# and a header at the same time.
csv_defaults = collections.OrderedDict([
    ('client',['']),
    ('User',['']),
    ('Timestamp',['']),
    ('APPLICATION_NAME',['']),
    ('VIEW_NAME',['']),
    ('BUTTON_NAME',['']),
    ('LANGUAGE',['']),
    ('PLATFORM',['']),
    ('USERAGENT',['']),
    ('VENDOR',['']),
    ('VENDOR_SUB',['']),
    ('COLORDEPTH',['']),
    ('HEIGHT',['']),
    ('ORIENTATION',['']),
    ('PIXELDEPTH',['']),
    ('WIDTH',['']),
    ('HASH',['']),
    ('HOST',['']),
    ('HOSTNAME',['']),
    ('HREF',['']),
    ('ORIGIN',['']),
    ('PORT',['']),
    ('PROTOCOL',['']),
    ('OS',['']),
    ('ZSYSTEM',['']),
    ('FIELD1',['']),
    ('FIELD2',['']),
    ('FIELD3',['']),
    ('FIELD4',['']),
    ('FIELD5',['']),
    ('PATH',[''])
])
# Decode a line from the CSV.
def csv_decoder(line):
    """Convert a CSV row to a dictionary of features."""
    parsed = tf.decode_csv(line, list(csv_defaults.values()))
    return dict(zip(csv_defaults.keys(),parsed))

# the train file has an extra empty line at the end.
# we'll use this method to filter that out.
def filter_empty_lines(line):
    return tf.not_equal(tf.size(tf.string_split([line], ',').values),0)

def create_train_input_fn(path):
    def input_fn():
        dataset = (
            tf.contrib.data.TextLineDataset(path) # create a dataset from a line
                #.filter(filter_empty_lines)
                .map(csv_decoder) #parse each row
                .shuffle(buffer_size=10000) # shuffle the dataset
                .repeat() # repeat indefinitely
                .batch(4000)) # batch the data
        # create iterator
        columns = dataset.make_one_shot_iterator().get_next()
        
        # separate the label and convert it to true/false
        income = tf.equal(columns.pop('FIELD3'),"Posted")
        return columns, income
    return input_fn

def create_test_input_fn(path):
    def input_fn():
        dataset = (
            tf.contrib.data.TextLineDataSet(path)
            #.skip(1) # the test file has a strange first line, we want to ignore this.
            #.filter(filter_empty_lines)
            .map(csv_decoder)
            .batch(32))
        # create iterator
        columns = dataset.make_one_shot_iterator().get_next()
        
        # separate the label and convert it to true/false
        income = tf.equal(columns.pop('FIELD3'), 'Posted')
        return columns, income
    return input_fn

# Here's code you can test the Dataset input function
train_input_fn = create_train_input_fn('ZUsageTrain.csv')
next_batch = train_input_fn()
with tf.Session() as sess:
    features, label = sess.run(next_batch)
    print(features['APPLICATION_NAME'])
    print(label)
    
    print()
    
    features, label = sess.run(next_batch)
    print(features['PATH'])
    print(label)

train_input_fn = create_train_input_fn('ZUsageTrain.csv')
test_input_fn = create_train_input_fn('ZUsageTest.csv')
feature_columns = [
    tf.feature_column.numeric_column('HASH'),
    ]
estimator = tf.estimator.DNNClassifier(hidden_units = [256, 128, 64],
                                       feature_columns = feature_columns,
                                       n_classes = 2,
                                       # creating a new folder in case you haven't cleared
                                       # the old one yet
                                       model_dir = 'graphs_datasets/dnn')

estimator.train(train_input_fn, steps=100)
estimator.evaluate(test_input_fn, steps=100)