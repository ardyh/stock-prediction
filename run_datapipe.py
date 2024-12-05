from __future__ import print_function  # For Python 2/3 compatibility
from DataPipe import DataPipe
import json
import numpy as np

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Option 2: Generate fixed-size batches
def extract_batches(phase='train', num_batches=10, output_file='extracted_batches.json'):
    data_pipe = DataPipe()
    batch_generator = data_pipe.batch_gen(phase)
    all_batches = []
    
    for i in range(num_batches):
        try:
            batch = next(batch_generator)
            batch_dict = {
                'batch_size': int(batch['batch_size']),
                'stock_batch': batch['stock_batch'],
                'T_batch': batch['T_batch'],
                'y_batch': batch['y_batch'],
                'main_mv_percent_batch': batch['main_mv_percent_batch'],
                'price_batch': batch['price_batch'],
                'word_batch': batch['word_batch'],
                'n_msgs_batch': batch['n_msgs_batch'],
                'n_words_batch': batch['n_words_batch']
            }
            all_batches.append(batch_dict)
            print("Processed batch {}/{}".format(i+1, num_batches))  # Changed f-string to .format()
        except StopIteration:
            print("Ran out of data")
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(all_batches, f, cls=NumpyEncoder)
        
    return all_batches

def extract_data(phase='train', output_file='extracted_data.json', debug=False):
    data_pipe = DataPipe()
    all_batches = []
    
    # Option 1: Generate by stocks
    for batch in data_pipe.batch_gen_by_stocks(phase):
        # Remove any tensorflow-specific elements if needed
        batch_dict = {
            's': batch['s'],
            'batch_size': int(batch['batch_size']),  # Convert from numpy to native types
            'stock_batch': batch['stock_batch'],
            'main_target_date_batch': batch['main_target_date_batch'],
            'T_batch': batch['T_batch'],
            'y_batch': batch['y_batch'],
            'main_mv_percent_batch': batch['main_mv_percent_batch'],
            'price_batch': batch['price_batch'],
            'word_batch': batch['word_batch'],
            'texts_batch': batch['texts_batch'],
            'n_msgs_batch': batch['n_msgs_batch'],
            'n_words_batch': batch['n_words_batch']
        }
        all_batches.append(batch_dict)
        print("Processed stock: {}".format(batch['s']))  # Changed f-string to .format()
        if debug:
            break
            #  print(all_batches)
            # return

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(all_batches, f, cls=NumpyEncoder)
        
    return all_batches

if __name__ == "__main__":
    # Choose one of these:
    # data = extract_data(phase='whole', output_file='../data/analysis/stock_data_v2.json', debug=True)
    data = extract_data(phase='whole', output_file='../data/analysis/stock_data.json')
    data = extract_data(phase='train', output_file='../data/analysis/stock_data_train.json')
    data = extract_data(phase='test', output_file='../data/analysis/stock_data_test.json')
    # or
    # batches = extract_batches(phase='train', num_batches=100, output_file='../data/analysis/batch_data.json')
