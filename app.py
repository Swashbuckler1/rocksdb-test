from flask import Flask, request, json
import rocksdb, pandas as pd, os
from sklearn import model_selection, svm


def preprocess():
    csv_url = 'https://raw.githubusercontent.com/Swashbuckler1/Drug-Consumption-Classifier/main/drug_data.csv'
    df = pd.read_csv(csv_url)
    df_cpy = df.copy()
    df_cpy.replace(['CL0', 'CL1', 'CL0.1', 'CL0.10', 'CL0.11', 'CL0.12', 'CL0.2', 'CL0.3', 'CL0.4', 'CL0.5',
                    'CL0.6', 'CL0.7', 'CL0.8', 'CL0.9'], -1, inplace=True)  # Non-users

    for i in range(2, 7):
        df_cpy.replace('CL' + str(i), 1, inplace=True)  # Users
    df_cpy.replace(['CL2.1', 'CL2.2', 'CL5.1'], 1, inplace=True)

    return df_cpy


app = Flask(__name__)


@app.route("/")
def init():
    return "Connected to api successfully"

@app.route('/create', methods=['POST'])
def post_data():
    content_type = request.content_type

    if content_type == 'application/json':
        data = request.json
        label = data["label"]
        kernel = data["kernel"] # linear

        df = preprocess()
        features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore',
                    'Cscore', 'Impulsive', 'SS']

        # Choose from these in the request body
        labels = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Cocaine', 'Crack', 'Ecstasy',
                  'Heroin', 'Ketamine', 'Legalh', 'LSD',
                  'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
        X = df[features].values
        y = df[label].values
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
        clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)

        accuracy = str(accuracy).encode('utf-8')
        key = label.encode('utf-8')
        accuracy_db.put(key, accuracy)

        return accuracy.decode('utf-8'), 201
    return "Bad request", 400

@app.route('/get/<label>', methods=['GET'])
def get_accuracy(label):
    key = label.encode('utf-8')
    accuracy = accuracy_db.get(key)

    if accuracy:
        return accuracy.decode('utf-8'), 200

    return "Not found", 404


if __name__ == '__main__':
    opts = rocksdb.Options()
    opts.create_if_missing = True
    opts.max_open_files = 300000
    opts.write_buffer_size = 67108864
    opts.max_write_buffer_number = 3
    opts.target_file_size_base = 67108864

    opts.table_factory = rocksdb.BlockBasedTableFactory(
        filter_policy=rocksdb.BloomFilterPolicy(10),
        block_cache=rocksdb.LRUCache(2 * (1024 ** 3)),
        block_cache_compressed=rocksdb.LRUCache(500 * (1024 ** 2)))
    opts.wal_dir = '/root/rocksdb-test/Accuracy.db'
    #os.system('rm Accuracy.db/LOCK')
    accuracy_db = rocksdb.DB('Accuracy.db', opts)

    app.run(debug=True, port=5001)

    accuracy_db.close()
