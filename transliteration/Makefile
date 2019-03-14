download_data data/raw/en-ja.txt:
	mkdir -p data/raw
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.txt -O data/raw/muse-en-ja.txt
	wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b -O data/raw/cmudict.txt
	wget https://raw.githubusercontent.com/eob/english-japanese-transliteration/master/data/japan_english_utf8.txt -O data/raw/eob-pairs.txt
extract_pairs data/processed/katakana_pairs.csv: download_data
	mkdir -p data/processed
	python scripts/extract_pairs_from_muse_dict.py data/raw/muse-en-ja.txt --output-file data/processed/muse_katakana_pairs.csv
	python scripts/extract_pairs_from_cmu_dict.py data/raw/cmudict.txt --output-file data/processed/cmu_pronunciation_pairs.csv
	python scripts/extract_pairs_from_eob_dict.py data/raw/eob-pairs.txt --output-file data/processed/eob_katakana_pairs.csv

make_splits data/split/pairs_train.csv: extract_pairs
	mkdir -p data/split
	python scripts/create_splits.py --output_prefix data/split/muse_pairs --splits train/8 valid/1 test/1 --input_files data/processed/muse_katakana_pairs.csv
	python scripts/create_splits.py --output_prefix data/split/cmu_pairs --splits train/8 valid/1 test/1 --input_files data/processed/cmu_pronunciation_pairs.csv
	python scripts/create_splits.py --output_prefix data/split/eob_pairs --splits train/8 valid/1 test/1 --input_files data/processed/eob_katakana_pairs.csv

tf_datasets: make_splits
	mkdir -p data/tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/muse_pairs_train.csv --output_file data/tfrecord/muse_train.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/muse_pairs_valid.csv --output_file data/tfrecord/muse_valid.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/muse_pairs_test.csv --output_file data/tfrecord/muse_test.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/cmu_pairs_train.csv --output_file data/tfrecord/cmu_train.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/cmu_pairs_valid.csv --output_file data/tfrecord/cmu_valid.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/cmu_pairs_test.csv --output_file data/tfrecord/cmu_test.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/eob_pairs_train.csv --output_file data/tfrecord/eob_train.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/eob_pairs_valid.csv --output_file data/tfrecord/eob_valid.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/eob_pairs_test.csv --output_file data/tfrecord/eob_test.tfrecord
