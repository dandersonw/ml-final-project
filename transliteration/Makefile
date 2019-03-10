download_data data/raw/en-ja.txt:
	mkdir -p data/raw
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.txt -O data/raw/en-ja.txt

extract_pairs data/processed/katakana_pairs.csv: download_data
	mkdir -p data/processed
	python scripts/extract_pairs_from_muse_dict.py data/raw/en-ja.txt --output-file data/processed/katakana_muse_pairs.csv


make_splits data/split/pairs_train.csv: extract_pairs
	mkdir -p data/split
	python scripts/create_splits.py --output_prefix data/split/pairs --splits train/8 valid/1 test/1 --input_files data/processed/katakana_muse_pairs.csv


tf_datasets: make_splits
	mkdir -p data/tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/pairs_train.csv --output_file data/tfrecord/train.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/pairs_valid.csv --output_file data/tfrecord/valid.tfrecord
	python scripts/create_tf_dataset.py --input_file data/split/pairs_test.csv --output_file data/tfrecord/test.tfrecord
