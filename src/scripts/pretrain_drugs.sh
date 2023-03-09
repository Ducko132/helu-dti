# Pull in SMILES value from pubchem
python3 ../dataprep/drug_processor.py -d nr
cd ../KPGT/scripts
python3 preprocess_downstream_dataset.py --data_path ../datasets --dataset nr
python extract_features.py --config base --model_path ../models/pretrained/base/base.pth --data_path ../datasets --dataset nr
