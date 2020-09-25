cd damd_multiwoz
python -m spacy download en_core_web_sm
# preprocessing
python data_analysis.py
python preprocess.py
# setup python path
# type pwd inside damd_multiwoz to find out the path of damd_multiwoz folder
export PYTHONPATH='path of damd_multiwoz folder'

