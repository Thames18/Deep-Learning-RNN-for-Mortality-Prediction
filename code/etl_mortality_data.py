import os
import pickle
import pandas as pd
import numpy as np

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

PATH_TRAIN = r"C:\Users\musta\Desktop\CSE-6250\HW4_2025fall\data\mortality\train"
PATH_VALIDATION = r"C:\Users\musta\Desktop\CSE-6250\HW4_2025fall\data\mortality\validation"
PATH_TEST = r"C:\Users\musta\Desktop\CSE-6250\HW4_2025fall\data\mortality/test"
PATH_OUTPUT = r"C:\Users\musta\Desktop\CSE-6250\HW4_2025fall\data\mortality\processed"


def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	# TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# TODO: Read the homework description carefully.
	m = 3
	if str(icd9_object)[0].isalpha() and str(icd9_object).startswith('V') ==True:
		m = 4
	if m >= len(str(icd9_object)):
		m= len(str(icd9_object))
	converted = str(icd9_object)[0:1]

	return converted


def build_codemap(df_icd9, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	df_digits = df_icd9['ICD9_CODE'].apply(transform)
	df_digits = df_digits.dropna()
	df_digits = df_digits[df_digits.str.len() > 0]
	df_digits_unique = sorted(df_digits.unique())
	codemap = {code: i for i, code in enumerate(df_digits_unique)}
	return codemap


def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# TODO: 1. Load data from the three csv files
	# TODO: Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_admissions = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
	df_diagnoses = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
	# TODO: 2. Convert diagnosis code in to unique feature ID.
	# TODO: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
	df_diagnoses['ICD9_CODE'] = df_diagnoses['ICD9_CODE'].transform(transform)
	# TODO: 3. Group the diagnosis codes for the same visit.
	group_diagnoses = df_diagnoses.groupby(['HADM_ID'])
	# TODO: 4. Group the visits for the same patient.
	group_admissions = df_admissions.groupby(['SUBJECT_ID'])
	# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# TODO: Visits for each patient must be sorted in chronological order.
	seq_data = []
	# TODO: 6. Make patient-id List and label List also.
	# TODO: The order of patients in the three List output must be consistent.
	patient_ids = []
	labels = []
	#seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]
	for id, values in group_admissions:
		seq = []
		values = values.sort_values('ADMITTIME')
		patient_ids.append(id)
		lab = np.array(df_mortality.loc[df_mortality['SUBJECT_ID'] == id]['MORTALITY'])[0]
		labels.append(lab)
		for x in range(len(values)):
			diagnose = np.array(group_diagnoses.get_group(values.iloc[x, :]['HADM_ID'])['ICD9_CODE'])
			diagnoses = list(filter(lambda a: a in codemap.keys(), diagnose))
			seq.append(list(map(lambda a: codemap[a], diagnoses)))
		seq_data.append(seq)
	return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")

if __name__ == '__main__':
	main()