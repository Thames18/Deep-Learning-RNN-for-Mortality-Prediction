import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset,Dataset
from functools import reduce

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	df = pd.read_csv(path)
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	#df.y = df.y.apply(lambda x: x-1)
	#y = df.iloc[:, -1].values
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	#X = df.iloc[:, :-1].to_numpy('float32')
	#y = df['y'].to_numpy('int64')
	assert 'y' in df.columns
	y = df['y'].astype(np.int64).to_numpy() - 1
	assert np.all((0 <= y) & (y < 5))
	#X = df.drop( columns=['y']).to_numpy(dtype=np.float32)
	#X_t = torch.from_numpy(data.astype('float32'))
	#y_t = torch.from_numpy(y).long()

	if model_type == 'MLP':
		data = torch.tensor(df.drop('y', axis = 1).values.astype(np.float32))
		target = torch.tensor((df['y'] - 1).values )
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = df.loc[:, 'X1' : 'X178'].values
		target = torch.tensor((df['y'] - 1).values)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1), target)
	elif model_type == 'RNN':
		data = df.loc[:, 'X1' : 'X178'].values
		target = torch.tensor((df['y'] - 1).values)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	if not seqs:
		return 2
	#skip empty patients
	feature = reduce(lambda acc, p: acc + p if p else acc, seqs, [])
	if not feature:
		return 2
	#skip empty visits
	features = reduce(lambda acc, v: acc + v if v else acc, feature, [])
	if not features:
		return 2
	return max(features) + 2


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""
		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")
		self.labels = labels
		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		answ = []
		for a in seqs:
			x = len(a)
			y = num_features
			mat = np.zeros((x, y))
			i = 0
			for step in a:
				step_indi = step if (step, (list, tuple, np.ndarray) )else [step]
				for y in step_indi:
					mat[i,y] = 1
				i+=1
			answ.append(mat)
		self.seqs = answ
    			
	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each 
	line = []
	i =0
	for x,y in batch:
			line.append((x.shape[0], i))
			i+=1
	line.sort(key=lambda x:x[0], reverse=True)
	line_row = line[0][0]
	line_col = batch[0][0].shape[1]
	line_seq=[]
	line_length= []
	line_label =[]

	for i in range(len(line)):
			idx = line[i][1]
			pat = batch[idx]
			line_label.append(pat[1])
			line_length.append(pat[0].shape[0])
			d = np.zeros((line_row, line_col))
			d[0: pat[0].shape[0], 0:pat[0].shape[1]] = pat[0]
			line_seq.append(d)

	seqs_tensor = torch.FloatTensor(line_seq)
	lengths_tensor = torch.LongTensor(line_length)
	labels_tensor = torch.LongTensor(line_label)

	return (seqs_tensor, lengths_tensor), labels_tensor
