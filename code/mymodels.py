import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as fun
from torch.autograd import Variable
##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		#self.input =nn.Linear(178,16, bias=True) #unimprovved version
		#self.hidden = nn.Linear(16, 5, bias=True) #unimprovved version
		self.input =nn.Linear(178,60, bias=True) #improvved version
		self.hidden = nn.Linear(60, 5, bias=True) #improvved version
		self.improve1 = nn.BatchNorm1d(178) #improvved version
		self.improve2 = nn.Dropout(p= 0.5) #improvved version


	def forward(self, x):
		#x = torch.sigmoid(self.input(x)) #unimprovved version
		#x = self.hidden(x) #unimprovved version
		x = torch.sigmoid(self.improve2(self.input(self.improve1(x))))#improvved version
		x = self.hidden(x) #improvved version
		
		return x

class MyCNN(nn.Module):
	def __init__(self, in_len = 178, num_clas = 5):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(1, 16, 5, padding = 2) #unimprovved version
		self.bn1   = nn.BatchNorm1d(16) #unimprovved version
		self.conv2 = nn.Conv1d(16,num_clas, kernel_size = 5, padding = 2) #unimprovved 
		#self.conv1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=7 , padding = 3),nn.BatchNorm1d(32), nn.GELU()) #improvved version
		#self.bn1   = nn.Sequential(nn.Conv1d(32, 64, kernel_size=5 , padding = 2),nn.BatchNorm1d(64), nn.GELU())#improvved version
		#self.conv2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=5 , padding = 2),nn.BatchNorm1d(128), nn.GELU()) #improvved version
		#self.improve1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1 , padding = 2),nn.BatchNorm1d(64), nn.GELU()) #improvved version
		#self.arrange = nn.Conv1d(64, num_clas, kernel_size=1) #improvved version
		#self.gap = nn.AdaptiveAvgPool1d(1) #improvved version

	def forward(self, x):
		x = fun.relu(self.bn1(self.conv1(x))) #unimprovved version
		x = self.conv2(x) #unimprovved version
		x = x.mean(dim= -1) #unimprovved version
		#x = self.conv1(x) #improvved version
		#x = self.bn1(x) #improvved version
		#x = self.conv2(x) #improvved version
		#x = self.improve1(x) #improvved version
		#x = self.arrange(x) #improvved version
		#x = self.gap(x).squeeze(-1) #improvved version

		return x


class MyRNN(nn.Module):
	def __init__(self, num_clas = 5):
		super(MyRNN, self).__init__()
		#self.rnn = nn.GRU(input_size = 1, hidden_size = 16, num_layers = 1, batch_first=True) #unimproved version
		#self.fc  = nn.Linear(in_features=16, out_features=5) #unimproved version
		self.rnn = nn.GRU(input_size = 1, hidden_size = 16, num_layers = 1, batch_first=True, dropout= 0.2) #improved version
		self.fc  = nn.Linear(in_features=16, out_features=5) #improved version
		#self.rnn = nn.GRU(input_size = 1, hidden_size = 64, num_layers = 2, batch_first=True) #improved version
		#self.attention1   = nn.Linear(64 , 64, bias = True)#improved version
		#self.attention2 = nn.Linear( 64, 1 , bias = True) #improved version
		#self.drop = nn.Dropout(p = 0.2) #improved version
		#self.fc = nn.Linear(64, num_clas) #improved version
		#self.norm = nn.LayerNorm(1) #improved version

	def forward(self, x):
		#x, _ = self.rnn(x) #unimproved version 
		#x = self.fc(x[:, -1, :]) #unimproved version
		x, _ = self.rnn(x) #improved version 
		x = fun.relu(x[:, -1, :]) #improved version
		x = self.fc(x) #improved version
		#x = self.norm(x) #nimproved version
		#out, _ = self.rnn(x) #nimproved version
		#s = self.attention2(torch.tanh(self.attention1(out))).squeeze(-1) #nimproved version
		#w = fun.softmax(s, dim=1) #nimproved version
		#vec = torch.bmm(w.unsqueeze(1), out).squeeze(1) #nimproved version
		#vec = self.drop(vec) #nimproved version
		#log = self.fc(vec) #nimproved version

		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		#self.input = nn.Linear(in_features=dim_input, out_features=32) #unimproved version
		#self.rnn = nn.GRU(input_size = 32, hidden_size=16, num_layers=1, batch_first=True) #unimproved version
		#self.input2 = nn.Linear(in_features=16, out_features=2) #unimproved version
		self.batch_first=True #improved version
		self.input = nn.Sequential(nn.Dropout(0.6), nn.Linear(dim_input, 128, bias=False), nn.Dropout(0.)) #improved version
		self.layer1 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True) #improved version
		self.layer11 = nn.Linear(in_features=128, out_features=1) #improved version
		self.layer11.bias.data.zero_() #improved version
		self.layer2 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True) #improved version
		self.layer21 = nn.Linear(in_features=128, out_features=128) #improved version
		self.layer21.bias.data.zero_() #improved version
		self.layero = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features= 128, out_features=2)) #improved version
		self.layero[1].bias.data.zero_() #improved version
		

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		#seq, lengths = input_tuple #unimproved version
		#seq = torch.tanh(self.input(seq)) #unimproved version
		#lengths_cpu = lengths.detach().cpu()
		#seq = pack_padded_sequence(seq, lengths_cpu, batch_first=True, enforce_sorted=False ) #unimproved version
		#seq, _ = self.rnn(seq) #unimproved version
		#seq, _ = pad_packed_sequence(seq, batch_first= True) #unimproved version 
		#seq = self.input2(seq[:, -1, :]) #unimproved version 
		#improved version
		seq, lengths = input_tuple #improved version
		a1, b1 = seq.size(0), seq.size(1) #improved version
		x = self.input(seq) #improved version
		bi = pack_padded_sequence(x, lengths, batch_first=self.batch_first) #improved version
		a , _ = self.layer1(bi) #improved version
		b , _ =  pad_packed_sequence(a, batch_first=self.batch_first) #improved version
		c = Variable(torch.FloatTensor([[1.0 if i < lengths[idx] else 0.0 for i in range(b1)] for idx in range(a1)]).unsqueeze(2), requires_grad=False) #improved version
		d = self.layer11(b) #improved version
		def maxv(x, c): #improved version
				expo = torch.exp(x) #improved version
				ms = expo * c #improved version
				st = torch.sum(ms, dim=1, keepdim=True) #improved version
				return ms/st #improved version 
		negative = torch.finfo(d.dtype).min #improved version
		d_mask = d.masked_fill( c== 0, negative) #improved version
		alpha = torch.softmax(d_mask, dim=1) #improved version
		e, _ = self.layer2(bi) #improved version
		out, _ = pad_packed_sequence(e, batch_first=self.batch_first) #improved version
		out = torch.tanh(self.layer21(out)) #improved version
		seq = torch.bmm(torch.transpose(alpha, 1, 2), out * x).squeeze(1) #improved version
		seq = self.layero(seq) #improved version
		return seq