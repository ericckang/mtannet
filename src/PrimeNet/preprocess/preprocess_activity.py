import os
from PrimeNet import utils
import torch
from torchvision.datasets.utils import download_url
from sklearn import model_selection


class PersonActivity(object):
	urls = [
		'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt',
	]

	tag_ids = [
		"010-000-024-033", #"ANKLE_LEFT",
		"010-000-030-096", #"ANKLE_RIGHT",
		"020-000-033-111", #"CHEST",
		"020-000-032-221" #"BELT"
	]
	
	tag_dict = {k: i for i, k in enumerate(tag_ids)}

	label_names = [
		 "walking",
		 "falling",
		 "lying down",
		 "lying",
		 "sitting down",
		 "sitting",
		 "standing up from lying",
		 "on all fours",
		 "sitting on the ground",
		 "standing up from sitting",
		 "standing up from sit on grnd"
	]

	#label_dict = {k: i for i, k in enumerate(label_names)}

	#Merge similar labels into one class
	label_dict = {
		"walking": 0,
		 "falling": 1,
		 "lying": 2,
		 "lying down": 2,
		 "sitting": 3,
		 "sitting down" : 3,
		 "standing up from lying": 4,
		 "standing up from sitting": 4,
		 "standing up from sit on grnd": 4,
		 "on all fours": 5,
		 "sitting on the ground": 6
		 }



	def __init__(self, root, download=False,
		reduce='average', max_seq_length = 50,
		n_samples = None, device = torch.device("cpu")):

		self.root = root
		self.reduce = reduce
		self.max_seq_length = max_seq_length

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if device == torch.device("cpu"):
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
		else:
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

		if n_samples is not None:
			self.data = self.data[:n_samples]



	def download(self):
		if self._check_exists():
			return

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)


		def save_record(records, record_id, tt, vals, mask, labels):
			tt = torch.tensor(tt).to(self.device)

			vals = torch.stack(vals)
			mask = torch.stack(mask)
			labels = torch.stack(labels)

			# flatten the measurements for different tags
			vals = vals.reshape(vals.size(0), -1)
			mask = mask.reshape(mask.size(0), -1)
			assert(len(tt) == vals.size(0))
			assert(mask.size(0) == vals.size(0))
			assert(labels.size(0) == vals.size(0))

			#records.append((record_id, tt, vals, mask, labels))

			seq_length = len(tt)
			# split the long time series into smaller ones
			offset = 0
			slide = self.max_seq_length // 2

			while (offset + self.max_seq_length < seq_length):
				idx = range(offset, offset + self.max_seq_length)
	
				first_tp = tt[idx][0]
				records.append((record_id, tt[idx] - first_tp, vals[idx], mask[idx], labels[idx]))
				offset += slide


		for url in self.urls:
			filename = url.rpartition('/')[2]
			download_url(url, self.raw_folder, filename, None)

			print('Processing {}...'.format(filename))

			dirname = os.path.join(self.raw_folder)
			records = []
			first_tp = None

			for txtfile in os.listdir(dirname):
				with open(os.path.join(dirname, txtfile)) as f:
					lines = f.readlines()
					prev_time = -1
					tt = []

					record_id = None
					for l in lines:
						cur_record_id, tag_id, time, date, val1, val2, val3, label = l.strip().split(',')
						value_vec = torch.Tensor((float(val1), float(val2), float(val3))).to(self.device)
						time = float(time)

						if cur_record_id != record_id:
							if record_id is not None:
								save_record(records, record_id, tt, vals, mask, labels)
							tt, vals, mask, nobs, labels = [], [], [], [], []
							record_id = cur_record_id
						
							tt = [torch.zeros(1).to(self.device)]
							vals = [torch.zeros(len(self.tag_ids),3).to(self.device)]
							mask = [torch.zeros(len(self.tag_ids),3).to(self.device)]
							nobs = [torch.zeros(len(self.tag_ids)).to(self.device)]
							labels = [torch.zeros(len(self.label_names)).to(self.device)]
							
							first_tp = time
							time = round((time - first_tp)/ 10**5)
							prev_time = time
						else:
							# for speed -- we actually don't need to quantize it in Latent ODE
							time = round((time - first_tp)/ 10**5) # quatizing by 100 ms. 10,000 is one millisecond, 10,000,000 is one second

						if time != prev_time:
							tt.append(time)
							vals.append(torch.zeros(len(self.tag_ids),3).to(self.device))
							mask.append(torch.zeros(len(self.tag_ids),3).to(self.device))
							nobs.append(torch.zeros(len(self.tag_ids)).to(self.device))
							labels.append(torch.zeros(len(self.label_names)).to(self.device))
							prev_time = time

						if tag_id in self.tag_ids:
							n_observations = nobs[-1][self.tag_dict[tag_id]]
							if (self.reduce == 'average') and (n_observations > 0):
								prev_val = vals[-1][self.tag_dict[tag_id]]
								new_val = (prev_val * n_observations + value_vec) / (n_observations + 1)
								vals[-1][self.tag_dict[tag_id]] = new_val
							else:
								vals[-1][self.tag_dict[tag_id]] = value_vec

							mask[-1][self.tag_dict[tag_id]] = 1
							nobs[-1][self.tag_dict[tag_id]] += 1

							if label in self.label_names:
								if torch.sum(labels[-1][self.label_dict[label]]) == 0:
									labels[-1][self.label_dict[label]] = 1
						else:
							assert tag_id == 'RecordID', 'Read unexpected tag id {}'.format(tag_id)
					save_record(records, record_id, tt, vals, mask, labels)
			
			torch.save(
				records,
				os.path.join(self.processed_folder, 'data.pt')
			)
				
		print('Done!')



	def _check_exists(self):
		for url in self.urls:
			filename = url.rpartition('/')[2]
			if not os.path.exists(
				os.path.join(self.processed_folder, 'data.pt')
			):
				return False
		return True



	@property
	def raw_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'raw')



	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')



	@property
	def data_file(self):
		return 'data.pt'



	def __getitem__(self, index):
		return self.data[index]



	def __len__(self):
		return len(self.data)



	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Max length: {}\n'.format(self.max_seq_length)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str



def get_person_id(record_id):
	# The first letter is the person id
	person_id = record_id[0]
	person_id = ord(person_id) - ord("A")
	return person_id



def get_train_val_test_data(args, device):
	n_samples = min(10000, args.n)
	dataset_obj = PersonActivity('PersonActivity', download=False, n_samples=n_samples, device=device)

	# train-val-test split (64%, 16%, 20%)
	train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8, random_state=42, shuffle=True)
	train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8, random_state=11, shuffle=True)
	# train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
	# test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

	activity = True
	print('classify: ', args.classif)
	print('activity: ', activity)

	train_data_combined = utils.variable_time_collate_fn(train_data, device, classify=args.classif, activity=activity)
	val_data_combined = utils.variable_time_collate_fn(val_data, device, classify=args.classif, activity=activity)
	test_data_combined = utils.variable_time_collate_fn(test_data, device, classify=args.classif, activity=activity)
	
	print(train_data_combined[1].sum(), val_data_combined[1].sum(), test_data_combined[1].sum())
	
	print(train_data_combined[0].size(), train_data_combined[1].size())
	print(val_data_combined[0].size(), val_data_combined[1].size())
	print(test_data_combined[0].size(), test_data_combined[1].size())
	
	# save the data and labels here
	# code removed to avoid accidental overwriting of the current data split