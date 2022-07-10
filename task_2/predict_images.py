'''
	predict_images.py
	Data Analytics 2 - WWU
	Dimitri Silva

	args:
		imgs_path (str)			-> 	Absolute path to folder containing PNG images to be predicted. 
									Default is the same path this python file is located.
		path_to_save_csvs (str)	->	Absolute path where to save the output CSVs. 
									Default is the same as imgs_path.
		path_to_model (str)		->	Absolute path to model checkpoint (suffix .pt).
									Default is the same path this python file is located
									plus model_ckpt.pt.
		plot_predictions (int)	->	Whether to plot images with predictions or not. 
									This value must be 0 (default) or 1. 

	run: 
		python3 predict_images.py imgs_path path_to_save_csvs path_to_model plot_predictions

		exemple:
			python3 
			predict_images.py 
			/home/dimi/validation_data/02_validation_data_images/
			/home/dimi/validation_data/02_validation_data_images/csv_results/ 
			/home/dimi/DA2Group10/task_2/results/model_checkpoints/model_checkpoint_all_layers_round_5-v1.pt 
			1
	
	obs:
		Need to have satellighte installed! 
		A requirements.pip file will be provided. This code was tested using Python 3.8.10.
'''

# import packages
import sys
import satellighte as sat
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from matplotlib.image import imsave
from PIL import Image
from pathlib import Path
from torchvision import transforms as tt
from torchvision.transforms.transforms import Resize

# defining function to get args or set defaults
def get_args():

	# imgs_path
	try:
		imgs_path = Path(r'{}'.format(sys.argv[1]))
	except:
		imgs_path = Path(__file__).resolve().parent
		print('\nCouldnt find imgs_path in sys.argv[1]. Using default:', imgs_path)
	
	# path_to_save_csvs
	try:
		path_to_save_csvs = Path(r'{}'.format(sys.argv[2]))
	except:
		path_to_save_csvs = imgs_path
		print('\nCouldnt find path_to_save_csvs (str) in sys.argv[2]. Using default:', path_to_save_csvs)

	# path_to_model
	try:
		path_to_model = Path(r'{}'.format(sys.argv[3]))
	except:
		path_to_model = Path(__file__).resolve().parent.joinpath('model_ckpt.pt')
		print('\nCouldnt find path_to_model (str) in sys.argv[3]. Using default:', path_to_model)

	# plot_predictions
	try:
		plot_predictions = int(sys.argv[4])
	except:
		plot_predictions = 0
		print('\nCouldnt find plot_predictions (int) in sys.argv[4]. Using default:', plot_predictions)

	assert imgs_path.is_dir(), 'Couldnt find imgs_path'
	assert path_to_save_csvs.is_dir(), 'Couldnt find path_to_save_csvs'
	assert path_to_model.is_file(), 'Couldnt find model checkpoint'
	assert plot_predictions == 0 or plot_predictions == 1, 'Plot_predictions must be either 0 or 1'

	return imgs_path, path_to_save_csvs, path_to_model, plot_predictions

# defining function to load model and its parameters
def load_model_and_params(path_to_model):

	# loading the model
	print('')
	if torch.cuda.is_available():
		model = torch.load(path_to_model)
	else:
		print('torch.cuda.is_available() returned False. Trying to load model using CPU...')
		model = torch.load(path_to_model, map_location=torch.device('cpu'))

	# setting to eval mode
	model.eval()
	print('The model was loaded successfully!')

	# creating transform
	val_tt = tt.Compose([tt.ToPILImage(), tt.Resize((240, 240))])

	# mapping the labels of pretrained model to our needs
	labels_map = {
		'Industrial':           'trampoline',
		'Highway':              'solar',
		'HerbaceousVegetation': 'pools',
		'AnnualCrop':           'background',
		'Forest':               'ponds'
	}

	return model, val_tt, labels_map

# defining function to get image and yield its windows
def sliding_window(img, size, step):
	step = int(step*size)
	for i in np.arange(0, img.shape[0], step):
		for j in np.arange(0, img.shape[1], step):
			x_min = i 
			x_max = i + size
			y_min = j
			y_max = j + size
			if x_max > img.shape[0]:
				x_max = img.shape[0] - 1
				x_min = x_max - size 
			if y_max > img.shape[1]:
				y_max = img.shape[1] - 1
				y_min = y_max - size 
			yield (
				img[y_min:y_max, x_min:x_max, :], 
				int((y_max+y_min)/2),
				int((x_max+x_min)/2),
				y_min,
				x_min,
				y_max,
				x_max,
			)

# defining function to get image, use the previous func to get windows, 
# and predict each window. This func returns a pd.DataFrame containing
# the first predictions
def slide_and_predict(img, model, val_tt, labels_map, size, step, columns):
	model.eval()
	windows = sliding_window(img, size, step)
	preds   = []

	for window, y_target, x_target, y_upper_left, x_upper_left, y_lower_right, x_lower_right in windows:
		try:
			img_tt    = np.array(val_tt(window))[:, :, :3]
			dict_pred = model.predict(img_tt)[0]
			pred      = max(dict_pred, key=dict_pred.get)
			pred_map  = labels_map.get(pred, 'background')
			if pred_map != 'background':
				preds.append([
					pred_map,
					y_target,
					x_target,
					y_upper_left,
					x_upper_left,
					y_lower_right,
					x_lower_right
				])
		except Exception as e:
			print('Failed to work on window:', e)

	df_pred_coordinates = pd.DataFrame(
		data    = preds,
		columns = columns
	)

	return df_pred_coordinates

# this function gets two rows of the DataFrame returned from the previous func
# and checks the intersections between the rows (overlaping windows)
# this func returns None if there is no intersection between the windows
# or the resulting coordinates of the intersection if there is one.
def check_intersection(row_a, row_b):

	# this is to make sure objects that are close togheter dont mess with the intersection
	subtract_border = 10

	if row_b['y_upper_left'] >= row_a['y_lower_right']:
		return None
	if row_b['x_upper_left'] >= row_a['x_lower_right']:
		return None
	if row_b['y_lower_right'] <= row_a['y_upper_left']:
		return None
	if row_b['x_lower_right'] <= row_a['x_upper_left']:
		return None
	
	row_a['y_upper_left']  = max(row_a['y_upper_left'], row_b['y_upper_left'])    + subtract_border
	row_a['y_lower_right'] = min(row_a['y_lower_right'], row_b['y_lower_right'])  - subtract_border
	row_a['x_upper_left']  = max(row_a['x_upper_left'], row_b['x_upper_left'])    + subtract_border
	row_a['x_lower_right'] = min(row_a['x_lower_right'], row_b['x_lower_right'])  - subtract_border
	
	return row_a

# this function gets the complete DataFrame containing the first predictions
# and checks all the possible intersections between sliding windows
def get_final_coordinates(df_pred_coordinates, increase_final, max_size_y, max_size_x):
	ignore_idxs        = []
	df_new_coordinates = pd.DataFrame(columns=df_pred_coordinates.columns)

	for i, row_a in df_pred_coordinates.iterrows():
		
		if i in ignore_idxs:
			continue
			
		final_row = row_a
		
		for j, row_b in df_pred_coordinates.iterrows():
			
			if j <= i:
			  continue

			if j in ignore_idxs:
				continue
				
			if row_a['label'] != row_b['label']:
				continue
				
			inter_row = check_intersection(final_row, row_b)
			if inter_row is not None:
				final_row = inter_row
				ignore_idxs.append(j)
				
		df_new_coordinates.loc[i] = final_row

	df_new_coordinates.reset_index(inplace=True, drop=True)
	df_new_coordinates['y_target']      = ((df_new_coordinates['y_upper_left']+df_new_coordinates['y_lower_right'])/2).astype(int)
	df_new_coordinates['x_target']      = ((df_new_coordinates['x_upper_left']+df_new_coordinates['x_lower_right'])/2).astype(int)
	df_new_coordinates['y_upper_left']  = df_new_coordinates['y_target'].apply(lambda c: max(0,          c-increase_final))
	df_new_coordinates['y_lower_right'] = df_new_coordinates['y_target'].apply(lambda c: min(max_size_y, c+increase_final))
	df_new_coordinates['x_upper_left']  = df_new_coordinates['x_target'].apply(lambda c: max(0,          c-increase_final))
	df_new_coordinates['x_lower_right'] = df_new_coordinates['x_target'].apply(lambda c: min(max_size_x, c+increase_final))
	
	return df_new_coordinates

# function to draw squares around the predictions and save the resulting image
# this func will only be runned if plot_predictions=1 (sys.argv[5])
def draw_predictions(img, df_preds_coordinates, path_to_save_csvs, img_name, suffix, channel=2, border=15):
    img_coordinates = img.copy()
    for i, row in df_preds_coordinates.iterrows():
        x_min = row['x_upper_left']
        x_max = row['x_lower_right']
        y_min = row['y_upper_left']
        y_max = row['y_lower_right']
        img_coordinates[y_min:y_min+border, x_min:x_max,        channel] = 255
        img_coordinates[y_max-border:y_max, x_min:x_max,        channel] = 255
        img_coordinates[y_min:y_max,        x_min:x_min+border, channel] = 255
        img_coordinates[y_min:y_max,        x_max-border:x_max, channel] = 255
    path_to_new_img = path_to_save_csvs.joinpath('{}_{}.png'.format(str(img_name)[:-4], suffix))
    imsave(path_to_new_img, img_coordinates)
    print('Image with predictions saved at:', path_to_new_img)

if __name__ == "__main__":

	# getting args
	imgs_path, path_to_save_csvs, path_to_model, plot_predictions = get_args()

	print('\nImages will be loaded from:', imgs_path)
	print('CSVs will be stored at:', path_to_save_csvs)
	print('The model will be loaded from this checkpoint file:', path_to_model)
	
	# loading model, transformation to input images and labels
	model, val_tt, labels_map = load_model_and_params(path_to_model)

	# columns to CSV
	columns = [
		'label',
		'y_target', # dropped before saving the CSV
		'x_target', # dropped before saving the CSV
		'y_upper_left',
		'x_upper_left',
		'y_lower_right',
		'x_lower_right'
	]

	# window size and overlap percentage
	overlap = 0.5
	size    = 256

	# after finding the center of intersections of same prediction, 
	# increase this amount for each direction in order to end up with
	# a 256x256 prediction square 
	increase_final = 128

	# for each PNG image in imgs_path dir
	img_count = 0
	for img_path in imgs_path.iterdir():
		
		# ignore not PNG files
		if img_path.suffix != '.png':
			continue

		img_count += 1

		# loading image
		print('\nLoading image at:', img_path)
		img = np.array(Image.open(img_path))
		max_size_y, max_size_x = img.shape[0], img.shape[1]

		# predicting its windows
		print('Predicting its sliding windows (may take several minutes)...')
		start = time()
		df_pred_coordinates = slide_and_predict(img, model, val_tt, labels_map, size, overlap, columns)
		end = time()
		print('Finished in {:.2f} minutes'.format((end-start)/60))

		# filtering the predictions to get only the intersections
		df_pred_coordinates = get_final_coordinates(df_pred_coordinates, increase_final, max_size_y, max_size_x)

		# deleting columns y_target and x_target + saving CSV file
		path_to_csv = path_to_save_csvs.joinpath(img_path.name[:-4] + '.csv')
		df_pred_coordinates.drop(['y_target', 'x_target'], axis=1, inplace=True)
		df_pred_coordinates.to_csv(path_to_csv, index=False)
		print('CSV stored at:', path_to_csv)

		# saving image with predictions squares
		if plot_predictions:
			draw_predictions(img, df_pred_coordinates, path_to_save_csvs, img_path.name, 'predictions', channel=2, border=15)

	# if no PNG was found
	if img_count == 0:
		print('\nATTENTION! No PNG images found in:', imgs_path)

	print('\nFinished.')
