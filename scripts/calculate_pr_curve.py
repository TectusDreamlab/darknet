import sys
import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--validation-file', required = True,
		help='validation file used to test, this should be a file\n')
	parser.add_argument('--names-file', required = True,
		help='path to class names, this should be a file\n')
	parser.add_argument('--results-folder', required = True,
		help='path to the valid results, this should be a folder\n')

	args = parser.parse_args()

	validation_file = args.validation_file
	names_file = args.names_file
	results_folder = args.results_folder

	class_results = {}
	class_ground_truth = {}

	confidence_thresh = 0.005

	# Get the class names
	with open(names_file) as f:
		names = f.read().strip('\n').split('\n')

	print(names)

	# Get the test results from the results folder
	for name in names:
		file_name = '%s/comp4_det_test_%s.txt'%(results_folder, name)
		with open(file_name) as f:
			lines = f.readlines()
			splitlines = [x.strip().split(' ') for x in lines]
			image_ids = [x[0] for x in splitlines]
			confidence = np.array([float(x[1]) for x in splitlines])
			bounding_boxes = np.array([[float(z) for z in x[2:]] for x in splitlines])
			bounding_boxes = np.array([[float(z) for z in x[2:]] for x in splitlines])

			# sort by confidence
			if confidence.size > 0:
				sorted_ind = np.argsort(-confidence)
				bounding_boxes = bounding_boxes[sorted_ind, :]
				confidence = confidence[sorted_ind]
				image_ids = [image_ids[x] for x in sorted_ind]

				class_results[name] = {
					'image_ids': image_ids,
					'confidence': confidence,
					'bounding_boxes': bounding_boxes,
				}

	# Get all the ground truths for the validation dataset.
	with open(validation_file) as f:
		lines = f.readlines()
		images = [x.strip() for x in lines]
		labels = [image.replace('JPEGImages', 'labels').replace('jpg', 'txt') for image in images]
		image_ids = [os.path.splitext(os.path.basename(image))[0] for image in images]

		for name in names:
			class_ground_truth[name] = {
				'total_objects': 0
			}

		for i in xrange(len(image_ids)):
			frame = cv2.imread(images[i])
			height, width, c = frame.shape

			with open(labels[i]) as f:
				lines = f.readlines()
				splitlines = [x.strip().split(' ') for x in lines]
				classes = [names[int(x[0])] for x in splitlines]
				bounding_boxes = [[(float(x[1]) - float(x[3]) / 2.0) * width, (float(x[2]) - float(x[4]) / 2.0) * height, (float(x[1]) + float(x[3]) / 2.0) * width, (float(x[2]) + float(x[4]) / 2.0) * height] for x in splitlines]

				# for each class, find out the bounding boxes.
				for j in xrange(len(classes)):
					if image_ids[i] in class_ground_truth[classes[j]]:
						class_ground_truth[classes[j]][image_ids[i]]['bounding_boxes'] = np.append(class_ground_truth[classes[j]][image_ids[i]]['bounding_boxes'], np.array([bounding_boxes[j]]), 0)
						class_ground_truth[classes[j]][image_ids[i]]['detected'] = np.append(class_ground_truth[classes[j]][image_ids[i]]['detected'], np.array([False]), 0)
					else:
						class_ground_truth[classes[j]][image_ids[i]] = {
							'bounding_boxes': np.array([bounding_boxes[j]]),
							'detected': np.array([False]),
						}
					class_ground_truth[classes[j]]['total_objects'] += 1

	# Filter the results, get rid of the values that are below the confidence.
	tps = np.array([])
	fps = np.array([])
	total_objects = 0
	false_positive_detections_iou = 0
	false_positive_detections_duplicates = 0

	for name in names:
		class_result = class_results[name]

		# print(name, class_result['image_ids'])

		ground_truth = class_ground_truth[name]

		total_objects += ground_truth['total_objects']

		num_detections = len(class_result['image_ids'])
		true_positive_detections = np.zeros(num_detections)
		false_positive_detections = np.zeros(num_detections)

		# Reset the state for the ground truth..
		for image_id in ground_truth:
			if image_id != 'total_objects':
				ground_truth[image_id]['detected'] = np.array([False]*ground_truth[image_id]['detected'].size)

		for d in range(num_detections):
			image_id = class_result['image_ids'][d]

			if image_id not in ground_truth:
				false_positive_detections[d] = 1
			else:
				ground_truth_bounding_boxes = ground_truth[image_id]['bounding_boxes']
				bb = class_result['bounding_boxes'][d, :].astype(float)
				# print('result:', bb)

  				iou_max = -np.inf
  				# print('ground truth', ground_truth_bounding_boxes)

  				if ground_truth_bounding_boxes.size > 0:
	  				# Compute IOU
	  				ixmin = np.maximum(ground_truth_bounding_boxes[:, 0], bb[0])
	  				iymin = np.maximum(ground_truth_bounding_boxes[:, 1], bb[1])
	  				ixmax = np.minimum(ground_truth_bounding_boxes[:, 2], bb[2])
	  				iymax = np.minimum(ground_truth_bounding_boxes[:, 3], bb[3])

	  				intersections = np.maximum(ixmax - ixmin + 1.0, 0.0) * np.maximum(iymax - iymin + 1.0, 0.0)

	  				unions = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) +
	  						(ground_truth_bounding_boxes[:, 2] - ground_truth_bounding_boxes[:, 0] + 1.0) *
	  						(ground_truth_bounding_boxes[:, 3] - ground_truth_bounding_boxes[:, 1] + 1.0) - intersections)

	  				ious = intersections / unions
	  				iou_max = np.max(ious)
	  				iou_max_index = np.argmax(ious)

	  			if class_result['confidence'][d] > confidence_thresh:
		  			if iou_max > 0.5:
		  				if not ground_truth[image_id]['detected'][iou_max_index]:
							true_positive_detections[d] = 1.
							ground_truth[image_id]['detected'][iou_max_index] = 1
						else:
							false_positive_detections_duplicates += 1
							false_positive_detections[d] = 1
		  			else:
		  				false_positive_detections_iou += 1
		  				false_positive_detections[d] = 1


	  	tps = np.append(tps, true_positive_detections)
	  	fps = np.append(fps, false_positive_detections)

	  	# # compute precision recall
		# tp = np.sum(true_positive_detections)
		# fp = np.sum(false_positive_detections)

		# print('number of true detections:', tp)
		# print('number of false detections:', fp)
		# print('false detection because of IOU', false_positive_detections_iou)
		# print('false detection because of duplicates', false_positive_detections_duplicates)

		recall = np.cumsum(true_positive_detections) / ground_truth['total_objects']
		precision = np.cumsum(true_positive_detections) / (np.cumsum(true_positive_detections) + np.cumsum(false_positive_detections))

		plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,
		                 color='b')

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Precision-Recall curve for %s'%name)

		plt.show()

	# print('For confidence level %0.3f, the precision is: %0.2f, recall is: %0.2f'%(confidence, precision, recall))




if __name__=="__main__":
	main(sys.argv)
