import sys
import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def avg_precision(names, class_results, class_ground_truth, iou, use_voc_07_metric):
	aps = []
	tps = []
	fps = []
	total_objects = 0
	avg_iou = []

	confidence_thresh = 0.005

	# Now for each class, we calculate the average presicion.
	for name in names:
		if name not in class_results:
			aps += [0.0]
			continue

		class_result = class_results[name]
		ground_truth = class_ground_truth[name]

		num_detections = len(class_result['image_ids'])
		true_positive_detections = np.zeros(num_detections)
		false_positive_detections = np.zeros(num_detections)

		# Reset the state for the ground truth..
		for image_id in ground_truth:
			if image_id != 'total_objects':
				ground_truth[image_id]['detected'] = np.array([False]*ground_truth[image_id]['detected'].size)

		for d in range(num_detections):

			image_id = class_result['image_ids'][d]

			# print('image id: %s'%(image_id))

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
		  			if iou_max > iou:
		  				if not ground_truth[image_id]['detected'][iou_max_index]:
							true_positive_detections[d] = 1
							avg_iou.append(iou_max)
							ground_truth[image_id]['detected'][iou_max_index] = 1
						else:
							false_positive_detections[d] = 1
		  			else:
		  				false_positive_detections[d] = 1

	  	# compute precision recall
		tp = np.cumsum(true_positive_detections)
		fp = np.cumsum(false_positive_detections)

		tps += [tp[-1:]]
		fps += [fp[-1:]]
		total_objects += ground_truth['total_objects']

		precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
		recall = tp / float(ground_truth['total_objects'])

		# print('precision for class:', name, precision)
		# print('recall for class:', name, recall)

		if use_voc_07_metric:
			# 11 point metric
			ap = 0.
			for t in np.arange(0., 1.1, 0.1):
				if np.sum(recall >= t) == 0:
					p = 0
				else:
					p = np.max(precision[recall >= t])
				ap = ap + p / 11.
		else:
			# correct AP calculation
			# first append sentinel values at the end
			mrec = np.concatenate(([0.], recall, [1.]))
			mpre = np.concatenate(([0.], precision, [0.]))

			# compute the precision envelope
			for i in range(mpre.size - 1, 0, -1):
				mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

			# to calculate area under PR curve, look for points
			# where X axis (recall) changes value
			i = np.where(mrec[1:] != mrec[:-1])[0]

			# and sum (\Delta recall) * prec
			ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

		# print('AP%d for class %s is %02.1f%%'%(iou*100, name, ap*100))
		aps += [ap]

	print('AP%d is %02.1f%%'%(iou*100, np.mean(aps)*100))
	print('Total true positives is:%d'%(np.sum(tps)))
	print('Total false positives is:%d'%(np.sum(fps)))
	print('Total objects is:%d'%(total_objects))
	iou = np.mean(avg_iou) if len(avg_iou) > 0 else .0
	print('Average IOU is:%02.f%%'%(iou*100))

	return np.sum(tps), np.sum(fps), total_objects, np.mean(aps)

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

		# print class_ground_truth

	# Let's plot the table for the average precision
	iou_range = np.arange(0.5, 1, 0.05)
	aps = []
	for x in iou_range:
		tp, fp, total, ap = avg_precision(names, class_results, class_ground_truth, x, True)
		aps += [ap]
		print('precision:%02.3f%%'%(tp/float(tp+fp)*100))
		print('recall:%02.3f%%\n'%(tp/float(total)*100))

	print('AP is %02.1f%%'%(np.mean(aps)*100))

	cell_text = [['%02.1f%%'%(aps[0]*100), '%02.1f%%'%(aps[5]*100), '%02.1f%%'%(aps[8]*100), '%02.1f%%'%(np.mean(aps)*100)]]
	# Add a table at the bottom of the axes
	the_table = plt.table(cellText=cell_text,
	                      rowLabels=['YOLOv2'],
	                      colLabels=['AP 0.5', 'AP 0.75', 'AP 0.9', 'AP 0.5:0.05:0.95'],
	                      loc='upper center')

	plt.title('Average Precisions')
	plt.axis('off')
	plt.show()


if __name__=="__main__":
	main(sys.argv)