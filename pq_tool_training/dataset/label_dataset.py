import os
from PIL import Image
from os import getcwd

classes = ["horizontal profometer", "vertical profometer", "pundit"]
ids = [
	['1-11', '19-24', '29-81', '137-214'],
	['12-18', '25-28', '82-136', '215-286'],
	['287-362']
]

wd = getcwd()

def convert(x1, y1, x2, y2, width, height):
	dw = 1./(width)
	dh = 1./(height)

	x = ((x1 + x2) / 2.0) * dw
	y = ((y1 + y2) / 2.0) * dh

	w = (x2 - x1) * dw
	h = (y2 - y1) * dh

	return x, y, w, h

def get_labels(start_id, end_id, cls, training_file, validate_file, class_training_file, class_validate_file):
	start = int(start_id)
	end = int(end_id)
	for i in range(start, end+1):
		# read the boundingboxes and write to yolo's required format.
		in_file = open('BoundingBoxLabels/%05d.txt'%(i))
		out_file = open('labels/%05d.txt'%(i), 'w')

		boundingboxes = in_file.read().split('\n')[1]
		x1, y1, x2, y2 = boundingboxes.split(' ')

		# get the image size
		im = Image.open('JPEGImages/%05d.jpg'%(i))
		width, height = im.size

		bb = convert(int(x1), int(y1), int(x2), int(y2), width, height)

		cls_id = classes.index(cls)

		try:
			out_file.write(str(cls_id) + " " + " ".join(['{0:0.4f}'.format(a) for a in bb]))
		except IOError as (errno,strerror):
			print "I/O error({0}): {1}".format(errno, strerror)

		if (i - start) % 5 == 4:
			validate_file.write('%s/JPEGImages/%05d.jpg\n'%(wd, i))
			class_validate_file.write('%s/JPEGImages/%05d.jpg\n'%(wd, i))
		else:
			training_file.write('%s/JPEGImages/%05d.jpg\n'%(wd, i))
			class_training_file.write('%s/JPEGImages/%05d.jpg\n'%(wd, i))

def main():
	if not os.path.exists('labels'):
		os.makedirs('labels')

	if not os.path.exists('distribution'):
		os.makedirs('distribution')

	training_file = open('training.txt', 'w')
	validate_file = open('val.txt', 'w')

	for i in xrange(len(classes)):
		id_distribution = ids[i]
		class_name = classes[i]

		class_training_file = open('distribution/%s_training.txt'%(class_name), 'w')
		class_validate_file = open('distribution/%s_val.txt'%(class_name), 'w')

		for class_ids in id_distribution:
			start_id, end_id = class_ids.split('-')
			get_labels(start_id, end_id, class_name, training_file, validate_file, class_training_file, class_validate_file)

		class_training_file.close()
		class_validate_file.close()

	training_file.close()
	validate_file.close()

if __name__ == "__main__":
	main()






