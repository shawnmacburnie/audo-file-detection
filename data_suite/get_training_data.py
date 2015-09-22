__author__ = 'shawn'
import theano
import numpy
def get_training_data(file_name, window_size, window_inc):
    '''
    :param file_name: Name of the file you want to load
    :param window_size: size of each window in tics (lines in file)
    :param window_inc: how much you want to jump to get to next window.
    :return: shared variable containing all a matrix of windows to train on.
    '''
    with open(file_name) as f:
        lines = f.readlines()
        training_data = []
        index = 0
        running = True
        running_total = 0
        while running:
            vectors = []
            single_training_example = []
            if index + window_inc >= len(lines):
                running = False
                vectors = lines [index: len(lines)]
            else:
                vectors = lines[index:index + window_inc]
            for vector in vectors:
                # this logic needs to be finished. I will try and work this later tonight.
                if type(vector) == int:
                    continue
                points = vector.split(' ')
                # print(len(points))
                begin_list = list(map(int, points[0:-5]))
                continue_flag = int(points[-1].replace('\n',''))
                note_length = points[-2]
                velocity = points[-3]
                position_in_bar = points[-4]
                begin_list.extend(string_to_bin_array(position_in_bar,16))
                begin_list.extend(string_to_bin_array(velocity, 4))
                begin_list.extend(string_to_bin_array(note_length, 16))
                begin_list.extend([continue_flag])
                single_training_example.extend(begin_list)
            training_data += [single_training_example]
            index += window_inc
        n_input = len(training_data[0])
        return n_input, to_shared(training_data)

def string_to_bin_array(number,size):
    number  = float(number)
    a = [0] * size
    index = round(size * number)
    if index >= size:
        index = size -1
    a[index] = 1
    return a

def to_shared(data):
    return theano.shared(numpy.asarray(data,dtype=theano.config.floatX), borrow=True)