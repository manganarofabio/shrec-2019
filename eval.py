import numpy as np
import argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from models import Rnn
import glob
import os
import csv
import torch
import tqdm

parser = argparse.ArgumentParser(description='evaluation script')

# parameters for classification model
parser.add_argument('--class_model_pth', type=str,
                    default="weights/classification.checkpoint.pth.tar")
parser.add_argument('--class_model', type=str, default='LSTM')
parser.add_argument('--input_size_class_model', type=int, default=208)
parser.add_argument('--hidden_size_class_model', type=int, default=256)
parser.add_argument('--n_layers_class_model', type=int, default=2)
parser.add_argument('--final_layer_class_model', type=str, default='fc')
parser.add_argument('--n_classes_class_model', type=int, default=5)
parser.add_argument('--in_file_mn_std_class', type=str, default='weights/mean_std_class.npz')
# parameters for segmentation model
parser.add_argument('--seg_model_pth', type=str,
                    default="weights/segmentation.checkpoint.pth.tar")
parser.add_argument('--seg_model', type=str, default='LSTM')
parser.add_argument('--input_size_seg_model', type=int, default=208)
parser.add_argument('--hidden_size_seg_model', type=int, default=128)
parser.add_argument('--n_layers_seg_model', type=int, default=2)
parser.add_argument('--final_layer_seg_model', type=str, default='fc')
parser.add_argument('--n_classes_seg_model', type=int, default=2)
parser.add_argument('--in_file_mn_std_seg', type=str, default='weights/mean_std_seg.npz')

# parameters for evaluation method
parser.add_argument('--th_is_gesture', type=int, default=5,
                    help="threshold of frames to consider the current sequence as valid gesture")
parser.add_argument('--th_end_gesture', type=int, default=5,
                    help="threshold of frames to consider the current sequence ended")
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset_path', type=str, default='./dataset',
                    help='path to Test directory')
parser.add_argument('--out_file', type=str, default='results.txt',
                    help='file to write results')
parser.add_argument('--no_preprocessing_class', action='store_true', default=False)
parser.add_argument('--no_preprocessing_seg', action='store_true', default=False)
parser.add_argument('--no_increased_input_size', action='store_true', default=False)


args = parser.parse_args()


class ShrecTestDataset(Dataset):

    def __init__(self, dataset_path, preprocessing_class=False, preprocessing_seg=False, increased_input_size=False,
                 coeff=1000,
                 in_file_mn_std_class='mean_std_class.npz',
                 in_file_mn_std_seg='mean_std_seg.npz'):

        self.dataset_path = dataset_path
        self.preprocessing_class = preprocessing_class
        self.preprocessing_seg = preprocessing_seg
        self.increased_input_size = increased_input_size
        self.coeff = coeff
        self.in_file_mn_std_class = in_file_mn_std_class
        self.in_file_mn_std_seg = in_file_mn_std_seg

        list_file = []
        list_file_info = []

        for f_in in sorted(glob.glob(os.path.join(dataset_path, '*')), key=lambda x: int(x[x.find('_')+1:-4])):
            with open(f_in, 'r') as csv_in:
                reader = csv.reader(csv_in)
                file = []
                file_info = []
                for row in reader:
                    frame = [float(x) for x in row[0].split(';')][2:]
                    frame_info = [float(x) for x in row[0].split(';')][:2]
                    file.append(frame)
                    file_info.append(frame_info)

                list_file.append(file)
                list_file_info.append(file_info)

        # list of file in Test directory (frame with input features)
        self.list_data = list_file
        # list of file in Test directory (frame with just id and timestamp)
        self.list_data_info = list_file_info

    def __getitem__(self, item):
        x = self.list_data[item]
        x_info = self.list_data_info[item]

        return x, x_info

    def __len__(self):
        return self.list_data.__len__()


class Eval:

    def __init__(self, test_loader, model_seg, model_class, device, th_is_gesture=5, th_end_gesture=5,
                 out_file="results.txt",  in_file_mn_std_class='mean_std_class.npz',
                 in_file_mn_std_seg='mean_std_seg.npz', preprocessing_class=False, preprocessing_seg=False,
                 increased_input_size=True, acceleration=True, coeff=1000):

        self.test_loader = test_loader
        self.model_seg = model_seg
        self.model_class = model_class
        self.th_is_gesture = th_is_gesture
        self.th_end_gesture = th_end_gesture
        self.device = device
        self.out_file = out_file
        self.in_file_mn_std_class = in_file_mn_std_class
        self.in_file_mn_std_seg = in_file_mn_std_seg
        self.preprocessing_class = preprocessing_class
        self.preprocessing_seg = preprocessing_seg
        self.increased_input_size = increased_input_size
        self.coeff = coeff
        self.acceleration = acceleration

        if self.acceleration and not self.increased_input_size:
            self.acceleration = False

        # loading norm parameters
        if not os.path.exists(self.in_file_mn_std_class) or not os.path.exists(self.in_file_mn_std_seg):
            print('files not existing')
            exit(-1)
        else:
            if self.preprocessing_class:
                # classification mean and std
                npzfile = np.load(self.in_file_mn_std_class)
                self.mean_class = npzfile['arr_0']
                self.std_class = npzfile['arr_1']
                # print('mean, std class loaded.')

            if self.preprocessing_seg:
                # segmentation mean and std
                npzfile = np.load(self.in_file_mn_std_seg)
                self.mean_seg = npzfile['arr_0']
                self.std_seg = npzfile['arr_1']
                # print('mean, std seg loaded.')

    def eval(self):

        self.model_seg.eval()
        self.model_class.eval()

        list_results, list_results_timestamp = [], []
        counter = 0

        for i, file in enumerate(tqdm.tqdm(self.test_loader)):
            # print("Computing file {}...".format(i), end="")

            list_end_frames, buffer, buffer_info = [], [], []
            prev_frame, prev_vel = None, None
            list_vel = []

            output_class = None
            classify = False
            ht, ct = None, None

            for j, frame in enumerate(file[0]):

                if self.increased_input_size:
                    if j == 0:
                        prev_frame = None
                        prev_vel = None
                    else:
                        prev_frame = file[0][j-1]

                    frame = increase_frame_input_size(frame, prev_frame)
                    list_vel.append(frame)

                    if self.acceleration:
                        frame = increase_frame_input_size(frame, list_vel[-1] if j > 0 else None, acceleration=True)
                        if len(list_vel) > 2:
                            list_vel.pop(0)

                if self.preprocessing_seg:
                    if not self.increased_input_size:
                        np_frame = ((frame-self.mean_seg[:112 if not self.acceleration else 208])/self.std_seg[:112
                                    if not self.acceleration else 208]).copy()
                    else:
                        np_frame = ((frame - self.mean_seg) / self.std_seg).copy()
                else:
                    np_frame = frame.copy()
                np_frame = np.asarray(np_frame)
                np_frame = np_frame[:, np.newaxis, np.newaxis]
                np_frame = np.transpose(np_frame, (1, 2, 0))
                np_frame = torch.Tensor(np_frame).to(self.device)

                if j == 0 or (ht is None and ct is None):
                    output, ht, ct = self.model_seg(np_frame, None)
                    output = output[:, -1]
                else:
                    output, ht, ct = self.model_seg(np_frame, (ht, ct))
                    output = output[:, -1]

                output = output.squeeze().argmax().item()

                if output == 1:
                    buffer.append(frame)
                    buffer_info.append(file[1][j])

                    if len(buffer) >= self.th_is_gesture:
                        classify = True

                elif output == 0:
                    if not classify:
                        buffer.clear()
                        buffer_info.clear()

                    elif classify:
                        list_end_frames.append(np_frame)

                    if len(list_end_frames) > self.th_end_gesture and classify:
                        classify = False
                        break

                if classify:

                    if self.preprocessing_class:
                        if not self.increased_input_size:
                            buffer_t = ((buffer-self.mean_class[:112])/self.std_class[:112]).copy()
                        else:
                            buffer_t = ((buffer-self.mean_class[:160 if not self.acceleration else 208]) /
                                        self.std_class[:208 if not self.acceleration else 208]).copy()
                    else:
                        buffer_t = buffer.copy()
                    buffer_t = np.expand_dims(buffer_t, axis=0)
                    buffer_t = torch.Tensor(buffer_t).to(self.device)

                    output_class = self.model_class(buffer_t, None)

                    output_class = output_class[0][:, -1].squeeze().argmax().item()


            # results for current file (trajectory_number, predicted_gesture_label, predicted_start_gesture,
            # predicted_gesture_end)
            if len(buffer_info) > 0:

                if len(buffer_info) > 0:
                    list_results.append([i, from_id_to_label(output_class), int(buffer_info[0][0].item()),
                                        int(buffer_info[-1][0].item())])
                    list_results_timestamp.append([i, from_id_to_label(output_class), buffer_info[0][1].item(),
                                                   buffer_info[-1][1].item()])
                else:
                    list_results.append([i, from_id_to_label(output_class), int(file[1][0][0].item()),
                                         int(file[1][-1][0].item())])
                    list_results_timestamp.append([i, from_id_to_label(output_class), file[1][0][1].item(),
                                                   file[1][-1][1].item()])
            else:
                # print('gesture not found')
                list_results.append([i, '-1'])
                list_results_timestamp.append([i, '-1'])

        print_results(list_results, self.out_file)
        print_results(list_results_timestamp, "{}_timestamp.txt".format(self.out_file[:-4]))
        print("Done.")


def from_id_to_label(output_class):

    if output_class == 0:
        gesture_label = 'X'
    elif output_class == 1:
        gesture_label = 'O'
    elif output_class == 2:
        gesture_label = 'V'
    elif output_class == 3:
        gesture_label = '^'
    elif output_class == 4:
        gesture_label = '[]'

    return gesture_label


# method to increas feature vector with velocity and acceleration
def increase_frame_input_size(frame, prev_frame, acceleration=False, coeff=1000):

    mask = [True, True, True, False, False, False, False] * 16
    if acceleration:
        if prev_frame is None:
            inc_features = list(np.ones(16*3) / coeff)
        else:
            inc_features = np.asarray(frame[112:]) - np.asarray(prev_frame[112:])
        frame = np.hstack((frame, inc_features))

    else:
        if prev_frame is None:
            inc_features = list(np.ones(16*3) / coeff)
        else:
            inc_features = np.asarray(frame)[mask] - np.asarray(prev_frame)[mask]
        frame = np.hstack((frame, inc_features))

    return frame


# method to print classification results
def print_results(list_results, output_file):

    with open(output_file, 'w', newline='') as out:
        writer = csv.writer(out, delimiter=';')
        for row in list_results:
            writer.writerow(row)


def main():

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    # print(device)

    # loading test set
    test_dataset = ShrecTestDataset(dataset_path=args.dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # create models
    model_class = Rnn(rnn_type=args.class_model, input_size=args.input_size_class_model,
                      hidden_size=args.hidden_size_class_model, batch_size=args.batch_size,
                      num_classes=args.n_classes_class_model, num_layers=args.n_layers_class_model, classification=True,
                      final_layer=args.final_layer_class_model).to(device)

    model_seg = Rnn(rnn_type=args.seg_model, input_size=args.input_size_seg_model,
                    hidden_size=args.hidden_size_seg_model, batch_size=args.batch_size,
                    num_classes=args.n_classes_seg_model, num_layers=args.n_layers_seg_model, classification=False,
                    final_layer=args.final_layer_seg_model).to(device)

    # load pretrained models
    model_class.load_state_dict(torch.load(args.class_model_pth, map_location=device)['state_dict'])
    model_seg.load_state_dict(torch.load(args.seg_model_pth, map_location=device)['state_dict'])

    # create Evaluator class
    evaluator = Eval(test_loader=test_loader, model_seg=model_seg, model_class=model_class, device=device,
                     th_is_gesture=args.th_is_gesture, th_end_gesture=args.th_end_gesture, out_file=args.out_file,
                     in_file_mn_std_class=args.in_file_mn_std_class, in_file_mn_std_seg=args.in_file_mn_std_seg,
                     preprocessing_class=not args.no_preprocessing_class, preprocessing_seg=not args.no_preprocessing_seg,
                     increased_input_size=not args.no_increased_input_size
                     )

    # run evaluation
    evaluator.eval()


if __name__ == '__main__':
    main()