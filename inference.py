import numpy as np
import cv2, os, argparse, audio
import subprocess
from tqdm import tqdm
import torch, face_detection
from models import Wav2Lip
import platform
from onnxruntime import InferenceSession

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = [b for b in boxes[len(boxes) - T:] if not np.array_equal(b, [0, 0, 1, 1])]
		else:
			window = [b for b in boxes[i : i + T] if not np.array_equal(b, [0, 0, 1, 1])]
		if len(window) > 0:
			boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			# cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			# raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
			results.append([0, 0, 1, 1])
		else:
			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)
			results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def load_scaler(model_path: str, device_id: int = 0) -> InferenceSession:
    providers = [('DmlExecutionProvider', {"device_id": str(device_id)})]
    return InferenceSession(model_path, providers=providers)

def main():
	still_reading = True
	video_stream = None
	full_mel_chunks = None
	out = None
	scaler = load_scaler('work/BSRGANx4_fp16.onnx')
	while still_reading:
		if not os.path.isfile(args.face):
			raise ValueError('--face argument must be a valid path to video/image file')

		elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
			still_reading = False
			full_frames = [cv2.imread(args.face)]
			fps = args.fps

		else:
			if video_stream is None:
				video_stream = cv2.VideoCapture(args.face, cv2.CAP_FFMPEG)
				video_stream.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
			fps = video_stream.get(cv2.CAP_PROP_FPS)

			print('Reading video frames...')

			full_frames = []
			while len(full_frames) < 1800:
				still_reading, frame = video_stream.read()
				if not still_reading:
					video_stream.release()
					break

				if args.resize_factor > 1:
					frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

				if args.rotate:
					frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

				y1, y2, x1, x2 = args.crop
				if x2 == -1: x2 = frame.shape[1]
				if y2 == -1: y2 = frame.shape[0]

				frame = frame[y1:y2, x1:x2]

				full_frames.append(frame)

		print ("Number of frames available for inference: "+str(len(full_frames)))

		if full_mel_chunks is None:
			if not args.audio.endswith('.wav'):
				print('Extracting raw audio...')
				command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

				subprocess.call(command, shell=True)
				args.audio = 'temp/temp.wav'

			wav = audio.load_wav(args.audio, 16000)
			mel = audio.melspectrogram(wav)
			print(mel.shape)

			if np.isnan(mel.reshape(-1)).sum() > 0:
				raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

			full_mel_chunks = []
			mel_idx_multiplier = 80./fps 
			i = 0
			while 1:
				start_idx = int(i * mel_idx_multiplier)
				if start_idx + mel_step_size > len(mel[0]):
					full_mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
					break
				full_mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
				i += 1

			print("Length of mel chunks: {}".format(len(full_mel_chunks)))

		mel_chunks = full_mel_chunks[:len(full_frames)]
		full_mel_chunks = full_mel_chunks[len(full_frames):]
		full_frames = full_frames[:len(mel_chunks)]

		batch_size = args.wav2lip_batch_size
		gen = datagen(full_frames.copy(), mel_chunks)

		for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
												total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
			if out is None:
				model = load_model(args.checkpoint_path)
				print ("Model loaded")

				frame_h, frame_w = full_frames[0].shape[:-1]
				out = cv2.VideoWriter('temp/result.m4v', 
										cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)

			pred = np.array([
				np.transpose(np.clip(np.squeeze(
					scaler.run(None, {scaler.get_inputs()[0].name: np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)})[0]
				, axis=0), 0, 1), (1, 2, 0)) for image in pred.cpu().numpy().transpose(0, 2, 3, 1)
			])

			for p, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				height = y2 - y1
				if height > 1:
					m = height // 2
					width = x2 - x1
					p = cv2.resize(p, (width, height), interpolation=cv2.INTER_LINEAR)
					height = height - m
					mask = np.zeros((height, width), dtype=np.float32)
					mask[48:-48, 48:-48] = 1
					mask = cv2.GaussianBlur(mask, (97, 97), 0)
					mask = mask[..., np.newaxis]
					n = 1 - mask
					p = p[m:, :] * 255
					f[y1 + m:y2, x1:x2] = (p * mask + f[y1 + m:y2, x1:x2] * n).astype(np.uint8)
				out.write(f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 -c copy {}'.format(args.audio, 'temp/result.m4v', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()
