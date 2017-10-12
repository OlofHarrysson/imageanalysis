%% Get the directory of a dataset

datadir = '../datasets/home3';
a = dir(datadir);

%% Select a filename
file = 'im1'

%% Generate filename with path and extension
fnamebild = [datadir filesep file '.jpg']
fnamefacit = [datadir filesep file '.txt']

%% Read an image and convert to double
bild = double(imread(fnamebild));

%% Read the ground truth interpretation
fid = fopen(fnamefacit);
facit = fgetl(fid)
fclose(fid);

%% Plot the image with ground truth as title

% figure(1); colormap(gray);
% imagesc(bild(:,1:200));
% title(facit);

%% Run your segmentation code
S = im2segment(bild);
b = S{1};
segment2features(b);

