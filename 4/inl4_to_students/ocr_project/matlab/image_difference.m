femfel_struct = load('femfel.mat');
img1 = femfel_struct.femfel1;
img2 = femfel_struct.femfel2;

img1_cropped = img1(4:end, 2:end-3, :);
height = size(img1, 1);
width = size(img1, 2);
img1_resized = imresize(img1_cropped, [height width]);



size(img1_cropped)
size(img1)

diff = abs(img1_resized - img2);
imshow(diff);
% imshowpair(img1_cropped, img1);
% imshowpair(img1_resized, img2, 'montage');
% imshowpair(img1_resized, img2, 'falsecolor');

sigma = 4;
blurred1 = imgaussfilt(img1, sigma);
blurred2 = imgaussfilt(img2, sigma);

diff = blurred1 - blurred2;

% imshow(diff);
% imshowpair(img1, img2, 'montage');
% imshowpair(img1, img2, 'diff','Scaling','joint');
% imshow(abs(cropped_img - img1));
% imshow(abs(img2 - img1));
