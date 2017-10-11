femfel_struct = load('femfel.mat');
img1 = femfel_struct.femfel1;
img2 = femfel_struct.femfel2;

padded_img = padarray(img2, [20,20], 0);

% size(img1)
% size(padded_img)

color_depth = size('rgb', 2);
total_c
for i = 1:color_depth
    c = normxcorr2(img1(:,:,i), padded_img(:,:,i));
    total_c += c
%     figure, surf(c), shading flat;
     
end

total_c

% imshowpair(img1, img2, 'montage');
% imshowpair(img1, img2, 'falsecolor');
% imshow(abs(img1 - img2));