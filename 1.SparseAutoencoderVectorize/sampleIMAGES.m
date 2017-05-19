function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 

patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

image_size = size(IMAGES);
patchNum = 0;
hMax = 20;
wMax = numpatches/image_size(3)/hMax;

for n = 1:image_size(3)
    for h = 1:hMax
        for w = 1:wMax
            patch = IMAGES(patchsize*w:(patchsize*(w+1)-1),patchsize*h:(patchsize*(h+1)-1),n);
            patch = reshape(patch,1,patchsize*patchsize);
            patchNum = patchNum + 1;
            patches(:,patchNum) = patch;
        end
    end
end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
% 把所有数据减去均值，那么所有数据的99.7%都会落在[-3*标准差，3*标准差]之间
% 所以我们只需要把剩下的0.03%的数据都置成-3*标准差或3*标准差即可。
% 这样所有数据都在[-3*标准差，3*标准差]之间，除以%3*标准差，那么数据都会在[-1,1]之间。
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
