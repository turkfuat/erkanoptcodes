# erkanoptcodes
Segmentation Code :
 clc
 clear
% % im_bin_final: mask image (logical)
% % im: original image (uint8)
% % im2: original image grayscale (uint8)
% % im_tumor: marked pixels (double vector)
% % im_unedited: original unmarked image;
% % im_merge: mask applied original image
% 
% %Selecting of images
% all_images = dir('*.1.jpg');
% number_of_images = length(all_images);
% 
% all_images2 = dir('*.2.jpg');
% number_of_images2 = length(all_images2);
% 
% %reading the images using imread function
% for i = 1:number_of_images
% %i = 3;   % image number
% i;
% name = all_images(i).name;
% im = imread(name);
% 
% %tresholding 
% im_bin = im(:,:,1) < 102 & im(:,:,2) > 101 & im(:,:,3) < 145;
% 
% 
% im_bin2 = double(im_bin); %binary convert
% se = strel('disk',5,8);
% 
% %morphologic operators
% im_bin_modi = imdilate(im_bin2,se);
% 
% 
% im_bin3 = imbinarize(im_bin_modi);
% im_bin4 = imfill(im_bin3,'holes');
% 
% L = bwlabel (im_bin4,8);  
% m = max(max(L));
% c = zeros(m,1);
% 
% 
% for j = 1:m
%     c(j) = nnz(L==j);
% end
% 
% [a,b] = max(c);
% 
% im_bin_final = L == b;
% 
% figure (1)
% imshow(im_bin_final)
% 
% % Shape Features
% 
% a = regionprops(im_bin_final,'Area','Perimeter','Eccentricity','Solidity');  % shape parameters
% shapes = [a.Area a.Perimeter  a.Eccentricity a.Solidity];
% maxradius= max(sqrt(a.Area/pi);
% minradius= min(sqrt(a.Area/pi);
% equivalentdiameter=(a.Area/pi);
% elongatedness= (a.Area/2*pi*maxradius).^2);
% circulation1=(sqrt(a.Area/(pi*(maxradius.^2)));
% circulation2=(sqrt(maxradius/minradius);
% compactness=(2*sqrt((a.Area*pi)/(a.Perimeter)));
% dispersion= (maxradius/minradius);
% thinnesratio=(4*pi*a.Area/a.Perimeter);
% shapeindex=(a.Perimeter/(2*maxradius));
% entropy=entropy(im_bin_final);
% 
% for j = 1:number_of_images2
% %i = 3;   % image number
% i;
% name2 = all_images2(j).name;
% im_original = imread(name2);
% 
% im_original = im_original(:,:,1:3);
% im_merge = double(rgb2gray(im_original)).*double(im_bin_final);
% 
% 
% % delete all zero rows & columns
% im_merge( ~any(im_merge,2), : ) = [];
% im_merge( :, ~any(im_merge,1) ) = [];
% figure(2)
% imshow(im_merge,[]);
% % Histogram Based Features
%  im2 = rgb2gray(im);
%  im_tumor = double(im2(im_bin_final == 1));
%  avg_int = mean(im_tumor);
%  standard = std(im_tumor);
%  smooth = 1 - 1/(1 + standard^2);
%  h = histogram(im_tumor,'BinWidth',1);
%  mad=mad(im_tumor);
%  skew=skewness(im_tumor);
%  kurt=kurtosis(im_tumor);
%  min=min(im_tumor);
%  max=max(im_tumor);
%  10th=prctile(im_tumor,10);
%  90th=prctile(im_tumor,90);
%  interq=igr(im_tumor);
%  rms=rmse(im_tumor);
%  med=median(im_tumor);
 
 %% Gray Level Co-Occurrence Matrix
% g1 = graycomatrix(im_merge, 'NumLevels',max(max(im_merge))+1, 'GrayLimits', [], 'Offset', [0 1]);
% g2 = graycomatrix(im_merge, 'NumLevels',max(max(im_merge))+1, 'GrayLimits', [], 'Offset', [-1 1]);
% g3 = graycomatrix(im_merge, 'NumLevels',max(max(im_merge))+1, 'GrayLimits', [], 'Offset', [-1 0]);
% g4 = graycomatrix(im_merge, 'NumLevels',max(max(im_merge))+1, 'GrayLimits', [], 'Offset', [-1 -1]);
% g1(1,1) = 0; % final gray level co occurence matrix
% g2(1,1) = 0;
% g3(1,1) = 0;
% g4(1,1) = 0;
% 
% gl1 = graycoprops(g1);
% gl2 = graycoprops(g2);
% gl3 = graycoprops(g3);
% gl4 = graycoprops(g4);
% 
% glcm_contrast1 = gl1.Contrast;
% glcm_correlation1 = gl1.Correlation;
% glcm_energy1 = gl1.Energy;
% glcm_homo1 = gl1.Homogeneity;
% 
% glcm_contrast2 = gl2.Contrast;
% glcm_correlation2 = gl2.Correlation;
% glcm_energy2 = gl2.Energy;
% glcm_homo2 = gl2.Homogeneity;
% 
% glcm_contrast3 = gl3.Contrast;
% glcm_correlation3 = gl3.Correlation;
% glcm_energy3 = gl3.Energy;
% glcm_homo3 = gl3.Homogeneity;
% 
% glcm_contrast4 = gl4.Contrast;
% glcm_correlation4 = gl4.Correlation;
% glcm_energy4 = gl4.Energy;
% glcm_homo4 = gl4.Homogeneity;
% 
% h1(i,:)=[gl1.Contrast,gl1.Correlation,gl1.Energy,gl1.Homogeneity];
% h2(i,:)=[gl2.Contrast,gl2.Correlation,gl2.Energy,gl2.Homogeneity];
% h3(i,:)=[gl3.Contrast,gl3.Correlation,gl3.Energy,gl3.Homogeneity];
% h4(i,:)=[gl4.Contrast,gl4.Correlation,gl4.Energy,gl4.Homogeneity];
% % 
% [GLRLM,SI]=grayrlmatrix(im_merge,'OFFSET',[1;2;3;4],'NumLevels',max(max(im_merge),'G',[]);
% % stats2=grayrlprops(GLRLM,{'SRE','LRE','GLN','RLN','RP','LGRE','HGRE','SRLGE','SRHGE','LRLGE','LRHGE'});

Relief Code :

Relief code
[idx,weights] = relieff(X,Y,10);

Binary Harris Hawk Optimization Code :



% feat     : feature vector (instances x features)
% label    : label vector (instance x 1)
% N        : Number of hawks
% max_Iter : Maximum number of iterations


% sFeat    : Selected features
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------

 Binary Harris Hawk Optimization
clc, clear, close; 
% Benchmark data set 
load ionosphere.mat; 

Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);

Parameter setting
N        = 10; 
max_Iter = 100;
% Binary Harris Hawk Optimization
[sFeat,Sf,Nf,curve] = jBHHO(feat,label,N,max_Iter,HO);

Plot convergence curve
plot(1:max_Iter,curve);
xlabel('Number of iterations');
ylabel('Fitness Value');
title('BHHO'); grid on;
