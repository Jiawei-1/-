path1='C:\Users\gjw\Desktop\3-26template.jpg'
path2='C:\Users\gjw\Desktop\3-28test.jpg'
path1='C:\Users\gjw\OneDrive\3-25test.jpg'
im1=imread(path1);
im2=imread(path2);
im1=rgb2gray(im1);
im2=rgb2gray(im2);
im1=double(im1);
im2=double(im2);
im=im1(1:898,1:854);
imshow(im,[])
figure()
imshow(im1)
corr=calcPearsonCorr2(im1,im2)
% p1=[754,644]
% p2=[437,505]
% r=167
% CorrAngle(path1,path2,p1,p2,r)

% im=imread('circle.png');
% im=rgb2gray(im);
% imshow(im)
% p1=[254,118];
% im2=im(p1(1,2)-2:p1(1,2)+2,p1(1,1)-3:p1(1,1)+3);
% imshow(im2);