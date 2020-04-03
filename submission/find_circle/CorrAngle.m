function angle=CorrAngle(path1,path2,p1,p2,r)
im1=imread(path1);
im2=imread(path2);
im1=rgb2gray(im1);
im2=rgb2gray(im2);
angle=0;
Max=0;
for i = 1:360
    im=imrotate(im2,i,p2);
    im1_cut=im1(p1(1,2)-r:p1(1,2)+r,p1(1,1)-r:p1(1,1)+r);
    im2_cut=im1(p2(1,2)-r:p2(1,2)+r,p2(1,1)-r:p2(1,1)+r);
    corr=calcPearsonCorr2(im1_cut,im2_cut);
    if corr>Max
        Max=corr;
        angle=i;
    end
end
imrotate(im2,3,'bi'(2,2))
% 
% im=imread('circle.png');
% im=rgb2gray(im);
% imshow(im)
% p1=[254,118];
% im2=im(p1(1,2)-2:p1(1,2)+2,p1(1,1)-3:p1(1,1)+3);
% imshow(im2);