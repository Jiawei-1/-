function GetCircle(path,filename)


filename2 = path;
% parameters
Tac = 10;
Tni = 0.1;

% read image
% disp('read image------------------------------------------------');
I = imread(filename2);
% figure;imshow(I);
% circle detection
% disp('circle detetion-------------------------------------------');
[circles, ~,~] = circleDetectionByArcsupportLS(I, Tac, Tni);
[m,n]=size(circles);
for i = 1:m
    CircleWriteXML(filename,circles(i,3),circles(i,1),circles(i,2))
end
% display
% disp('show------------------------------------------------------');
% circles
% disp(['number of circles£º',num2str(size(circles,1))]);
% disp('draw circles----------------------------------------------');
% dispImg = drawCircle(I,circles(:,1:2),circles(:,3));
% figure;
% imshow(dispImg);
% 
% path='circle.png';
% save_path='circle2.png';
% img_crop(path,save_path)
end