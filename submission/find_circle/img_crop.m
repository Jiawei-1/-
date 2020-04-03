function img_crop(path,save_path)
img=imread(path);
img=imcrop(img,[0,0,300,300]);
imwrite(img,save_path);
end