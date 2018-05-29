clear
num = 971;
% load annotation_list.mat
dst_dir = '/home/jihang/Jiahong/pix2pix-tensorflow/OCT_data/train_balance_round1/';
ori_dir = '/data/OCT/Annotations/';
ann_dir = ori_dir;
% dst_dir = '/home/jihang/Jiahong/pix2pix-tensorflow/OCT_data/train_clean_mask/';
% ori_dir = '/home/jihang/Jiahong/data/OCT/Annotations/';
% ann_dir = '/home/jihang/Jiahong/data/OCT/Annotations/';

margin = 0;
extend = 50;
low = 30;
high = 80;
block = 256

% for t = [1, 3] %test_traindata
% for t = [3,4,6,8,13,14,21]  %clean
% for t = [1,22,25,26,28,32,33,34]    %noisy
% for t = [15, "corneaSubject1", "corneaSubject5", "corneaSubject6", "corneaSubject7", ...
%         "corneaSubject8", "palisades25_102AM_040417", "palisades28_172_180213",...
%         "palisades31_187_180213"]    % new data
% for t = ["3", "13"] % clean and dark
% for t = ["1", "32", "33", "34"]  % noisy
% for t = ["25", "26", "28", "corneaSubject1"]    % cornea
for t = ["palisades25_102AM_040417", "palisades28_172_180213", "palisades31_187_180213"]
    t = char(t)
    ori_path = [ori_dir num2str(t) '/original/']
    ann_path = [ann_dir num2str(t) '/originalAnnotationTejas_new/'];
    ann_directory = dir(ann_path);
    for i = 1 : length(ann_directory)
%     for i = 1 : 10 : length(ann_directory)
        if(isequal(ann_directory(i).name, '.') || isequal(ann_directory(i).name, '..'))   %exclude hidden folder & not folder
            continue;
        end
        num = num + 1;
        img_ori_path = [ori_path ann_directory(i).name]; %all images path
        img_ori = imread(img_ori_path);
        img_ann_path = [ann_path ann_directory(i).name];
        img_ann = imread(img_ann_path);
        [row col cha] = size(img_ann);
%         figure(1)
%         imshow(img_ann);
        img_clear = img_ori;
        img_mask = zeros(row, col);
        num_front = 0;
        for j = 1 : col
            i = 1;
            while(i <= row - 5)
%                 num_front = num_front + 1;
%                 if(img_ann(i+margin,j,1)==255 && img_ann(i+margin,j,2)==0 && img_ann(i+margin,j,3)==0)
                if(img_ann(i+margin,j,1)>=200 && img_ann(i+margin,j,2)<200 && img_ann(i+margin,j,3)<200)
                    break;
                else
                  %img_clear(i,j) = low+rand()*(high-low);
                  img_clear(i,j) = 0;
                  img_mask(i,j) = 255;
                end
                i = i + 1;
            end
            lastrow = min(row-1, i+extend);
            img_mask(i:lastrow,j) = 255;
        end
%         figure(1);
%         imshow(img_mask);
%         figure(2);
%         imshow(img_ori);


        for k = 1 : floor(col/block)
            img_ori_tmp = img_ori(:, (k-1)*block+1:k*block);
            img_clear_tmp = img_clear(:, (k-1)*block+1:k*block);
            img_mask_tmp = img_mask(:, (k-1)*block+1:k*block);
            img_dst_path = [dst_dir num2str(num, '%04d') '_' num2str(k) '.png'];
            img = [img_ori_tmp img_clear_tmp img_mask_tmp];
%                 img = repmat(img,1,1,3);
            imwrite(img, img_dst_path);
        end
        if(col ~= 100 * k)
            img_ori_tmp = img_ori(:, col-block+1:col);
            img_clear_tmp = img_clear(:, col-block+1:col);
            img_mask_tmp = img_mask(:, col-block+1:col);
            img_dst_path = [dst_dir num2str(num, '%04d') '_' num2str(k+1) '.png'];
            img = [img_ori_tmp img_clear_tmp img_mask_tmp];
%                 img = repmat(img,1,1,3);
            imwrite(img, img_dst_path);
        end
    end
end
