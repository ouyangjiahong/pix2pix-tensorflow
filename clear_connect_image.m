clear
num = 0;
% load annotation_list.mat
dst_dir = '/home/jihang/Jiahong/pix2pix-tensorflow/OCT_data/';
ori_dir = '/home/jihang/Jiahong/data/OCT/Annotations/';
ann_dir = '/home/jihang/Jiahong/data/OCT/Annotations/';

margin = 0;
low = 30;
high = 80;
block = 256

for j in ["25_new"]
    j = char(j)
    ori_path = [ori_dir num2str(j) '/original/']
    ann_path = [ann_dir num2str(j) '/originalAnnotationTejas_new/'];
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
        for j = 1 : col
            i = 1;
            while(i <= row - 5)
                if(img_ann(i+margin,j,1)==255 && img_ann(i+margin,j,2)==0 && img_ann(i+margin,j,3)==0)
                    break;
                else
                  %img_clear(i,j) = low+rand()*(high-low);
                  img_clear(i,j) = 0;
                end
                i = i + 1;
            end
        end
%         figure(2);
%         imshow(img_ori);

%         if(row>=1.5*col)
%             img_clear = img_clear(1 : row/2, :);
%             img_ori = img_ori(1 : row/2, :);
%         end
%         img_ori = imresize(img_ori, [256, 256]);
%         img_clear = imresize(img_clear, [256, 256]);
%         img = [img_ori img_clear];
%         figure(3);
%         imshow(img);

%         if(mod(num,5)==0)  
%             for k = 1 : floor(col/block)
%                 img_ori_tmp = img_ori(:, (k-1)*block+1:k*block);
%                 img_clear_tmp = img_clear(:, (k-1)*block+1:k*block);
%                 img_dst_path = [dst_dir 'test_random/' num2str(num, '%04d') '_' num2str(k) '.png'];
%                 img = [img_ori_tmp img_clear_tmp];
%                 img = repmat(img,1,1,3);
%                 imwrite(img, img_dst_path);
%             end
%             if(col ~= block * k)
%                 img_ori_tmp = img_ori(:, col-block+1:col);
%                 img_clear_tmp = img_clear(:, col-block+1:col);
%                 img_dst_path = [dst_dir 'test_random/' num2str(num, '%04d') '_' num2str(k+1) '.png'];
%                 img = [img_ori_tmp img_clear_tmp];
%                 img = repmat(img,1,1,3);
%                 imwrite(img, img_dst_path);
%             end
%             
%         else
            for k = 1 : floor(col/block)
                img_ori_tmp = img_ori(:, (k-1)*block+1:k*block);
                img_clear_tmp = img_clear(:, (k-1)*block+1:k*block);
                img_dst_path = [dst_dir 'test_testdata_clean_25/' num2str(num, '%04d') '_' num2str(k) '.png'];
                img = [img_ori_tmp img_clear_tmp];
%                 img = repmat(img,1,1,3);
                imwrite(img, img_dst_path);
            end
            if(col ~= 100 * k)
                img_ori_tmp = img_ori(:, col-block+1:col);
                img_clear_tmp = img_clear(:, col-block+1:col);
                img_dst_path = [dst_dir 'test_testdata_clean_25/' num2str(num, '%04d') '_' num2str(k+1) '.png'];
                img = [img_ori_tmp img_clear_tmp];
%                 img = repmat(img,1,1,3);
                imwrite(img, img_dst_path);
            end
%         end
    end
end
